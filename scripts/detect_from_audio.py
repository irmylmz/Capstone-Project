#!/usr/bin/env python3
"""Bir ses dosyasindan RD haritalari uretir ve egitilmis modelle her parcada hedef var/yok der.

Ses dosyasi sadece FM vericinin yayinidir; hedef, gozetim kanalina simulasyonla eklenir. Bu yuzden
her parca icin hem MODELIN KARARI hem de GERCEK (simulasyonda eklenen) durum yazdirilir.

Ornekler (proje kokunden):
    python scripts/detect_from_audio.py sarki.mp3                       # %50 rastgele hedef
    python scripts/detect_from_audio.py sarki.mp3 --target none         # hic hedef yok (yanlis alarm testi)
    python scripts/detect_from_audio.py sarki.mp3 --target always --range_km 40 --doppler_hz -80 --snr_db -28
    python scripts/detect_from_audio.py sarki.mp3 --segments 10 --save_plot sonuc.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from passive_radar.data.transforms import rd_map_to_tensor  # noqa: E402
from passive_radar.models.checkpoint import load_checkpoint  # noqa: E402
from passive_radar.simulation.audio_rd import TargetSpec, audio_to_rd_maps, load_sim_config  # noqa: E402
from passive_radar.training.runtime import resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sesten hedef var/yok tespiti (FM pasif radar simulasyonu + CNN)")
    ap.add_argument("audio", help="ses dosyasi (.mp3, .wav, ...)")
    ap.add_argument("--target", choices=["random", "always", "none"], default="random",
                    help="gozetim kanalina hedef ekosu: random (%%50), always, none")
    ap.add_argument("--range_km", type=float, default=None, help="hedef bistatik menzili [km] (bos: rastgele)")
    ap.add_argument("--doppler_hz", type=float, default=None, help="hedef Doppler [Hz] (bos: rastgele)")
    ap.add_argument("--snr_db", type=float, default=None, help="hedef giris SNR [dB] (bos: rastgele, -38..-20)")
    ap.add_argument("--segments", type=int, default=20, help="islenecek 0.5 s'lik parca sayisi (0 = tumu)")
    ap.add_argument("--start_s", type=float, default=0.0, help="baslangic saniyesi")
    ap.add_argument("--threshold", type=float, default=0.5, help="karar esigi (p_target >= esik -> hedef var)")
    ap.add_argument("--model", default=str(ROOT / "models" / "dataset_1000_best_model.pt"))
    ap.add_argument("--dataset", default=str(ROOT / "data" / "dataset_1000"),
                    help="simulasyon parametreleri bu klasorun config.json'undan okunur")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_plot", default=None, help="haritalari ve kararlari PNG olarak kaydet")
    a = ap.parse_args()

    cfg = load_sim_config(Path(a.dataset))
    device = resolve_device("auto")
    ckpt = load_checkpoint(Path(a.model), device=device)
    model, rd_config = ckpt.model.eval(), ckpt.preprocessing_config

    print(f"Ses: {a.audio}\nModel: {Path(a.model).name} | cihaz: {device} | esik: {a.threshold}")
    segs = audio_to_rd_maps(a.audio, cfg, target=a.target,
                            target_spec=TargetSpec(a.range_km, a.doppler_hz, a.snr_db),
                            max_segments=a.segments or None, start_s=a.start_s, seed=a.seed)
    if not segs:
        sys.exit("Islenecek (sessiz olmayan) parca bulunamadi.")

    with torch.no_grad():
        batch = torch.stack([rd_map_to_tensor(s.db_map, rd_config) for s in segs]).to(device)
        p_target = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()

    print(f"\n{'parca':>5} {'zaman [s]':>13} {'gercek':>8} {'menzil':>8} {'Doppler':>8} {'SNRout':>7} | "
          f"{'p_target':>8} {'karar':>10} {'':>6}")
    correct = 0
    for s, p in zip(segs, p_target):
        pred = p >= a.threshold
        ok = pred == s.has_target
        correct += ok
        truth = "HEDEF" if s.has_target else "yok"
        extra = (f"{s.range_km:6.1f}km {s.doppler_hz:+7.1f}Hz {s.snr_out_db:6.1f}" if s.has_target
                 else f"{'-':>8} {'-':>8} {'-':>7}")
        print(f"{s.index:5d} {s.t_start_s:6.1f}-{s.t_end_s:5.1f}  {truth:>8} {extra} | {p:8.3f} "
              f"{'HEDEF VAR' if pred else 'hedef yok':>10} {'✓' if ok else '✗ HATA'}")

    n, nt = len(segs), sum(s.has_target for s in segs)
    preds = p_target >= a.threshold
    truth = np.array([s.has_target for s in segs])
    print(f"\nToplam {n} parca | dogruluk {correct / n:.1%}", end="")
    if nt:
        print(f" | tespit {preds[truth].mean():.1%} ({nt} hedefli)", end="")
    if n - nt:
        print(f" | yanlis alarm {preds[~truth].mean():.1%} ({n - nt} hedefsiz)", end="")
    print()

    if a.save_plot:
        import matplotlib.pyplot as plt
        k = min(len(segs), 8)
        fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 4), squeeze=False)
        ext = (0, cfg["r_max"] * cfg["range_cell_m"] / 1e3, -cfg["fD_max"], cfg["fD_max"])
        for ax, s, p in zip(axes.flat, segs[:k], p_target[:k]):
            ax.imshow(np.clip(s.db_map.astype(np.float32), 0, 40), aspect="auto", extent=ext, cmap="jet")
            if s.has_target:
                ax.plot(s.range_km, s.doppler_hz, "o", ms=14, mfc="none", mec="white")
            ax.set_title(f"t={s.t_start_s:.1f}s gercek={'H' if s.has_target else '-'}\np={p:.2f} -> "
                         f"{'HEDEF' if p >= a.threshold else 'yok'}", fontsize=9)
            ax.set_xlabel("menzil [km]")
        axes.flat[0].set_ylabel("Doppler [Hz]")
        fig.tight_layout()
        fig.savefig(a.save_plot, dpi=110)
        print(f"Grafik: {a.save_plot}")


if __name__ == "__main__":
    main()
