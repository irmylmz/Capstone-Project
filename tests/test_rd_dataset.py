"""Tests for the dataset_500 index (synthetic folder in tmp_path)."""

import csv

import numpy as np
import pytest

from passive_radar.data.rd_dataset import load_rd_samples, samples_to_records

FIELDS = ["file", "class_dir", "source_group", "source_file", "segment", "t_start_s", "t_end_s", "audio_rms",
          "has_target", "range_cell", "doppler_bin", "delay_samples", "doppler_hz", "bistatic_range_km",
          "snr_in_db", "snr_out_db", "visible"]


def _make(tmp_path, rows):
    for cls in ("target", "notarget"):
        (tmp_path / cls / "rd_npy").mkdir(parents=True)
    with open(tmp_path / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in FIELDS} | r
            w.writerow(full)
            np.save(tmp_path / r["class_dir"] / "rd_npy" / f"{r['file']}.npy", np.zeros((5, 4), np.float16))


def _row(name, label, src):
    base = {"file": name, "class_dir": "target" if label else "notarget", "source_group": "g",
            "source_file": src, "t_start_s": "0", "t_end_s": "0.5", "has_target": str(label)}
    if label:
        base |= {"range_cell": "3", "doppler_bin": "2", "doppler_hz": "-12.5", "bistatic_range_km": "4.1",
                 "snr_out_db": "15.0", "visible": "1"}
    return base


def test_load_and_records(tmp_path):
    _make(tmp_path, [_row("a", 1, "f1"), _row("b", 0, "f1"), _row("c", 0, "f2")])
    samples = load_rd_samples(tmp_path)
    assert [s.label for s in samples] == [1, 0, 0]
    assert samples[0].visible is True and samples[1].visible is None
    records = samples_to_records(samples, tmp_path)
    assert records[0].path.name == "a.npy" and records[0].path.parent.parent.name == "target"
    assert {r.source_id for r in records} == {"f1", "f2"}


def test_inconsistent_class_dir_rejected(tmp_path):
    bad = _row("a", 1, "f1") | {"class_dir": "notarget"}
    _make(tmp_path, [bad])
    with pytest.raises(ValueError):
        load_rd_samples(tmp_path)


def test_doppler_augment_preserves_shape_and_values():
    from passive_radar.data.transforms import doppler_augment

    rng = np.random.default_rng(0)
    m = np.zeros((11, 4), np.float32)
    m[2, 1] = 30.0  # one "target" cell
    for _ in range(20):
        out = doppler_augment(m, rng, max_shift=3)
        assert out.shape == m.shape
        assert np.count_nonzero(out) == 1 and out.max() == 30.0
        assert np.argwhere(out)[0][1] == 1  # range cell unchanged
