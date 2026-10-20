"""Generate a labelled dataset of simulated Range-Doppler maps from audio recordings.

For every audio segment one ``target`` and one ``no_target`` map are produced from
the *same* illuminator, so the only systematic difference between the classes is
the target echo (not the audio content).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from passive_radar.config import CLASS_NAMES, SimulationConfig
from passive_radar.data.indexing import ImageRecord, normalize_source_id
from passive_radar.simulation.ambiguity import cross_ambiguity
from passive_radar.simulation.fm import AUDIO_EXTENSIONS, AudioLoadError, fm_signal_from_audio, load_audio_mono
from passive_radar.simulation.scenario import TargetParameters, range_cells, sample_target, synthesize_channels

METADATA_FILE = "metadata.csv"
MAPS_SUBDIR = "maps"
_DB_FLOOR = 1e-12


@dataclass(frozen=True)
class SimulatedSample:
    """Metadata row describing one generated Range-Doppler map."""

    path: str  # relative to the dataset directory
    label: int
    source_id: str
    recording: str
    segment_start_s: float
    target_range_km: float | None
    target_doppler_hz: float | None
    target_snr_db: float | None


def list_audio_files(directory: Path) -> list[Path]:
    """Audio files directly inside ``directory`` (sorted, hidden files ignored).

    Raises:
        FileNotFoundError: If ``directory`` does not exist or holds no audio.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) in {directory}")
    return files


def magnitude_db(rd_map: np.ndarray) -> np.ndarray:
    """|map| in dB as ``float32`` (what is stored on disk)."""
    return (20 * np.log10(np.abs(rd_map) + _DB_FLOOR)).astype(np.float32)


def segment_starts(audio_length: int, window: int, count: int, rng: np.random.Generator) -> list[int]:
    """Up to ``count`` non-overlapping window start indices spread over the recording."""
    slots = audio_length // window
    if slots == 0:
        return []
    chosen = rng.choice(slots, size=min(count, slots), replace=False)
    return sorted(int(slot) * window for slot in chosen)


@dataclass(frozen=True)
class _SignalLengths:
    """Sample counts derived from the configuration for one audio sample rate."""

    num_samples: int  # coherent integration length (baseband samples)
    history: int  # extra leading samples needed for the longest echo delay
    max_range_cells: int
    audio_window: int  # audio samples consumed per segment


def _signal_lengths(config: SimulationConfig, audio_rate: int) -> _SignalLengths:
    sample_rate = audio_rate * config.upsample_factor
    max_cells = range_cells(config.max_range_km, sample_rate)
    num_samples = int(config.integration_time_s * sample_rate)
    audio_window = int(np.ceil((num_samples + max_cells) / config.upsample_factor)) + 1
    return _SignalLengths(num_samples, max_cells, max_cells, audio_window)


def simulate_pair(
    audio: np.ndarray, audio_rate: int, start: int, config: SimulationConfig, rng: np.random.Generator
) -> tuple[dict[int, np.ndarray], TargetParameters]:
    """Return {label: dB map} for one audio segment plus the target parameters."""
    lengths = _signal_lengths(config, audio_rate)
    num_samples, history, max_cells = lengths.num_samples, lengths.history, lengths.max_range_cells
    illuminator, sample_rate = fm_signal_from_audio(
        audio, audio_rate, start, num_samples + history, config.upsample_factor, config.fm_deviation_hz
    )
    target = sample_target(rng, config, sample_rate)
    maps = {}
    for label, echo in ((CLASS_NAMES.index("target"), target), (CLASS_NAMES.index("no_target"), None)):
        reference, surveillance = synthesize_channels(illuminator, history, sample_rate, echo, rng)
        maps[label] = magnitude_db(
            cross_ambiguity(reference, surveillance, sample_rate, config.max_doppler_hz, max_cells)
        )
    return maps, target


def generate_dataset(
    audio_files: Sequence[Path],
    output_dir: Path,
    config: SimulationConfig,
    progress: Callable[[str], None] | None = print,
) -> list[SimulatedSample]:
    """Simulate maps for every audio file, save them as ``.npy`` and write ``metadata.csv``.

    Recordings that cannot be decoded or are too short are skipped (and reported).
    """
    output_dir = Path(output_dir)
    (output_dir / MAPS_SUBDIR).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    samples: list[SimulatedSample] = []

    for file_index, audio_path in enumerate(audio_files):
        try:
            audio, audio_rate = load_audio_mono(audio_path)
        except AudioLoadError as error:
            if progress:
                progress(f"skip {audio_path.name}: {error}")
            continue
        window = _signal_lengths(config, audio_rate).audio_window
        for segment_index, start in enumerate(segment_starts(audio.size, window, config.segments_per_recording, rng)):
            if np.max(np.abs(audio[start:start + window])) == 0:
                continue  # silent segment: FM signal undefined
            maps, target = simulate_pair(audio, audio_rate, start, config, rng)
            for label, db_map in maps.items():
                relative = f"{MAPS_SUBDIR}/{file_index:04d}_{segment_index:03d}_{CLASS_NAMES[label]}.npy"
                np.save(output_dir / relative, db_map)
                has_target = label == CLASS_NAMES.index("target")
                samples.append(SimulatedSample(
                    path=relative,
                    label=label,
                    source_id=normalize_source_id(audio_path.name),
                    recording=audio_path.name,
                    segment_start_s=round(start / audio_rate, 3),
                    target_range_km=round(target.bistatic_range_km, 3) if has_target else None,
                    target_doppler_hz=round(target.doppler_hz, 3) if has_target else None,
                    target_snr_db=round(target.snr_db, 2) if has_target else None,
                ))
        if progress:
            progress(f"[{file_index + 1}/{len(audio_files)}] {audio_path.name}: {len(samples)} maps so far")

    write_metadata(samples, output_dir / METADATA_FILE)
    return samples


def write_metadata(samples: Sequence[SimulatedSample], csv_path: Path) -> None:
    """Write sample metadata as CSV."""
    fields = list(SimulatedSample.__dataclass_fields__)
    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def load_simulated_records(dataset_dir: Path) -> list[ImageRecord]:
    """Read ``metadata.csv`` into records usable by the splitting and dataset code.

    Raises:
        FileNotFoundError: If the dataset has not been generated yet.
    """
    csv_path = Path(dataset_dir) / METADATA_FILE
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found. Run notebooks/05_simulate_dataset.ipynb first.")
    with csv_path.open(encoding="utf-8") as handle:
        return [
            ImageRecord(path=Path(dataset_dir) / row["path"], label=int(row["label"]), source_id=row["source_id"])
            for row in csv.DictReader(handle)
        ]
