"""Central, configurable settings for the passive radar target detection project.

Every tunable value lives here as a dataclass field so that notebooks, training
and inference share one source of truth instead of scattering magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = ("no_target", "target")
"""Class names in label-index order (index 0 = no target, index 1 = target)."""

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
DEFAULT_CHECKPOINT_NAME = "best_model.pt"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until a directory containing ``src/passive_radar`` is found.

    ``start`` defaults to this module's location, so the project root is found
    regardless of the current working directory.

    Raises:
        FileNotFoundError: If no project root exists above ``start``.
    """
    current = Path(start or __file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "passive_radar").is_dir():
            return candidate
    raise FileNotFoundError(f"Project root (containing src/passive_radar) not found above {current}")


@dataclass(frozen=True)
class PathsConfig:
    """File-system locations used by the project."""

    project_root: Path
    data_dir: Path
    labeled_dir: Path
    unlabeled_dir: Path
    models_dir: Path
    outputs_dir: Path
    audio_dir: Path
    simulated_dir: Path

    @classmethod
    def from_root(cls, project_root: Path) -> "PathsConfig":
        """Build the standard project layout below ``project_root``."""
        root = Path(project_root).resolve()
        data_dir = root / "data"
        return cls(
            project_root=root,
            data_dir=data_dir,
            labeled_dir=data_dir / "train",
            unlabeled_dir=data_dir / "test",
            models_dir=root / "models",
            outputs_dir=root / "outputs",
            audio_dir=root / "music_samples",
            simulated_dir=data_dir / "simulated",
        )

    @property
    def checkpoint_path(self) -> Path:
        """Default location of the best model checkpoint."""
        return self.models_dir / DEFAULT_CHECKPOINT_NAME

    @property
    def splits_csv(self) -> Path:
        """CSV file that stores the train/val/test assignment of every image."""
        return self.outputs_dir / "splits.csv"

    @property
    def simulated_outputs_dir(self) -> Path:
        """Outputs (splits, metrics, figures) of the simulated-data experiment."""
        return self.outputs_dir / "simulated"

    @property
    def simulated_checkpoint_path(self) -> Path:
        """Best model trained on simulated Range-Doppler maps."""
        return self.models_dir / "simulated_best_model.pt"

    @property
    def rd_dataset_500_dir(self) -> Path:
        """Labelled 500 target / 500 notarget RD dataset (from p_radar_sim/rd_dataset)."""
        return self.data_dir / "dataset_500"

    @property
    def rd_dataset_500_outputs_dir(self) -> Path:
        """Outputs (splits, metrics, figures) of the dataset_500 experiment."""
        return self.outputs_dir / "dataset_500"

    @property
    def rd_dataset_500_checkpoint_path(self) -> Path:
        """Best model trained on dataset_500."""
        return self.models_dir / "dataset500_best_model.pt"

    @property
    def normalization_json(self) -> Path:
        """JSON file with per-channel mean/std computed on the training split."""
        return self.outputs_dir / "normalization.json"


@dataclass(frozen=True)
class PreprocessingConfig:
    """How a raw Range-Doppler figure is turned into model input.

    The source PNGs are Plotly figures (2240 x 740 px) with axes, titles and a
    colour bar. ``plot_box`` is the pixel box ``(left, top, right, bottom)`` of the
    heat-map area in that reference layout; cropping to it removes only
    decoration, never Range-Doppler cells.
    """

    reference_size: tuple[int, int] = (2240, 740)  # (width, height)
    plot_box: tuple[int, int, int, int] = (160, 200, 2048, 580)
    target_size: tuple[int, int] = (640, 128)  # (width, height); keeps the ~4.97:1 plot aspect
    pad_value: int = 0
    resample: str = "lanczos"
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @property
    def num_channels(self) -> int:
        """Number of image channels fed to the network (RGB)."""
        return len(self.mean)


@dataclass(frozen=True)
class SplitConfig:
    """Group-aware split settings (groups = source recordings)."""

    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_holdout_group_fraction: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters of :class:`RangeDopplerCNN`."""

    in_channels: int = 3
    num_classes: int = len(CLASS_NAMES)
    conv_channels: tuple[int, ...] = (16, 32, 64, 128)
    kernel_size: int = 3
    pooled_size: tuple[int, int] = (2, 8)  # (height, width) after adaptive pooling
    global_pool: str = "avg"  # "avg" keeps texture statistics, "max" keeps the strongest local peak
    hidden_units: int = 128
    dropout: float = 0.4


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation and early-stopping settings."""

    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 30
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    num_workers: int = 0  # 0 is the safe default on macOS notebooks
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"
    seed: int = 42
    cache_images: bool = True


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class SimulationConfig:
    """FM passive radar scenario used to generate labelled Range-Doppler maps.

    The illuminator is an FM broadcast signal modulated by an audio recording.
    A ``target`` sample adds one delayed, Doppler-shifted, attenuated copy of the
    illuminator to the surveillance channel; a ``no_target`` sample does not.
    Both contain independent receiver noise.
    """

    carrier_frequency_hz: float = 100e6
    fm_deviation_hz: float = 75e3
    upsample_factor: int = 5  # 44.1 kHz audio -> 220.5 kHz complex baseband
    integration_time_s: float = 0.25
    max_range_km: float = 100.0
    max_speed_kmh: float = 1000.0  # sets the Doppler axis: +-2 v / lambda
    target_range_km: tuple[float, float] = (5.0, 95.0)
    target_doppler_margin_hz: float = 10.0  # keep targets away from the Doppler axis edges
    # Per-sample echo-to-noise ratio before integration. With 0.25 s at 220.5 kHz the coherent
    # processing gain is ~47 dB, so this spans barely visible (~2 dB) to obvious (~22 dB) targets.
    target_snr_db: tuple[float, float] = (-45.0, -25.0)
    segments_per_recording: int = 10
    seed: int = 7

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength."""
        return SPEED_OF_LIGHT_M_S / self.carrier_frequency_hz

    @property
    def max_doppler_hz(self) -> float:
        """Largest bistatic Doppler frequency shown on the map."""
        return 2 * (self.max_speed_kmh / 3.6) / self.wavelength_m


@dataclass(frozen=True)
class RangeDopplerMapConfig:
    """Conversion of a complex Range-Doppler map into network input."""

    dynamic_range_db: float = 40.0  # values below (median floor) are clipped, above floor+range saturate

    @property
    def num_channels(self) -> int:
        """Range-Doppler maps are single-channel."""
        return 1


@dataclass(frozen=True)
class ProjectConfig:
    """Bundle of all configuration sections."""

    paths: PathsConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    rd_map: RangeDopplerMapConfig = field(default_factory=RangeDopplerMapConfig)

    @classmethod
    def default(cls, project_root: Path | None = None) -> "ProjectConfig":
        """Create the default configuration for the project found at/above ``project_root``."""
        root = project_root if project_root is not None else find_project_root()
        return cls(paths=PathsConfig.from_root(root))
