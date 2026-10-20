"""Tests for dataset indexing, splitting and the PyTorch dataset."""

from __future__ import annotations

import shutil
import unicodedata
from collections import Counter

import pytest
import torch

from passive_radar.config import CLASS_NAMES, SplitConfig
from passive_radar.data.dataset import RangeDopplerDataset, make_data_loader
from passive_radar.data.indexing import (
    SPLIT_NAMES,
    ImageRecord,
    assert_no_group_leakage,
    deduplicate_records,
    load_split_csv,
    normalize_source_id,
    records_in_split,
    save_split_csv,
    scan_labeled_directory,
    scan_unlabeled_directory,
    split_by_source,
)
from passive_radar.data.preprocessing import ImageLoadError


@pytest.mark.parametrize(
    "filename",
    [
        "Beni Hatırla (128kbit_AAC)Part002.png",
        "Beni Hatırla (128kbit_AAC)Part017_range_doppler.png",
        "Beni Hatırla (128kbit_AAC)Part002(1).png",
        "Beni_Hatırla_128kbit_AAC_Part005.png",
    ],
)
def test_source_id_ignores_segment_and_suffixes(filename):
    assert normalize_source_id(filename) == "beni hatırla"


def test_source_id_unicode_normalisation():
    composed = "Çalışma (128kbit_AAC)Part001.png"
    decomposed = unicodedata.normalize("NFD", composed)
    assert normalize_source_id(composed) == normalize_source_id(decomposed)


def test_scan_returns_correct_labels(labeled_dir):
    records = scan_labeled_directory(labeled_dir)
    assert len(records) == 32
    for record in records:
        assert CLASS_NAMES[record.label] == record.path.parent.name
    assert Counter(record.label for record in records) == {0: 16, 1: 16}


def test_scan_ignores_hidden_and_non_image_files(labeled_dir):
    (labeled_dir / "target" / ".DS_Store").write_bytes(b"x")
    (labeled_dir / "target" / "notes.txt").write_text("x")
    assert len(scan_labeled_directory(labeled_dir)) == 32


def test_scan_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_labeled_directory(tmp_path / "nope")


def test_scan_missing_class_folder_raises(labeled_dir):
    shutil.rmtree(labeled_dir / "target")
    with pytest.raises(FileNotFoundError, match="target"):
        scan_labeled_directory(labeled_dir)


def test_scan_unlabeled(labeled_dir):
    records = scan_unlabeled_directory(labeled_dir / "target")
    assert len(records) == 16 and all(record.label is None for record in records)


def test_deduplicate_removes_copies_and_conflicts(tmp_path):
    def record(name, label, md5):
        return ImageRecord(path=tmp_path / name, label=label, source_id="s", md5=md5)

    records = [record("a.png", 1, "h1"), record("a(1).png", 1, "h1"), record("b.png", 0, "h2"),
               record("c.png", 0, "h3"), record("c_copy.png", 1, "h3")]
    kept, report = deduplicate_records(records)
    assert [r.path.name for r in kept] == ["a.png", "b.png"]
    assert report.duplicate_copies_removed == 1
    assert report.conflicting_label_images_removed == 2


def test_split_has_no_recording_leakage(labeled_dir):
    records = scan_labeled_directory(labeled_dir)
    config = SplitConfig(val_fraction=0.25, test_fraction=0.25, max_holdout_group_fraction=0.5, seed=0)
    assigned = split_by_source(records, config)
    assert len(assigned) == len(records)
    assert {record.split for record in assigned} == set(SPLIT_NAMES)
    assert_no_group_leakage(assigned)
    for split in SPLIT_NAMES:
        assert {r.label for r in records_in_split(assigned, split)} == {0, 1}


def _synthetic_records(tmp_path, big_size: int, small_sources: int) -> list[ImageRecord]:
    records = [ImageRecord(tmp_path / f"big{i}.png", i % 2, "big recording") for i in range(big_size)]
    for source in range(small_sources):
        records += [ImageRecord(tmp_path / f"s{source}_{label}.png", label, f"small {source}") for label in (0, 1)]
    return records


def test_large_recordings_are_forced_into_train(tmp_path):
    records = _synthetic_records(tmp_path, big_size=20, small_sources=20)
    config = SplitConfig(val_fraction=0.2, test_fraction=0.2, max_holdout_group_fraction=0.2, seed=0)
    assigned = split_by_source(records, config)
    assert {r.split for r in assigned if r.source_id == "big recording"} == {"train"}


def test_split_without_holdout_candidates_raises(tmp_path):
    records = _synthetic_records(tmp_path, big_size=20, small_sources=0)
    with pytest.raises(ValueError, match="hold-out"):
        split_by_source(records, SplitConfig())


def test_leakage_check_detects_shared_recording(tmp_path):
    records = [ImageRecord(tmp_path / "a.png", 0, "song", split="train"),
               ImageRecord(tmp_path / "b.png", 1, "song", split="test")]
    with pytest.raises(ValueError, match="multiple splits"):
        assert_no_group_leakage(records)


def test_split_csv_round_trip(labeled_dir, tmp_path):
    config = SplitConfig(val_fraction=0.25, test_fraction=0.25, max_holdout_group_fraction=0.5, seed=0)
    assigned = split_by_source(scan_labeled_directory(labeled_dir), config)
    csv_path = tmp_path / "splits.csv"
    save_split_csv(assigned, csv_path, relative_to=tmp_path)
    assert load_split_csv(csv_path, relative_to=tmp_path) == assigned


def test_load_missing_split_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_preprocessing"):
        load_split_csv(tmp_path / "missing.csv", relative_to=tmp_path)


def test_torch_dataset_returns_tensor_and_label(labeled_dir, preprocessing_config):
    records = scan_labeled_directory(labeled_dir)
    dataset = RangeDopplerDataset(records, preprocessing_config, cache_images=True)
    image, label = dataset[len(dataset) - 1]
    assert image.shape == (3, 128, 640)
    assert label == records[-1].label == 1
    assert dataset.labels == [record.label for record in records]


def test_data_loader_batches(labeled_dir, preprocessing_config):
    dataset = RangeDopplerDataset(scan_labeled_directory(labeled_dir), preprocessing_config)
    images, labels = next(iter(make_data_loader(dataset, batch_size=4, shuffle=True, seed=1)))
    assert images.shape == (4, 3, 128, 640)
    assert labels.dtype == torch.int64


def test_torch_dataset_rejects_unlabeled(labeled_dir, preprocessing_config):
    with pytest.raises(ValueError):
        RangeDopplerDataset(scan_unlabeled_directory(labeled_dir / "target"), preprocessing_config)


def test_corrupt_file_in_dataset_raises_clear_error(corrupt_path, preprocessing_config):
    record = ImageRecord(path=corrupt_path, label=0, source_id="x")
    with pytest.raises(ImageLoadError):
        RangeDopplerDataset([record], preprocessing_config)[0]
