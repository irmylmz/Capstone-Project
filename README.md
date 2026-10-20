# Capstone-Project

## CNN-Based Passive Radar Target Detection

This project classifies passive radar **Range-Doppler** images into two classes, `target` and `no_target`, using a PyTorch CNN.
Reusable logic lives in `src/passive_radar/`. Analysis, experiments and reporting are done in the Jupyter notebooks under `notebooks/`.

> **Status:**
> * The image model (`data/train`) has been trained and evaluated (results below).
> * **Important finding:** the simulation notebooks that produced the images call `cc_detector_fd(s, s)`, i.e. the surveillance channel is identical to the reference. **No physical target echo was added to these images.** The physical meaning of the `target`/`no_target` labels could therefore not be verified.
> * To address this, a **simulated dataset with physically defined labels** was added (notebooks 05–06, `src/passive_radar/simulation/`). That code has been checked in the sandbox; it has not yet been run on the Mac.

---

## Dataset Summary (`data/`)

| | Value |
|---|---|
| `data/train/target` | 510 PNG |
| `data/train/no_target` | 510 PNG |
| `data/test` | 40 PNG, **unlabeled** |
| Size / mode | All images are 2240 × 740 RGBA. No corrupt files. |
| Content | Plotly figures: axes, labels, colorbar. Heat-map area: Bistatic range 0–100 km × Doppler ±180 Hz (jet colormap) |
| Label meaning | `target` = a simulated target echo was added to that segment (the audio recording is used as the illuminator) |

Key findings:

* **Duplicates:** there are 19 byte-identical copies inside `train`, mostly files with `(1)`/`(2)` suffixes. The files are not deleted; they are only dropped from the index.
* **`data/test` is not independent:** all 40 images come from a single recording (Sam Altman TED2025). That recording also appears in `train`, and 16 of the images are byte-identical to images in `train/no_target`.
  For this reason `data/test` is used only as an **unlabeled prediction demo**.
* **Recording leakage:** each image is a segment of an audio recording (`...PartNNN.png`), and 86 of the 193 recordings contain both classes.
  Splitting is therefore done **by recording**, not by image.
  Very large recordings that hold more than 5% of the data (white noise with ~170 images, Eric Schmidt with 64) are always placed in train.
* Resulting split (seed 42): train 779 / val 111 / test 111 images (147 / 18 / 27 recordings). The classes are balanced in every split.
* `_range_doppler` and `(1)` file-name suffixes are correlated with the class. The model never sees file names, but this suggests that two different generation pipelines were used.

## Preprocessing

1. Load the image safely. Missing or corrupt files raise `ImageLoadError`. RGBA is converted to RGB.
2. **Heat-map crop:** `plot_box=(160, 200, 2048, 580)`. Only the axes, labels and colorbar are removed; no Range-Doppler cells are lost. The crop box was verified on all 1060 images.
3. **Aspect-preserving resize:** the heat-map has a 4.97:1 aspect ratio.
   * 512 × 128 (4:1) would either distort the image by 24% or spend ~20% of the height on padding.
   * The default is therefore **640 × 128** (~2 px of letterbox padding). It can be changed in `config.py`.
4. Per-channel mean/std normalization computed **on the train split only**.

Flips and rotations are not used, because the range and Doppler axes carry physical meaning.
The same `PreprocessingConfig`, including the normalization values, is stored in the checkpoint and reused at inference time.

## Model and Training

* `RangeDopplerCNN`:
  * 4 × [Conv3×3 → BatchNorm → ReLU → MaxPool], with channels 16-32-64-128
  * AdaptiveAvgPool (2×8)
  * Dropout → FC(128) → ReLU → Dropout → FC(2)
* Training: CrossEntropyLoss, Adam with lr=1e-3, batch size 16, up to 30 epochs, early stopping with patience 5 on **validation loss**.
* The checkpoint (`models/best_model.pt`) contains the weights, the class names, the model configuration and the preprocessing configuration. It is loaded with `torch.load(weights_only=True)`.
* The test split is not loaded in notebook 03. It is used only in notebook 04.
* Device selection: CUDA → **MPS (Apple Silicon)** → CPU.

## Project Structure

```
Capstone-Project/
├── data/                       # provided dataset (never modified)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb    # writes outputs/splits.csv and outputs/normalization.json
│   ├── 03_model_training.ipynb   # writes models/best_model.pt and outputs/training_history.json
│   ├── 04_evaluation.ipynb       # writes outputs/test_metrics.json and outputs/figures/*
│   ├── 05_simulate_dataset.ipynb # writes data/simulated/ (maps/*.npy + metadata.csv)
│   └── 06_simulated_training_evaluation.ipynb  # writes models/simulated_best_model.pt and outputs/simulated/
├── src/passive_radar/
│   ├── config.py                 # all configurable values (dataclasses)
│   ├── data/
│   │   ├── indexing.py           # scanning, source_id, deduplication, group split (torch-free)
│   │   ├── preprocessing.py      # loading, crop, letterbox, normalization stats (torch-free)
│   │   ├── transforms.py         # numpy → normalized tensor
│   │   └── dataset.py            # RangeDopplerDataset, DataLoader
│   ├── models/
│   │   ├── cnn.py
│   │   └── checkpoint.py
│   ├── training/
│   │   ├── trainer.py            # Trainer, EarlyStopping, TrainingHistory
│   │   └── runtime.py            # device, seed
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── simulation/               # FM IoO, target scenario, cross-ambiguity, dataset generation
│   │   ├── fm.py  scenario.py  ambiguity.py  dataset_builder.py
│   └── inference/
│       └── predictor.py          # RangeDopplerPredictor, predict_image
├── tests/                      # pytest, synthetic images only (tmp_path)
├── scripts/run_pipeline.sh     # pytest + notebooks 01→04
├── models/  outputs/
├── requirements.txt  pyproject.toml  .gitignore
```

## Setup (macOS / Apple Silicon)

The existing environment is `~/Desktop/venv`:

* Python 3.12.7 (Homebrew)
* torch 2.5.1, numpy 2.1.3, pillow 11.0, scikit-learn 1.6.1, matplotlib 3.9.2, pandas 2.2.3, jupyter

The only missing package is **pytest**:

```bash
~/Desktop/venv/bin/pip install "pytest>=8"
```

For a fresh environment, use `python3.12 -m venv venv && venv/bin/pip install -r requirements.txt`.

## Running

From the project root:

```bash
cd ~/Desktop/Projects/Capstone_Project/Capstone-Project

# Tests + notebooks 01→04 (image dataset) — logs: outputs/logs/
bash scripts/run_pipeline.sh

# Tests + notebooks 05→06 (simulated dataset; requires music_samples/)
bash scripts/run_pipeline.sh sim

# Or step by step
~/Desktop/venv/bin/python -m pytest -v
~/Desktop/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
~/Desktop/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
~/Desktop/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_training.ipynb
~/Desktop/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

Run the notebooks in **order**: 02 produces the split and the normalization statistics, 03 trains the model, and 04 evaluates it.
To work interactively, use `~/Desktop/venv/bin/jupyter lab`.

### Single-Image Prediction

```python
import sys; sys.path.insert(0, "src")   # or: pip install -e .
from passive_radar.inference.predictor import predict_image

result = predict_image("data/test/<file>.png")
print(result)              # file: target (no_target=0.12, target=0.88)
result.probabilities       # {'no_target': ..., 'target': ...}
```

If there is no model file, `ModelNotFoundError` is raised; if the image is missing or corrupt, `ImageLoadError` is raised.
Only a target present/absent decision is produced. **No range, velocity or position is estimated.**

## Results

### Image model (`data/train`, first run on the Mac, before the split fix)

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Val (102) | 0.63 | 0.64 | 0.87 | 0.74 | 0.62 |
| Test (97) | 0.70 | 0.70 | 0.92 | 0.79 | 0.68 |

* The model is strongly biased towards "target": only 12 of 36 no_target images were classified correctly.
* An AUC of 0.6–0.7 indicates weak separation, which is consistent with the labels having no physical counterpart in the images.
* On the Mac the split came out as 802/102/97 instead of 779/111/111, because of Unicode (NFD) file-name ordering. This has been fixed; the split is now identical on every operating system. The notebooks need to be re-run.

### Simulated model (notebooks 05–06)

*Not run yet.* After it runs, the results are written to `outputs/simulated/`:

* `test_metrics.json`
* `figures/detection_vs_snr.png`

80% accuracy can be a target, but it is not guaranteed.

## Simulated FM Passive Radar Dataset

* **Illuminator:** an FM signal modulated with an audio recording, as complex baseband (75 kHz deviation, 44.1 kHz × 5 = 220.5 kHz sampling).
  The original simulator used a real-valued `sin`, which made the ±Doppler sign ambiguous; the complex form avoids this.
* **target:** surveillance channel = noise + a copy of the illuminator that is delayed (bistatic range 5–95 km), Doppler-shifted (±175 Hz, about ±945 km/h) and attenuated (SNR −45…−25 dB per sample, which is roughly +2…+22 dB after the ~47 dB processing gain).
* **no_target:** noise only.
* Both classes are produced from **the same illuminator**, so the only systematic difference between them is the echo.
* Range-Doppler map: the same algorithm as pyAPRiL `cc_detector_fd`, implemented in NumPy. A test checks the result against pyAPRiL when it is installed.
  Each map is 185 Doppler bins × 73 range cells, stored as a dB `.npy` file.
* The ground truth (range, Doppler, SNR) is stored in `data/simulated/metadata.csv`.
* Model input: 1 channel; values are taken relative to the median noise floor and clipped to a 40 dB range. The CNN uses global **max** pooling, which is suitable for detecting a single peak.
* Assumptions:
  * ideal reference channel
  * no direct-path interference and no clutter
  * a single target

  Possible extensions: direct path + clutter with ECA clutter cancellation, and multiple targets.

The recordings are expected in `music_samples/` at the project root. A different folder can be set via `AUDIO_DIR` in notebook 05.

## Limitations

* The data is simulated. Generalization to real passive radar measurements has not been verified.
* The model learns from colormap images. If the colorbar range is set automatically for each image, absolute power information is lost.
  If the raw Range-Doppler matrices are available, working directly with them is recommended.
* The test split is small (~110 images, 27 recordings), so the uncertainty of the metrics is high.
* Training on MPS may not be fully deterministic.
* The earlier `RangeDoppler_Model.ipynb` (in `~/Desktop/venv`) selected the model on the test set and used flip/rotation augmentation.
  This project fixes both issues.

---

## dataset_500 (labelled RD maps from `p_radar_sim/rd_dataset`)

* `data/dataset_500/`: **500 target + 500 notarget** Range-Doppler maps (371 Doppler × 73 range, ±185 Hz, 0–99 km), copied from
  `../p_radar_sim/rd_dataset/dataset_500`. Generated by `gen_rd_dataset.py`: FM illuminator from 1-minute audio files,
  direct path + static clutter + ECA cancellation + noise; `target` maps contain an echo with a known delay/Doppler.
* Folders: `target/` and `notarget/`, each with `images/` (PNG), `masks/` (label mask PNG) and `rd_npy/` (dB, float16).
  `meta.csv` holds the labels (range cell, Doppler, km, SNR, `visible`), `kaynaklar.csv` the source audio file and time window of every map.
* Code: `src/passive_radar/data/rd_dataset.py` (index → `ImageRecord`), test: `tests/test_rd_dataset.py`.
* Notebook: `notebooks/07_dataset500_training_evaluation.ipynb`. Split by audio file (no file in two splits):
  train 713 / val 144 / test 143. Outputs: `models/dataset500_best_model.pt`, `outputs/dataset_500/`.

```bash
bash scripts/run_pipeline.sh rd500
```

### dataset_1000 (improved run, notebook 08)

* `data/dataset_1000/`: **1000 target + 1000 notarget**, 195 audio files from all 14 folders, stronger target SNR
  (input −38…−20 dB, 747 of 1000 targets visible).
* Notebook `08_dataset1000_training_evaluation.ipynb`: Doppler mirror/shift augmentation on train only, weight decay 1e-4,
  decision threshold chosen on **val** for a 10 % false-alarm rate, then reported on test.
  First run on dataset_500 (notebook 07): test false-alarm rate 0.40, visible-target detection 0.93.

```bash
bash scripts/run_pipeline.sh rd1000
```

### Sesten hedef tespiti (tek komut)

`scripts/detect_from_audio.py` bir ses dosyasını 0.5 s'lik parçalara böler, her parça için dataset ile **aynı** simülasyonla
RD haritası üretir (`src/passive_radar/simulation/audio_rd.py`, parametreler `data/dataset_1000/config.json`'dan) ve
`models/dataset_1000_best_model.pt` ile hedef var/yok kararı verir. Ses dosyası yalnızca FM yayınıdır; hedef gözetim kanalına
simülasyonla eklenir, bu yüzden her parça için modelin kararı ile gerçek durum yan yana yazdırılır.

```bash
python scripts/detect_from_audio.py sarki.mp3                     # %50 rastgele hedef
python scripts/detect_from_audio.py sarki.mp3 --target none       # hiç hedef yok
python scripts/detect_from_audio.py sarki.mp3 --target always --range_km 40 --doppler_hz -80 --snr_db -28 --save_plot sonuc.png
```
