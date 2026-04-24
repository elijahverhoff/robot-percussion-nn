# Beat Tracking with a Causal Temporal Convolutional Network

Course project for ECE 490: Neural Networks. Trains a causal TCN for
real-time beat tracking on audio, intended to drive a mechanical
actuator for synchronized percussion. Progress-report submission.

**[Progress report (PDF)](reports/progress-report.pdf)**

## Results

- Test F-measure: 0.855 (Ballroom: 0.908, GTZAN: 0.825)
- Non-causal baseline: 0.873 F-measure (+1.8 point cost of real-time
  constraint)
- Activation timing: +0.5 ms mean, 15 ms jitter

Full results, figures, and analysis are in the attached progress
report PDF and in the notebook.

## Reproducing

The full pipeline is in `beat-tracking-tcn.ipynb`, designed to run
in Google Colab with GPU runtime.

### Data

Datasets are not included in this repository. To re-run training:

- Ballroom: download `data1.tar.gz` from the ISMIR 2004 tempo contest
  distribution (mtg.upf.edu).
- GTZAN: available via Kaggle ("GTZAN Dataset - Music Genre
  Classification").
- Beat annotations:
  [CPJKU/BallroomAnnotations](https://github.com/CPJKU/BallroomAnnotations)
  and
  [TempoBeatDownbeat/gtzan_tempo_beat](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat).

Place the archives in a Drive folder and adjust `DRIVE_DATASETS` in
Cell 1 to point to it. The notebook's Section 1 handles extraction
and annotation loading automatically.

### Notebook sections

1. Setup — mount Drive, extract archives, load annotations.
2. Dataset and caching — compute log-mel features, build PyTorch
   Dataset, train/val/test split.
3. Model — causal TCN definition.
4. Training — 30-epoch runs for v1, v2, and non-causal ablation.
5. Evaluation — metrics, per-genre breakdown, latency analysis,
   qualitative examples.

### Runtime

- Extraction + cache restore: ~5 minutes.
- Training (one 30-epoch run): ~4 minutes on a T4 GPU.
- Evaluation: ~1 minute.

## Remaining work

Real-time inference pipeline and hardware integration with a
mechanical actuator, scheduled for the final presentation.