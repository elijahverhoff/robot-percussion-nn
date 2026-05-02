# Beat Tracking with a Causal Temporal Convolutional Network

Course project for ECE 490/590: Neural Networks. Trains a causal TCN for real-time
beat tracking on audio, paired with an Arduino-driven servo that strikes a target
in time with music. Final-report submission.

**[Final report (PDF)](reports/final-report.pdf)**
**[Progress report (PDF)](reports/progress-report.pdf)**
****[Demo video](https://drive.google.com/file/d/10GNSU-em0evfEN7q8_hzUeWL1ydSq0uN/view?usp=sharing)****
****[Video presentation](https://drive.google.com/file/d/1qecCG8x6ITU950NMUBe_pkGRXNBVxNBn/view?usp=sharing)****

## Results

- Test F-measure: 0.855 (Ballroom: 0.908, GTZAN: 0.825)
- Non-causal baseline: 0.873 F (+1.8 point cost of real-time constraint)
- Activation timing: +0.5 ms mean, 15 ms jitter
- Per-genre range: 0.96 (reggae) to 0.59 (classical)
- End-to-end robotic synchronization with measured ~93 ms latency budget
  (78 ms servo + 15 ms model jitter), well below human just-noticeable-difference

Full methodology, figures, and analysis are in the final report.

## Repository structure

```
.
├── beat-tracking-tcn.ipynb        Main notebook (training + evaluation)
├── figures/                       Evaluation figures referenced in report
├── reports/                       Final and progress report PDFs
├── strikerServo/                  Arduino sketch for servo control
│   └── strikerServo.ino
├── trackingDemo/                  Robotics demo runtime
│   ├── extract_beats.py           Run model on audio → JSON of beat times
│   ├── run_demo.py                Play audio + send strikes to Arduino
│   └── songs/                     Beat-time JSONs for demo songs (audio
│                                  not included, see "Demo songs" below)
├── requirements.txt               Python package dependencies
├── README.md
└── .gitignore
```

## Reproducing

The project has two reproducible pieces: the **model training and evaluation**
(in the notebook, runs in Google Colab) and the **robotics demo** (local Python and Arduino, runs on a laptop with the trained model).

### Part 1: Model training and evaluation

The full ML pipeline is implemented in `beat-tracking-tcn.ipynb` and runs
end-to-end in Google Colab with a GPU runtime (tested on T4). Datasets are not
distributed with this repository; see "Data acquisition" below for sourcing
instructions.

#### Environment

- Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU).
- Google Drive mounted for persistent storage of datasets, cached features,
  and model checkpoints.
- Python 3.12 (Colab default as of April 2026).

Package versions are pinned loosely in `requirements.txt`; Colab's
preinstalled environment satisfies most dependencies, with only `mir_eval`
needing installation (handled automatically by the notebook).

#### Data acquisition

Two audio datasets and two annotation repositories are required.

**Ballroom audio** (698 tracks, ~30s each, 8 ballroom dance genres).
Distributed by the Music Technology Group at Universitat Pompeu Fabra as
part of the ISMIR 2004 tempo contest:
- Source: `http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`
- Citation: Gouyon et al., "An experimental comparison of audio tempo
  induction algorithms," IEEE TASLP, 2006.

**GTZAN audio** (1000 tracks, 30s each, 10 popular-music genres).
Originally from Tzanetakis & Cook 2002. Commonly redistributed; the version
used here was sourced from Kaggle ("GTZAN Dataset - Music Genre
Classification"). One track (`jazz.00054.wav`) is known to be corrupted and
is automatically excluded by the notebook.
- Source: `https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download`
- Citation: Tzanetakis & Cook, "Musical genre classification of audio
  signals," IEEE TSAP, 2002.
- Known issues: Sturm, "The GTZAN dataset: Its contents, its faults, their
  effects on evaluation, and its future use," 2013.

**Ballroom beat annotations.** Two-column format (time, beat-in-bar).
- Source: https://github.com/CPJKU/BallroomAnnotations
- Citation: Krebs et al., "Rhythmic pattern modeling for beat and downbeat
  tracking in musical audio," ISMIR 2013.

**GTZAN beat annotations.** Same format as Ballroom.
- Source: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
- Derived from: Marchand & Peeters, "Swing ratio estimation," DAFx 2015.
- Six tracks (`jazz.00003`, `.00009`, `.00010`, `.00014`, `.00018`, `.00020`)
  have empty annotation files in this distribution and are excluded from
  evaluation.

#### Setup

1. **Upload audio archives to Google Drive.** Place `ballroom.tar.gz` and
   `GTZAN.zip` in a single folder. Note the Drive path, as you will need to
   point the notebook at it.

2. **Open the notebook in Colab** and change runtime type to GPU.

3. **Edit the `DRIVE_DATASETS` path** in the first code cell of Section 1
   to match where you placed the archives, for example:
   ```python
   DRIVE_DATASETS = '/content/drive/MyDrive/YOUR_FOLDER_HERE'
   ```

#### Running the notebook

The notebook is organized into five sections, each self-contained:

1. **Setup**: Mounts Drive, extracts audio archives into Colab local
   storage, clones annotation repositories, builds the combined track
   list. Includes a visual QA cell that plots spectrogram + beat-target
   alignment for several random tracks.
   *Runtime: 2–3 minutes.*

2. **Dataset and caching**: Computes log-mel features and Gaussian-smoothed
   beat targets for all tracks and saves them to `/content/cache/`. On
   first run, set `REBUILD_CACHE = True` in the first code cell to build
   the cache from audio (~30 minutes) and back it up to Drive. On
   subsequent runs, leave `REBUILD_CACHE = False` to restore from Drive
   (~2 minutes). Also defines the `BeatDataset` class and builds
   train/val/test loaders.
   *Runtime on first build: ~35 minutes. Subsequent runs: ~2 minutes.*

3. **Model**: Defines the causal TCN architecture (`BeatTCN` class)
   with a `causal` flag so the same code path supports both the main
   causal model and the non-causal ablation. Instantiates a fresh model
   and verifies forward-pass shape preservation.
   *Runtime: <10 seconds.*

4. **Training**: Runs three training experiments under matched
   hyperparameters:
   - v1 (main causal model, dilations 1–32, 284k parameters)
   - v2 (extended receptive field, dilations 1–128, 367k parameters)
   - Non-causal variant (identical to v1 but with symmetric padding)

   Each experiment runs for 30 epochs and saves its best-val-F checkpoint
   to Drive. Checkpoint filenames: `best_v1.pt`, `best_causal.pt`,
   `best_noncausal.pt`. Training histories are saved as pickle files for
   later plotting.
   *Runtime: ~4 minutes per experiment on a T4 GPU.*

5. **Evaluation**: Loads the best checkpoint and computes all reported
   metrics:
   - `mir_eval` metrics (F-measure, Cemgil, CMLt, AMLt) overall and per
     dataset
   - Per-genre F-measure breakdown on GTZAN
   - Activation timing histogram (per-beat offset between activation peaks
     and annotated beats)
   - Causal vs. non-causal comparison
   - Qualitative plots of best- and worst-scoring tracks

   Also includes the annotation-completeness audit that identifies and
   excludes the six empty-annotation GTZAN tracks from final metrics.
   *Runtime: ~2 minutes.*

#### Reproducing reported metrics without retraining

To verify the reported numbers using the included checkpoints without
running training from scratch, run Sections 1, 2, 3, then skip Section 4
and run Section 5 directly. The evaluation cells load weights from
`best.pt` (which contains the v1 model) and produce the metrics and
figures cited in the report.

*Total runtime for verification-only: ~7 minutes.*

#### Expected outputs

If the pipeline is configured correctly, the QA plots in Section 1 will
show red beat markers aligned with activation peaks and with visible
onsets in the spectrograms. The Section 3 model cell will print
`Model: 284,001 parameters, running on cuda`. Section 5 will reproduce
all figures in the `figures/` directory and print metrics matching Table
1 of the final report.

### Part 2: Robotics demo

The robotics demo plays a song through the laptop's audio output and
synchronously triggers servo strikes via an Arduino. Code lives in
`trackingDemo/` (Python) and `strikerServo/` (Arduino sketch).

#### Hardware

- Arduino Mega 2560 (other Arduinos with `Servo.h` should also work).
- SG90 hobby servo (or equivalent).
- Wall-adapter power supply for the Arduino's barrel jack. **USB-only
  power is not sufficient** — the SG90's current draw causes brownouts and
  unstable strikes. Any 7–12V wall adapter works.
- A target object for the servo to strike (a hardback book, ceramic mug,
  or similar — anything that produces a clean, audible tap).

#### Wiring

| Servo wire | Arduino pin |
|---|---|
| Brown / Black (GND) | GND |
| Red (V+) | 5V |
| Orange / Yellow (signal) | Pin 9 |

#### Software setup

1. **Download the trained model checkpoint.** Save `best_v1.pt` from the
   Colab Drive folder into `trackingDemo/` alongside the Python scripts.

2. **Install Python dependencies.** From the `trackingDemo/` directory:
   ```
   python -m pip install -r ../requirements.txt
   python -m pip install pyserial sounddevice soundfile
   ```

3. **Upload the Arduino sketch.** Open `strikerServo/strikerServo.ino`
   in the Arduino IDE, select the correct board and port, and upload.
   The sketch implements a non-blocking state machine that drains the
   serial buffer and silently drops strike commands received while the
   servo is busy, preventing queue buildup at fast tempos.

4. **Find your Arduino's COM port** in Device Manager (Windows) or via
   `ls /dev/tty.*` (Mac/Linux). Edit `SERIAL_PORT` in `run_demo.py` to
   match.

#### Running the demo

```
cd trackingDemo
python extract_beats.py songs/your_song.mp3
python run_demo.py songs/your_song.mp3
```

The first command runs the model on the song and writes
`your_song.beats.json` alongside it. The second command plays the audio and
sends strike commands to the Arduino at `(beat_time − latency_compensation)`
for each predicted beat.

#### Latency tuning

Two latencies must be measured for your specific setup and combined into
the constants in `run_demo.py`:

- **Audio output latency** (~270 ms on the development laptop's HDMI output;
  varies by audio device). Measure by running the demo on a song with a
  clear, steady beat and adjusting `LEAD_PADDING_S` until the strikes land
  on-beat. Different audio outputs (built-in speakers, HDMI, Bluetooth)
  have different latencies.

- **Servo strike latency** (~78 ms for an SG90 with a small swing).
  Measure once via slow-motion video (240 fps phone slow-mo works) by
  recording the on-board LED and the servo arm simultaneously. Subtract
  the LED-on frame from the impact frame and divide by the frame rate.
  Set this in the `SERVO_LATENCY_S` constant.

The default values in the script are calibrated for the development setup
and will likely need adjustment.

### Demo songs

The `trackingDemo/songs/` directory contains pre-computed beat times
(`*.beats.json`) for the songs used in the demonstration video, but not
the audio files themselves (omitted for copyright reasons). To reproduce
the demo, source the corresponding audio files (Stayin' Alive, Billie Jean,
Sir Duke, Hallelujah Chorus, Take Five) and place the .mp3 files in this
directory with the same filename stems. Alternatively, run
`extract_beats.py` on any audio file of your choice to generate a fresh
JSON.
