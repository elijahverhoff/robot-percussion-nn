"""
Play an audio file and send strike commands to the Arduino at predicted
beat times, compensated for servo latency.

Usage:  python run_demo.py songs/your_song.mp3
"""

import sys
import json
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import serial


# --- Configuration ---
SERIAL_PORT       = 'COM3'   # check Device Manager for actual port
SERIAL_BAUD       = 115200
SERVO_LATENCY_S   = 0.078    # 78 ms measured latency
LEAD_PADDING_S    = 0.270    # extra safety margin for OS scheduler jitter


def load_audio(path):
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # to mono
    return audio.astype(np.float32), sr


def load_beats(audio_path):
    beats_path = Path(audio_path).with_suffix('.beats.json')
    with open(beats_path) as f:
        data = json.load(f)
    return np.array(data['beats'])


def open_arduino(port=SERIAL_PORT, baud=SERIAL_BAUD):
    print(f"Opening {port} at {baud} baud...")
    arduino = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # Arduino resets on serial open; wait for boot
    # Drain any startup messages:
    while arduino.in_waiting:
        arduino.readline()
    print("Arduino ready")
    return arduino


def schedule_strikes(arduino, beat_times, t_start, advance_s):
    """Send 's' to Arduino at each (beat_time - advance_s), measured from t_start."""
    for beat_time in beat_times:
        target = t_start + beat_time - advance_s
        now = time.perf_counter()
        wait = target - now
        if wait > 0:
            time.sleep(wait)
        arduino.write(b's')
        # Optional: print live timing info
        actual_offset = (time.perf_counter() - t_start) - (beat_time - advance_s)
        print(f"  beat at {beat_time:6.2f}s   send offset {actual_offset*1000:+5.1f}ms")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_demo.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    audio, sr = load_audio(audio_path)
    beat_times = load_beats(audio_path)
    print(f"Loaded {len(audio)/sr:.1f}s of audio at {sr} Hz")
    print(f"Loaded {len(beat_times)} beats")

    arduino = open_arduino()

    # Filter out beats that occur before the servo can be commanded
    # (i.e., beats earlier than SERVO_LATENCY into the song):
    advance = SERVO_LATENCY_S + LEAD_PADDING_S
    valid_beats = beat_times[beat_times >= advance]
    print(f"Triggering on {len(valid_beats)} beats "
          f"(skipped {len(beat_times) - len(valid_beats)} too early)")

    print("\nStarting playback in 1 second...")
    time.sleep(1.0)

    # Start audio playback
    sd.play(audio, sr)
    t_start = time.perf_counter()

    # Run scheduler in main thread (audio plays in background)
    schedule_strikes(arduino, valid_beats, t_start, advance)

    # Wait for audio to finish
    sd.wait()
    print("\nDemo complete")
    arduino.close()


if __name__ == '__main__':
    main()