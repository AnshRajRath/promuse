from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
import librosa
import parselmouth


PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
ARRAY_DIR = PROJECT_ROOT / "data" / "processed" / "speech_arrays"
FEATURE_CSV = METADATA_DIR / "speech_features.csv"
ERROR_CSV = METADATA_DIR / "speech_feature_errors.csv"

for directory in [METADATA_DIR, ARRAY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


TARGET_SR = 16000
FMIN = 50
FMAX = 300
N_MFCC = 13


def extract_speech_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    duration = len(y) / sr

    f0 = librosa.yin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=1024,
        hop_length=256
    )

    f0 = f0.astype(np.float32)

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=256
    )[0].astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=1024,
        hop_length=256
    )[0].astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=1024,
        hop_length=256
    ).astype(np.float32)

    valid_f0 = f0[np.isfinite(f0)]

    if len(valid_f0):
        pitch_mean = float(np.mean(valid_f0))
        pitch_median = float(np.median(valid_f0))
        pitch_min = float(np.min(valid_f0))
        pitch_max = float(np.max(valid_f0))
        pitch_std = float(np.std(valid_f0))
        pitch_range = pitch_max - pitch_min
    else:
        pitch_mean = np.nan
        pitch_median = np.nan
        pitch_min = np.nan
        pitch_max = np.nan
        pitch_std = np.nan
        pitch_range = np.nan

    sound = parselmouth.Sound(
        y,
        sampling_frequency=sr
    )

    try:
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            FMIN,
            FMAX
        )

        jitter = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3
        )

        shimmer = parselmouth.praat.call(
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6
        )

    except Exception:
        jitter = np.nan
        shimmer = np.nan

    features = {
        "sample_rate": sr,
        "duration": duration,

        "pitch_mean": pitch_mean,
        "pitch_median": pitch_median,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,

        "energy_mean": float(np.mean(rms)),
        "energy_std": float(np.std(rms)),
        "energy_min": float(np.min(rms)),
        "energy_max": float(np.max(rms)),
        "energy_range": float(np.max(rms) - np.min(rms)),

        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_min": float(np.min(zcr)),
        "zcr_max": float(np.max(zcr)),

        "jitter": jitter,
        "shimmer": shimmer
    }

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return {
        "features": features,
        "pitch": f0,
        "energy": rms,
        "zcr": zcr,
        "mfcc": mfcc
    }


def process_single_file(task):
    index, row = task

    try:
        audio_path = Path(row["file"])

        result = extract_speech_features(
            audio_path
        )

        file_id = (
            f"{row['dataset']}_"
            f"{row['speaker']}_"
            f"{index}"
        )

        return {
            "success": True,
            "index": index,
            "file_id": file_id,
            "dataset": row["dataset"],
            "file": str(audio_path),
            "speaker": row["speaker"],
            "emotion": row["emotion"],
            "emotion_id": row["emotion_id"],
            "features": result["features"],
            "pitch": result["pitch"],
            "energy": result["energy"],
            "zcr": result["zcr"],
            "mfcc": result["mfcc"]
        }

    except Exception as e:
        return {
            "success": False,
            "index": index,
            "file": str(row["file"]),
            "error": repr(e)
        }


def save_feature_arrays(
    file_id,
    pitch,
    energy,
    zcr,
    mfcc
):
    np.save(
        ARRAY_DIR / f"{file_id}_pitch.npy",
        pitch
    )

    np.save(
        ARRAY_DIR / f"{file_id}_energy.npy",
        energy
    )

    np.save(
        ARRAY_DIR / f"{file_id}_zcr.npy",
        zcr
    )

    np.save(
        ARRAY_DIR / f"{file_id}_mfcc.npy",
        mfcc
    )


def process_all_speech_files(
    speech_metadata,
    max_workers=6
):
    tasks = list(speech_metadata.iterrows())

    total = len(tasks)

    if max_workers is None:
        max_workers = max(
            1,
            (os.cpu_count() or 2) - 2
        )

    print("=" * 55)
    print("FAST SPEECH FEATURE EXTRACTION")
    print("=" * 55)
    print(f"Total files : {total}")
    print(f"Workers     : {max_workers}")
    print(f"Target SR   : {TARGET_SR} Hz")
    print(f"Pitch range : {FMIN}-{FMAX} Hz")
    print("Figures     : Disabled")
    print()

    results = []
    errors = []
    completed = 0

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                process_single_file,
                task
            )
            for task in tasks
        ]

        for future in as_completed(futures):

            completed += 1

            try:
                result = future.result()

            except Exception as e:
                errors.append({
                    "index": None,
                    "file": None,
                    "error": repr(e)
                })
                continue

            if result["success"]:

                file_id = result["file_id"]

                save_feature_arrays(
                    file_id,
                    result["pitch"],
                    result["energy"],
                    result["zcr"],
                    result["mfcc"]
                )

                results.append({
                    "file_id": file_id,
                    "dataset": result["dataset"],
                    "file": result["file"],
                    "speaker": result["speaker"],
                    "emotion": result["emotion"],
                    "emotion_id": result["emotion_id"],
                    **result["features"]
                })

            else:

                errors.append({
                    "index": result["index"],
                    "file": result["file"],
                    "error": result["error"]
                })

            if completed % 50 == 0 or completed == total:
                print(
                    f"Processed {completed}/{total}"
                )

    feature_df = pd.DataFrame(results)
    error_df = pd.DataFrame(errors)

    if not feature_df.empty:

        feature_df = feature_df.sort_values(
            "file_id"
        ).reset_index(drop=True)

        feature_df.to_csv(
            FEATURE_CSV,
            index=False
        )

    if not error_df.empty:

        error_df.to_csv(
            ERROR_CSV,
            index=False
        )

    print()
    print("=" * 55)
    print("PROCESSING COMPLETE")
    print("=" * 55)
    print(f"Successful files : {len(feature_df)}")
    print(f"Failed files     : {len(error_df)}")
    print(f"Feature CSV      : {FEATURE_CSV}")
    print(f"Arrays           : {ARRAY_DIR}")

    if not error_df.empty:
        print(f"Error CSV        : {ERROR_CSV}")

    return feature_df, error_df


if __name__ == "__main__":
    print(
        "Import this module from your notebook or main script."
    )
    print(
        "Example:"
    )
    print(
        "from spext_fast import process_all_speech_files"
    )
    print()
    print(
        "speech_features, speech_errors = "
        "process_all_speech_files(speech_metadata, max_workers=6)"
    )