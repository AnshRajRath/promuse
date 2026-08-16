from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import pandas as pd
import librosa

DEAM_PATH = Path("../data/music/DEAM")
AUDIO_PATH = DEAM_PATH / "DEAM_audio" / "MEMD_audio"
METADATA_PATH = Path("../data/metadata/music_metadata.csv")
OUTPUT_PATH = Path("../data/processed/music_features.csv")

SR = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def summarize(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {f"{prefix}_{x}": np.nan for x in ["mean", "std", "min", "max", "range"]}

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
    }


def extract_features(audio_path):
    song_id = int(audio_path.stem)

    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)

        if y.size == 0:
            raise ValueError("Empty audio")

        features = {
            "song_id": song_id,
            "audio_path": str(audio_path),
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "sample_rate": sr,
        }

        rms = librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features.update(summarize(rms, "rms"))

        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features.update(summarize(centroid, "spectral_centroid"))

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features.update(summarize(bandwidth, "spectral_bandwidth"))

        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features.update(summarize(rolloff, "spectral_rolloff"))

        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        for i, row in enumerate(contrast):
            features.update(summarize(row, f"spectral_contrast_{i + 1}"))

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        for i, row in enumerate(mfcc):
            features.update(summarize(row, f"mfcc_{i + 1}"))

        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        for i, row in enumerate(chroma):
            features.update(summarize(row, f"chroma_{i + 1}"))

        tempo = librosa.feature.tempo(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        features["tempo"] = float(np.nanmean(tempo))

        return {
            "success": True,
            "features": features,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "features": None,
            "error": f"{type(e).__name__}: {e}",
            "song_id": song_id,
            "audio_path": str(audio_path)
        }


def process_music_files(max_workers=8, limit=None):
    print("=" * 55)
    print("Starting music feature extraction")
    print("=" * 55)

    audio_files = sorted(AUDIO_PATH.glob("*.mp3"))

    if limit is not None:
        audio_files = audio_files[:limit]

    print(f"Total files : {len(audio_files)}")
    print(f"Workers     : {max_workers}")
    print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    successful = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_features, path): path
            for path in audio_files
        }

        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if result["success"]:
                successful.append(result["features"])
            else:
                failed.append({
                    "song_id": result["song_id"],
                    "audio_path": result["audio_path"],
                    "error": result["error"]
                })

            if completed % 25 == 0 or completed == len(audio_files):
                print(f"Processed {completed}/{len(audio_files)}")

    feature_df = pd.DataFrame(successful)

    if METADATA_PATH.exists():
        metadata_df = pd.read_csv(METADATA_PATH)
        feature_df["song_id"] = feature_df["song_id"].astype(int)
        metadata_df["song_id"] = metadata_df["song_id"].astype(int)

        feature_df = feature_df.merge(
            metadata_df,
            on="song_id",
            how="left",
            suffixes=("", "_metadata")
        )

        if "audio_path_metadata" in feature_df.columns:
            feature_df.drop(columns=["audio_path_metadata"], inplace=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 55)
    print("Processing complete")
    print("=" * 55)
    print(f"Successful files : {len(successful)}")
    print(f"Failed files     : {len(failed)}")
    print(f"Features         : {feature_df.shape[1]}")
    print(f"Output           : {OUTPUT_PATH.resolve()}")
    print(f"Time             : {elapsed / 60:.2f} minutes")

    if elapsed > 0:
        print(f"Speed             : {len(audio_files) / elapsed * 60:.1f} files/minute")

    if failed:
        error_path = OUTPUT_PATH.with_name("music_feature_errors.csv")
        pd.DataFrame(failed).to_csv(error_path, index=False)
        print(f"Errors            : {error_path.resolve()}")

    return feature_df, failed


if __name__ == "__main__":
    process_music_files(max_workers=8, limit=None)