"""Bounded CPU features from raw XJTU-SY windows; no invented stage/RUL labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np


STAT_NAMES = ('mean', 'std', 'rms', 'peak_to_peak', 'absolute_peak', 'skew',
              'kurtosis', 'crest_factor', 'shape_factor', 'spectral_centroid_hz',
              'spectral_entropy', 'power_0_3200', 'power_3200_6400', 'power_6400_12800')
FEATURE_NAMES = [f'{axis}_{stat}' for axis in ('horizontal', 'vertical') for stat in STAT_NAMES]


def features(window, sampling_hz=25600):
    window = np.asarray(window, dtype=float)
    if window.shape != (32768, 2) or not np.isfinite(window).all():
        raise ValueError('Expected one finite 32768 x 2 vibration window')
    result = []
    for signal in window.T:
        mean, std = signal.mean(), signal.std()
        rms, peak = np.sqrt(np.mean(signal ** 2)), np.max(np.abs(signal))
        standardized = (signal - mean) / std if std else np.zeros_like(signal)
        power = np.abs(np.fft.rfft(signal - mean)) ** 2
        frequencies = np.fft.rfftfreq(len(signal), d=1 / sampling_hz)
        total = power.sum()
        distribution = power / total if total else np.zeros_like(power)
        nonzero = distribution[distribution > 0]
        entropy = -np.sum(nonzero * np.log(nonzero)) / np.log(len(power))
        band_power = [distribution[(frequencies >= low) & (frequencies < high)].sum()
                      for low, high in [(0, 3200), (3200, 6400), (6400, 12801)]]
        result.extend([mean, std, rms, np.ptp(signal), peak,
                       np.mean(standardized ** 3), np.mean(standardized ** 4),
                       peak / rms if rms else 0,
                       rms / np.mean(np.abs(signal)) if rms else 0,
                       np.sum(frequencies * distribution), entropy, *band_power])
    return np.asarray(result)


def process_file(path_string):
    path = Path(path_string)
    raw = path.read_bytes()
    window = np.loadtxt(io.BytesIO(raw), delimiter=',', skiprows=1)
    return features(window), hashlib.sha256(raw).hexdigest()


def extract(root, output, workers=2):
    root, output = Path(root).resolve(), Path(output).resolve()
    if workers not in (1, 2, 3, 4):
        raise ValueError('Use at most four CPU workers')
    paths = sorted(root.glob('*/*/*.csv'), key=lambda p: (p.parent.parent.name, p.parent.name, int(p.stem)))
    if len(paths) != 9216 or len({p.parent for p in paths}) != 15:
        raise ValueError('Expected the complete 15-bearing, 9216-window source dataset')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(task='raw signal feature preparation', labels_available=False,
                    source_root=str(root), sampling_hz=25600, samples_per_window=32768,
                    window_seconds=1.28, acquisition_interval_seconds=60,
                    feature_names=FEATURE_NAMES,
                    definition='Population standard deviation; raw kurtosis; centered one-sided FFT power fractions.',
                    normalization='None fitted across windows; later scaling must use training data only.',
                    labels='Terminal fault metadata is not a per-window fault/stage/RUL label.', files=[])
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (path, result) in enumerate(zip(paths, pool.map(process_file, map(str, paths), chunksize=16)), 1):
            vector, sha = result
            rows.append(vector)
            manifest['files'].append(dict(path=str(path.relative_to(root)), sha256=sha,
                                          bearing=path.parent.name, condition=path.parent.parent.name,
                                          acquisition_index=int(path.stem)))
            if i % 500 == 0:
                print(f'Extracted {i}/{len(paths)} windows', flush=True)
    X = np.stack(rows)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite feature values')
    np.savez_compressed(output / 'features.npz', X=X)
    manifest.update(rows=len(X), columns=X.shape[1], completed=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Completed {X.shape}; no labels assigned', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Extract interpretable CPU vibration features')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    extract(args.data, args.output, args.workers)


if __name__ == '__main__':
    main()
