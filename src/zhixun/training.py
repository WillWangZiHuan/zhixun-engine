"""Run each FLAML experiment in a bounded, independent CPU process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np


def train_candidate(directory, X_train, y_train, X_tune, y_tune, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / 'input.npz', X_train=X_train, y_train=y_train,
             X_tune=X_tune, y_tune=y_tune)
    (directory / 'request.json').write_text(json.dumps(config, indent=2))
    env = {key: value for key, value in os.environ.items()
           if key not in ('PYTRIO_API_KEY', 'SWANLAB_API_KEY')}
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        # RF parallelism is set by n_jobs; avoid nested BLAS/OpenMP fan-out.
        env[key] = '1'
    started = time.monotonic()
    try:
        with (directory / 'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'zhixun.worker', str(directory.resolve())],
                           env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=config['hard_timeout_seconds'], check=True)
    finally:
        # Train/tune arrays need not be duplicated in every experiment artifact.
        (directory / 'input.npz').unlink(missing_ok=True)
        (directory / 'wall_time.json').write_text(json.dumps({
            'seconds': time.monotonic() - started}))
    model = joblib.load(directory / 'model.joblib')
    metadata = json.loads((directory / 'result.json').read_text())
    metadata['wall_seconds'] = time.monotonic() - started
    return model, metadata
