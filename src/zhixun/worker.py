"""Internal training worker. Gate/test labels are never passed here."""
import json
import resource
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from flaml import AutoML
from .policy import ESTIMATORS


def main():
    directory = Path(sys.argv[1])
    config = json.loads((directory / 'request.json').read_text())
    data = np.load(directory / 'input.npz', allow_pickle=False)
    estimator = config.get('estimator', 'rf')
    if estimator not in ESTIMATORS:
        raise ValueError('Unsupported CPU estimator')
    model = AutoML()
    started = time.monotonic()
    model.fit(X_train=data['X_train'], y_train=data['y_train'],
              X_val=data['X_tune'], y_val=data['y_tune'],
              task=config['task'], metric='accuracy' if config['task'] == 'classification' else 'mae',
              estimator_list=[estimator], eval_method='holdout', retrain_full=False,
              time_budget=config['train_seconds'], max_iter=config['max_iter'],
              n_jobs=config['n_jobs'], seed=config['seed'],
              mlflow_logging=False, verbose=0, log_type='all',
              log_file_name=str(directory / 'flaml_trials.jsonl'))
    prediction = np.asarray(model.predict(data['X_tune']))
    if prediction.shape != data['y_tune'].shape or not np.isfinite(prediction).all():
        raise ValueError('Training produced invalid predictions')
    # FLAML owns preprocessing and the inverse class mapping. Persist the complete
    # predictor, otherwise nonzero/noncontiguous classes silently change at reload.
    joblib.dump(model, directory / 'model.joblib')
    restored = joblib.load(directory / 'model.joblib')
    restored_prediction = restored.predict(data['X_tune'])
    if config['task'] == 'classification':
        np.testing.assert_array_equal(prediction, restored_prediction)
    else:
        # Parallel tree reduction can change the last floating-point bits.
        np.testing.assert_allclose(prediction, restored_prediction, rtol=1e-12, atol=1e-12)
    (directory / 'result.json').write_text(json.dumps({
        'estimator': model.best_estimator, 'parameters': model.best_config,
        'tuning_loss': float(model.best_loss), 'fit_seconds': time.monotonic() - started,
        'prediction_roundtrip': 'passed',
        'artifact_bytes': (directory / 'model.joblib').stat().st_size,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        'train_rows': len(data['y_train']), 'tune_rows': len(data['y_tune']),
    }, indent=2))


if __name__ == '__main__':
    main()
