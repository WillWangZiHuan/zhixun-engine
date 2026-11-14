"""Strict predict -> feedback -> train -> gate -> probation adaptation loop."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
from river.drift import ADWIN

from .experience import Experience
from .training import train_candidate
from .policy import decide, ESTIMATORS


@dataclass(frozen=True)
class Config:
    task: str = 'classification'
    use_experience: bool = True
    include_followup: bool = True
    policy: str = 'rule'
    estimator: str = 'rf'
    seed: int = 42
    windows: tuple = (256, 768)
    holdout_rows: int = 128
    delta: float = 0.01
    cooldown_rows: int = 256
    probation_rows: int = 256
    min_gain: float = 0.02
    rollback_margin: float = 0.03
    recovery_margin: float = 0.10
    max_attempts: int = 12
    train_seconds: float = 2.0
    hard_timeout_seconds: float = 30.0
    max_iter: int = 6
    n_jobs: int = 4

    def __post_init__(self):
        if type(self.include_followup) is not bool:
            raise ValueError('include_followup must be boolean')
        if self.estimator not in ESTIMATORS:
            raise ValueError('Unsupported CPU estimator')
        if self.policy not in ('rule', 'pytrio'):
            raise ValueError('policy must be rule or pytrio')
        if self.task not in ('classification', 'regression'):
            raise ValueError('task must be classification or regression')
        if not self.windows or any(not isinstance(w, int) or w < 32 for w in self.windows):
            raise ValueError('windows must contain integers >= 32')
        if len(set(self.windows)) != len(self.windows):
            raise ValueError('windows must be unique')
        for key in ('holdout_rows', 'cooldown_rows', 'probation_rows', 'max_attempts', 'max_iter', 'n_jobs'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('train_seconds', 'hard_timeout_seconds', 'min_gain', 'rollback_margin', 'recovery_margin'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 < self.delta < 1 or self.n_jobs > 4:
            raise ValueError('delta must be in (0, 1) and n_jobs <= 4')


class Engine:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.memory = Experience()
        self.detector = ADWIN(delta=config.delta)
        self.pending = None
        self.closed = False
        self.probation = None
        self.active = None
        self.static = None
        self.version = 'bootstrap'
        self.observed = 0
        self.last_attempt_at = 0
        self.attempts = 0
        self.decisions = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.incident = False
        self.incident_start = None
        self.drift_count = 0
        self.promotions = 0
        self.rollbacks = 0
        self.stream_rows = 0
        self.loss_sum = 0.0
        self.static_loss_sum = 0.0
        self.event_count = 0
        self.scale = 1.0
        self.X = None
        self.y = None
        self.reference_mean = None
        self.reference_std = None
        self.reference_loss = None
        self._event('created', config=asdict(config), mode='designated_pytrio_driver' if config.policy == 'pytrio' else 'local_rule_policy_no_llm')

    def _event(self, kind, **fields):
        self.event_count += 1
        event = dict(sequence=self.event_count, kind=kind,
                     utc=datetime.now(timezone.utc).isoformat(), observed_through=self.observed,
                     active_version=self.version, **fields)
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError('Features must be a nonempty finite numeric 2D array')
        if self.X is not None and X.shape[1] != self.X.shape[1]:
            raise ValueError('Feature schema changed')
        return X.copy()

    def _labels(self, y, n):
        y = np.asarray(y)
        if y.shape != (n,) or not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
            raise ValueError('Labels must be a finite numeric 1D array matching the batch')
        if self.config.task == 'classification' and not np.equal(y, np.floor(y)).all():
            raise ValueError('Classification labels must be integer class IDs')
        return y.copy()

    def _predict(self, model, X):
        prediction = np.asarray(model.predict(X))
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Model produced invalid predictions')
        return prediction

    def losses(self, y, prediction, bounded=False):
        if self.config.task == 'classification':
            return (y != prediction).astype(float)
        loss = np.abs(y - prediction)
        return np.minimum(loss / self.scale, 1.0) if bounded else loss

    def decision_losses(self, y, prediction):
        # Keep MAE ordering for regression; clipping is only for ADWIN input.
        raw = self.losses(y, prediction)
        return raw / self.scale if self.config.task == 'regression' else raw

    def _training_config(self):
        return {key: getattr(self.config, key) for key in
                ('task', 'seed', 'n_jobs', 'train_seconds', 'hard_timeout_seconds', 'max_iter', 'estimator')}

    def bootstrap(self, X, y, *, train_indices, tune_indices, gate_indices):
        if self.active is not None:
            raise RuntimeError('Engine is already initialized')
        X = self._features(X)
        y = self._labels(y, len(X))
        splits = [np.asarray(v) for v in (train_indices, tune_indices, gate_indices)]
        if any(v.ndim != 1 or len(v) == 0 or v.dtype.kind not in 'iu' for v in splits):
            raise ValueError('Bootstrap partitions must contain nonempty integer index arrays')
        if not np.array_equal(np.sort(np.concatenate(splits)), np.arange(len(y))):
            raise ValueError('Bootstrap partitions must be disjoint and cover each input row exactly once')
        train, tune, gate = splits
        if len(train) < 32:
            raise ValueError('Bootstrap requires at least 32 training rows')
        if self.config.task == 'classification':
            if len(np.unique(y[train])) < 2 or not np.array_equal(np.unique(y[train]), np.unique(y)):
                raise ValueError('Bootstrap training must cover every known initial class')
        self.scale = max(float(np.std(y[train])) * 3, 1e-6)
        self.active, info = train_candidate(self.output / 'experiments' / 'bootstrap',
            X[train], y[train], X[tune], y[tune], self._training_config())
        self.static = self.active
        self.reference_mean = X[train].mean(axis=0)
        self.reference_std = np.maximum(X[train].std(axis=0), 1e-6)
        self.X, self.y = X, y
        self.observed = len(y)
        self.last_attempt_at = self.observed
        gate_prediction = self._predict(self.active, X[gate])
        gate_loss = self.losses(y[gate], gate_prediction, bounded=True)
        self.reference_loss = float(self.decision_losses(y[gate], gate_prediction).mean())
        for loss in gate_loss:
            self.detector.update(float(loss))
        self._trim_history()
        self._event('bootstrapped', split={name: v.tolist() for name, v in
                    zip(('train_indices', 'tune_indices', 'gate_indices'), splits)},
                    gate_loss=float(gate_loss.mean()), training=info)
        self.checkpoint()

    def predict_batch(self, batch_id, X):
        if self.closed:
            raise RuntimeError('The stream is closed')
        if self.active is None:
            raise RuntimeError('Call bootstrap first')
        if self.pending is not None:
            raise RuntimeError('Labels for the pending batch are required before another batch')
        if not isinstance(batch_id, int) or batch_id != self.observed:
            raise ValueError('batch_id must equal the next chronological row offset')
        X = self._features(X)
        prediction = self._predict(self.active, X)
        previous = self._predict(self.probation['previous_model'], X) if self.probation else None
        self.pending = dict(batch_id=batch_id, X=X, prediction=prediction.copy(),
                            static_prediction=self._predict(self.static, X), previous_prediction=previous)
        self._event('predicted', batch_id=batch_id, rows=len(X))
        self.checkpoint()
        return prediction.copy()

    def observe_labels(self, batch_id, y, *, final=False):
        if type(final) is not bool:
            raise ValueError('final must be boolean')
        if self.pending is None or batch_id != self.pending['batch_id']:
            raise ValueError('Labels must match the pending prediction batch')
        pending = self.pending
        y = self._labels(y, len(pending['X']))
        loss = self.losses(y, pending['prediction'])
        bounded = self.losses(y, pending['prediction'], bounded=True)
        static_loss = self.losses(y, pending['static_prediction'])
        self.observed += len(y)
        self.stream_rows += len(y)
        self.loss_sum += float(loss.sum())
        self.static_loss_sum += float(static_loss.sum())
        self.X = np.concatenate([self.X, pending['X']])
        self.y = np.concatenate([self.y, y])
        self._trim_history()
        self.pending = None
        self._event('feedback', batch_id=batch_id, rows=len(y), loss=float(loss.mean()),
                    static_loss=float(static_loss.mean()), bounded_loss=float(bounded.mean()))

        rolled_back = False
        if self.probation:
            probation = self.probation
            probation['rows'] += len(y)
            probation['candidate_loss'] += float(self.decision_losses(y, pending['prediction']).sum())
            probation['previous_loss'] += float(self.decision_losses(y, pending['previous_prediction']).sum())
            if probation['rows'] >= self.config.probation_rows:
                difference = (probation['candidate_loss'] - probation['previous_loss']) / probation['rows']
                followup = self.memory.complete_followup(probation['experience_id'],
                    start=probation['start'], end=self.observed, rows=probation['rows'],
                    candidate_loss=probation['candidate_loss']/probation['rows'],
                    incumbent_loss=probation['previous_loss']/probation['rows'])
                self._event('followup_completed', experience_id=probation['experience_id'],
                            followup=followup)
                if difference > self.config.rollback_margin:
                    self.active = probation['previous_model']
                    self.version = probation['previous_version']
                    self.memory.rollback(probation['experience_id'], difference, self.observed)
                    self.rollbacks += 1
                    self.incident = True
                    self.incident_start = self.observed
                    self.last_attempt_at = self.observed
                    self._event('rolled_back', failed_version=probation['candidate_version'],
                                normalized_loss_increase=difference, evaluated_rows=probation['rows'])
                    rolled_back = True
                else:
                    candidate_mean = probation['candidate_loss'] / probation['rows']
                    self.incident = candidate_mean > self.reference_loss + self.config.recovery_margin
                    if not self.incident:
                        self.incident_start = None
                    self._event('probation_passed', normalized_loss_difference=difference,
                                evaluated_rows=probation['rows'], recovery_needed=self.incident,
                                candidate_loss=candidate_mean,
                                recovery_target=self.reference_loss + self.config.recovery_margin)
                self.probation = None
        if rolled_back:
            self.detector = ADWIN(delta=self.config.delta)
        else:
            for index, value in enumerate(bounded):
                before = self.detector.estimation
                self.detector.update(float(value))
                if self.detector.drift_detected and self.detector.estimation > before:
                    self.drift_count += 1
                    alarm_offset = self.observed - len(y) + index + 1
                    if not self.incident:
                        # Conservative cutoff: use only labels arriving AFTER detection.
                        # This is not knowledge of the generator's true change point.
                        self.incident_start = alarm_offset
                    self.incident = True
                    shift = np.abs((pending['X'].mean(axis=0) - self.reference_mean) / self.reference_std)
                    self._event('drift_detected', alarm_offset=alarm_offset,
                                bounded_loss=float(bounded.mean()),
                                detector_mean=float(self.detector.estimation),
                                top_shift_features=np.argsort(shift)[-3:][::-1].tolist(),
                                max_standardized_mean_shift=float(shift.max()),
                                interpretation='statistical_change_not_causal_diagnosis')
        if final:
            self.closed = True
            self._event('stream_closed', probation_unresolved=self.probation is not None)
        if (not self.closed and self.incident and self.probation is None
                and self._eligible_windows()
                and self.observed - self.last_attempt_at >= max(self.config.cooldown_rows, self.config.holdout_rows)
                and self.attempts < self.config.max_attempts
                and (self.config.policy == 'rule' or self.decisions < self.config.max_attempts)):
            self._adapt(float(bounded.mean()))
        self.checkpoint()
        return self.summary()

    def _trim_history(self):
        capacity = max(self.config.windows) + 2 * self.config.holdout_rows
        self.X, self.y = self.X[-capacity:], self.y[-capacity:]

    def _eligible_windows(self):
        available = len(self.y)
        if self.incident_start is not None:
            available = min(available, self.observed - self.incident_start)
        return tuple(w for w in self.config.windows if w + 2 * self.config.holdout_rows <= available)

    def _adapt(self, recent_loss):
        context = 'high_error' if recent_loss >= 0.4 else 'moderate_error'
        estimator = self.config.estimator
        if self.config.policy == 'pytrio':
            self.decisions += 1
            # Diagnostic inputs exclude the fresh tuning/gate tail. No raw labels
            # or hidden future batches are made available to the driver.
            hold = self.config.holdout_rows
            history = self.memory.driver_entries(at=self.observed, reserved_rows=2*hold,
                include_followup=self.config.include_followup) if self.config.use_experience else []
            diagnostic_X, diagnostic_y = self.X[:-2 * hold], self.y[:-2 * hold]
            shift = np.abs((diagnostic_X.mean(axis=0) - self.reference_mean) / self.reference_std)
            request = dict(seed=self.config.seed, task=self.config.task,
                observed_through=self.observed, eligible_windows=list(self._eligible_windows()),
                attempts_remaining=self.config.max_attempts - self.attempts,
                training_budget_seconds=self.config.train_seconds, cpu_cores=self.config.n_jobs,
                diagnostic_loss=float(self.decision_losses(diagnostic_y,
                    self._predict(self.active, diagnostic_X)).mean()),
                top_shift_features=np.argsort(shift)[-3:][::-1].tolist(),
                max_standardized_mean_shift=float(shift.max()),
                experience=history)
            self._event('driver_requested', decision_id=self.decisions, request=request,
                        reserved_input_tokens=4096, reserved_output_tokens=512)
            self.checkpoint()
            try:
                decision, usage = decide(self.output / 'decisions' / f'{self.decisions:03d}', request)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                self._event('driver_failed', decision_id=self.decisions, error=type(error).__name__)
                self.checkpoint()
                raise
            self.input_tokens += usage['input_tokens']
            self.output_tokens += usage['output_tokens']
            self._event('driver_decided', decision_id=self.decisions, decision=decision, usage=usage)
            if decision['action'] == 'wait':
                self.last_attempt_at = self.observed
                return
            window, estimator = decision['window'], decision['estimator']
        else:
            window, decision = (self.memory if self.config.use_experience else Experience()).choose(
                context, self._eligible_windows())
        hold = self.config.holdout_rows
        X, y = self.X[-(window + 2 * hold):], self.y[-(window + 2 * hold):]
        self.attempts += 1
        self.last_attempt_at = self.observed
        candidate_version = f'candidate_{self.attempts:03d}'
        # The fresh gate was never passed to FLAML or used by an earlier attempt.
        evidence = dict(train_start=self.observed - len(y), train_end=self.observed - 2 * hold,
                        tune_start=self.observed - 2 * hold, tune_end=self.observed - hold,
                        gate_start=self.observed - hold, gate_end=self.observed)
        self._event('experiment_started', candidate=candidate_version, window=window,
                    context=context, decision=decision, estimator=estimator, split=evidence)
        training_started = time.monotonic()
        try:
            candidate, info = train_candidate(self.output / 'experiments' / candidate_version,
                X[:-2 * hold], y[:-2 * hold], X[-2 * hold:-hold], y[-2 * hold:-hold],
                self._training_config() | {'estimator': estimator})
            old_loss = float(self.decision_losses(y[-hold:], self._predict(self.active, X[-hold:])).mean())
            new_loss = float(self.decision_losses(y[-hold:], self._predict(candidate, X[-hold:])).mean())
        except (subprocess.SubprocessError, OSError, ValueError, EOFError) as error:
            self.memory.record(context=context, window=window, status='failed', reward=-1.0,
                               at=self.observed, evidence={**evidence, 'candidate': candidate_version,
                                   'estimator': estimator, 'error': type(error).__name__,
                                   'training': {'wall_seconds': time.monotonic() - training_started}})
            self._event('experiment_failed', candidate=candidate_version,
                        error=type(error).__name__, detail=str(error)[:1000])
            return
        gain = old_loss - new_loss
        accepted = gain >= self.config.min_gain
        entry_id = self.memory.record(context=context, window=window,
            status='promoted' if accepted else 'rejected', reward=gain, at=self.observed,
            evidence={**evidence, 'estimator': estimator, 'candidate': candidate_version, 'incumbent_loss': old_loss,
                      'candidate_loss': new_loss, 'training': info})
        if accepted:
            self.probation = dict(previous_model=self.active, previous_version=self.version,
                candidate_version=candidate_version, experience_id=entry_id,
                start=self.observed, rows=0, candidate_loss=0.0, previous_loss=0.0)
            self.active, self.version = candidate, candidate_version
            self.promotions += 1
            # Publishing a partial improvement does not establish recovery.
            # Future labels end the incident, or trigger another budgeted attempt.
            self.incident = True
            self.detector = ADWIN(delta=self.config.delta)
        self._event('promoted' if accepted else 'rejected', candidate=candidate_version,
                    incumbent_loss=old_loss, candidate_loss=new_loss, gain=gain,
                    experience_id=entry_id, training=info)

    def summary(self):
        return dict(task=self.config.task, stream_closed=self.closed,
                    stream_rows=self.stream_rows, observed_rows=self.observed,
                    active_version=self.version, attempts=self.attempts, drift_events=self.drift_count,
                    promotions=self.promotions, rollbacks=self.rollbacks,
                    metric='error_rate' if self.config.task == 'classification' else 'MAE',
                    adaptive_loss=self.loss_sum / self.stream_rows if self.stream_rows else None,
                    static_loss=self.static_loss_sum / self.stream_rows if self.stream_rows else None,
                    pending_batch=self.pending['batch_id'] if self.pending else None,
                    probation_active=self.probation is not None,
                    recovery_pending=self.incident,
                    recovery_data_start=self.incident_start,
                    budget_remaining=self.config.max_attempts - self.attempts,
                    scope='local_validation_not_official_score',
                    policy=self.config.policy, use_experience=self.config.use_experience,
                    include_followup=self.config.include_followup,
                    driver_calls=self.decisions, driver_input_tokens=self.input_tokens,
                    driver_output_tokens=self.output_tokens)

    def checkpoint(self):
        # Only reload checkpoints produced locally by this project (joblib is pickle).
        temp = self.output / 'checkpoint.tmp'
        joblib.dump(self, temp)
        os.replace(temp, self.output / 'checkpoint.joblib')
        for name, value in [('experience.json', self.memory.entries), ('summary.json', self.summary())]:
            temp = self.output / (name + '.tmp')
            temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            os.replace(temp, self.output / name)

    @classmethod
    def restore(cls, output):
        engine = joblib.load(Path(output) / 'checkpoint.joblib')
        if not isinstance(engine, cls):
            raise TypeError('Not an Engine checkpoint')
        if engine.output != Path(output).resolve():
            raise ValueError('Checkpoint directory moved; use its original output location')
        return engine
