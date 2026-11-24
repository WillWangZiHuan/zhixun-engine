"""Matched, training-only updates to the supplied regression network."""
from copy import deepcopy
import numpy as np
from .bearing_update import continue_bearing
from .engine_update import continue_engine


def bearing_head(model, data, objective="long_life"):
    hidden, prediction = model.forward(model.normalized(data['x']))
    design = np.column_stack((hidden, np.ones(len(hidden))))
    target = (data['y'] - model.meta['target_mean']) / model.meta['target_scale']
    residual = target - prediction[:, 0]
    if objective == "long_life":
        weight = np.exp(np.clip(2 * target, -5, 5))
        damping = 0.01
    else:
        weight = np.clip(np.abs(residual), 0.1, 5.0)
        damping = 0.1
    weight /= weight.sum()
    normal = design.T @ (weight[:, None] * design)
    normal.flat[::len(normal) + 1] += damping
    delta = np.linalg.solve(normal, design.T @ (weight * residual))
    model.params['w2'][:, 0] += delta[:-1]
    model.params['b2'] += delta[-1]


def scaled_probe(parent, pilot, scale):
    if scale == 1:
        return pilot
    warm = deepcopy(parent)
    for name in warm.params:
        warm.params[name] += scale * (pilot.params[name] - parent.params[name])
    return warm


def correct_tail_bias(parent_prediction, pilot, train):
    """Take one output-bias Newton step on training NASA loss, bounded by the pilot's movement."""
    model = deepcopy(pilot)
    prediction = pilot.predict(train['x'])
    error = prediction - train['y']
    slope = np.where(error >= 0, 1 / 10, -1 / 13)
    exponent = error * slope
    weight = np.exp(exponent - exponent.max())
    step = -(weight @ slope) / (weight @ (slope * slope))
    radius = np.sqrt(np.mean((prediction - parent_prediction) ** 2))
    model.params['b2'] += np.clip(step, -radius, radius) / model.meta['target_scale']
    return model


def paired_probes(parent, train, valid, score, tail_conflict=False):
    key = parent.meta['profile']
    reference = deepcopy(parent)
    alternative = deepcopy(parent)
    if key == 'B':
        continue_bearing(reference, train['x'], train['y'])
        bearing_head(alternative, train)
        mode = 'long_life_head'
    else:
        continue_engine(reference, train['x'], train['y'])
        continue_engine(alternative, train['x'], train['y'], passes=6, tail_mix=0.10)
        mode = 'tail_conditioned'
    models = {'reference': reference, mode: alternative, 'adam_only': parent}
    if key == 'B':
        conservative = deepcopy(parent)
        bearing_head(conservative, train, objective='residual')
        models['residual_head'] = conservative
    half = deepcopy(parent)
    for name in half.params:
        half.params[name] += 0.5 * (alternative.params[name] - parent.params[name])
    models[mode + '_half'] = half
    if key == 'C':
        middle = deepcopy(reference)
        for name in middle.params:
            middle.params[name] += 0.5 * (alternative.params[name] - reference.params[name])
        models['balanced'] = middle
        if tail_conflict:
            parent_prediction = parent.predict(train['x'])
            models['calibrated_reference'] = correct_tail_bias(parent_prediction, reference, train)
            models['calibrated_half'] = correct_tail_bias(parent_prediction, scaled_probe(parent, reference, 0.5), train)
    measurements = {name: score(model, valid) for name, model in models.items()}
    if key == 'B':
        cuts = np.quantile(train['y'], [1/3, 2/3])
        bands = np.searchsorted(cuts, valid['y'])
        for name, model in models.items():
            error = model.predict(valid['x']) - valid['y']
            for index, label in enumerate(('short', 'middle', 'long')):
                subset = error[bands == index]
                if len(subset):
                    measurements[name]['rmse_' + label] = float(np.sqrt(np.mean(subset**2)))
    return models, measurements
