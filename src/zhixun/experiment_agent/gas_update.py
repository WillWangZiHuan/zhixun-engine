"""Learn the confusion pair inside the existing MLP and anchor its other training logits."""
import numpy as np


def continue_gas(model, inputs, targets):
    steady = np.arange(inputs.shape[1]) % 8 == 1
    pair = [model.meta['classes'].index(label) for label in ('Ethanol', 'Acetaldehyde')]
    rng = np.random.default_rng([model.meta['seed'], model.meta['steps']])
    # Keep costs bounded when a stage exposes more training rows.
    groups = []
    for label in range(len(model.meta['classes'])):
        indices = np.flatnonzero(targets == label)
        if len(indices) > 512:
            indices = rng.choice(indices, 512, replace=False)
        groups.append(indices)
    selected = np.sort(np.concatenate(groups))
    x = model.normalized(inputs[selected])
    y = targets[selected]
    paired = np.isin(y, pair)
    counts = [int(np.sum(y == label)) for label in pair]
    report = dict(updates=0, pair_rows=counts, anchor_rows=int(np.sum(~paired)))
    if min(counts) == 0 or report['anchor_rows'] == 0:
        return report
    features = x[paired].copy()
    features[:, ~steady] = 0
    labels = (y[paired] == pair[1]).astype(int)
    weights = 0.5 / np.asarray(counts)[labels]
    design = np.column_stack((features[:, steady], np.ones(len(features))))
    gram = design.T @ (weights[:, None] * design)
    gram.flat[::len(gram)+1] += 1e-4
    condition = np.linalg.solve(gram, np.eye(len(gram)))
    anchors = x[~paired]
    anchor_design = np.column_stack((anchors[:, steady], np.ones(len(anchors))))
    original_logits = model.forward(anchors)[1]
    for _ in range(96):
        hidden, output = model.forward(features)
        logits = output[:, pair]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(labels)), labels] -= 1.0
        delta = probabilities * weights[:, None]
        hidden_delta = (delta @ model.params['w2'][:, pair].T) * (1.0-hidden**2)
        anchor_hidden, anchor_logits = model.forward(anchors)
        anchor_delta = (anchor_logits - original_logits) * (0.1 / len(anchors))
        anchor_gradient = (anchor_delta @ model.params['w2'].T) * (1.0 - anchor_hidden**2)
        direction = condition @ (design.T @ hidden_delta + anchor_design.T @ anchor_gradient)
        direction *= min(1.0, 20.0 / max(1e-12, np.linalg.norm(direction)))
        model.params['w1'][steady] -= 0.15 * direction[:-1]
        model.params['b1'] -= 0.15 * direction[-1]
        model.params['w2'] -= 0.15 * (anchor_hidden.T @ anchor_delta)
        model.params['b2'] -= 0.15 * anchor_delta.sum(axis=0)
        model.params['w2'][:, pair] -= 0.15 * (hidden.T @ delta)
        model.params['b2'][pair] -= 0.15 * delta.sum(axis=0)
    report['updates'] = 96
    return report
