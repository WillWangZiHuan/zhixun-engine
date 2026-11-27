"""Condition gradients of the inherited turbofan MLP using the training input covariance."""
import json

import numpy as np


def _condition_inputs(model, inputs, targets):
    rng = np.random.default_rng([model.meta['seed'], model.meta['steps']])
    if len(targets) > 131072:
        selected = rng.choice(len(targets), 131072, replace=False)
        inputs, targets = inputs[selected], targets[selected]
    x = model.normalized(inputs)
    design = np.column_stack((x, np.ones(len(x))))
    y = (targets - model.meta['target_mean']) / model.meta['target_scale']
    covariance = design.T @ design / len(y)
    covariance.flat[::len(covariance) + 1] += 1e-5
    preconditioner = np.linalg.solve(covariance, np.eye(len(covariance)))
    batches = max(1, len(y) // 2048)
    start_loss = float(np.mean((model.forward(x)[1][:, 0] - y)**2))
    for _ in range(2):
        for indices in np.array_split(rng.permutation(len(y)), batches):
            batch = design[indices]
            hidden, prediction = model.forward(batch[:, :-1])
            residual = prediction[:, 0] - y[indices]
            derivative = 2.0 * np.clip(residual, -8.0, 8.0) / len(indices)
            hidden_gradient = (1.0 - hidden**2) * model.params['w2'][:, 0] * derivative[:, None]
            direction = preconditioner @ (batch.T @ hidden_gradient)
            model.params['w1'] -= 0.04 * direction[:-1]
            model.params['b1'] -= 0.04 * direction[-1]
            model.params['w2'][:, 0] -= 0.04 * (hidden.T @ derivative)
            model.params['b2'] -= 0.04 * derivative.sum()
    end_loss = float(np.mean((model.forward(x)[1][:, 0] - y)**2))
    return dict(rows=len(y), passes=2, updates=2*batches, damping=1e-5,
                mean_squared_error_before=start_loss, mean_squared_error_after=end_loss)


def _limited_memory_minimize(objective, initial, max_iterations):
    """L-BFGS with descent and Armijo checks, in the training coordinate system."""
    point = initial
    loss, gradient = objective(point)
    first_loss = loss
    history = []
    performed = 0
    for _ in range(max_iterations):
        direction = gradient.copy()
        coefficients = []
        for step, change, reciprocal in reversed(history):
            coefficient = reciprocal * np.dot(step, direction)
            coefficients.append(coefficient)
            direction -= coefficient * change
        if history:
            step, change, _ = history[-1]
            direction *= np.dot(step, change) / np.dot(change, change)
        for (step, change, reciprocal), coefficient in zip(history, reversed(coefficients)):
            direction += step * (coefficient - reciprocal * np.dot(change, direction))
        direction *= -1
        slope = np.dot(gradient, direction)
        if slope >= 0:
            direction = -gradient
            slope = -np.dot(gradient, gradient)
        if abs(slope) < 1e-15:
            break
        step_size = 1.0
        for _ in range(16):
            candidate = point + step_size * direction
            candidate_loss, candidate_gradient = objective(candidate)
            if np.isfinite(candidate_loss) and candidate_loss <= loss + 1e-4 * step_size * slope:
                break
            step_size *= .5
        else:
            break
        step = candidate - point
        change = candidate_gradient - gradient
        curvature = np.dot(step, change)
        if curvature > 1e-12:
            if len(history) == 8:
                history.pop(0)
            history.append((step, change, 1.0 / curvature))
        point, loss, gradient = candidate, candidate_loss, candidate_gradient
        performed += 1
    return point, first_loss, loss, performed


def _fit_engine(model, inputs, targets, old_inputs, protect_extremes):
    if protect_extremes:
        parent_output = model.predict(inputs)
    parent_predictions = [model.predict(rows) for rows in old_inputs]
    report = _condition_inputs(model, inputs, targets)
    rng = np.random.default_rng([model.meta['seed'], model.meta['steps']])
    chosen = rng.choice(len(targets), min(len(targets), 32768), replace=False)
    x = model.normalized(inputs[chosen])
    y = (targets[chosen] - model.meta['target_mean']) / model.meta['target_scale']
    eigenvalues, rotation = np.linalg.eigh(x.T @ x / len(x))
    scales = np.sqrt(np.maximum(eigenvalues, 1e-7))
    transform = rotation / scales
    design = x @ transform
    if protect_extremes:
        target_scale = model.meta['target_scale']
        anchor = (parent_output[chosen] - model.meta['target_mean']) / target_scale
        lower, upper = np.quantile(anchor, [.1, .9])
        tails = [(anchor >= upper, 1, 10), (anchor <= lower, -1, 13)]
    width = model.meta['hidden']
    end = x.shape[1] * width
    original = model.params
    reference = model.forward(x)[1][:, 0]
    initial = np.concatenate(((scales[:, None] * (rotation.T @ original['w1'])).ravel(),
                              original['b1'], original['w2'][:, 0], original['b2']))
    if old_inputs:
        old_design = np.vstack([model.normalized(rows) @ transform for rows in old_inputs])
        old_targets = (np.concatenate(parent_predictions) - model.meta['target_mean']) / model.meta['target_scale']
        weights = np.concatenate([np.full(len(rows), 1.0 / len(rows)) for rows in old_inputs])
        group_weight = 1.0 + weights.sum()

    def loss_gradient(parameters):
        hidden = np.tanh(design @ parameters[:end].reshape(-1, width) + parameters[end:end + width])
        readout = parameters[end + width:-1]
        prediction = hidden @ readout + parameters[-1]
        residual = prediction - y
        deviation = prediction - reference
        # A fourth-power penalty restrains large changes at individual training samples.
        square = deviation * deviation
        objective = np.mean(residual**2 + 3.0 * square**2)
        derivative = (2.0 * residual + 12.0 * square * deviation) / len(y)
        if protect_extremes:
            for mask, sign, cycles in tails:
                scale = target_scale / cycles
                drift = np.maximum(sign * (prediction[mask] - anchor[mask]), 0)
                objective += .5 * scale**2 * np.mean(drift**2)
                derivative[mask] += sign * scale**2 * drift / mask.sum()
        back = (1.0 - hidden**2) * readout * derivative[:, None]
        gradient = np.concatenate(((design.T @ back).ravel(), back.sum(axis=0),
                                   hidden.T @ derivative, [derivative.sum()]))
        if old_inputs:
            old_hidden = np.tanh(old_design @ parameters[:end].reshape(-1, width) + parameters[end:end + width])
            delta = old_hidden @ readout + parameters[-1] - old_targets
            objective = (objective + weights @ (delta ** 2)) / group_weight
            slope = 2 * weights * delta
            back_old = (1 - old_hidden ** 2) * readout * slope[:, None]
            grad_old = np.concatenate(((old_design.T @ back_old).ravel(), back_old.sum(axis=0),
                                       old_hidden.T @ slope, [slope.sum()]))
            gradient = (gradient + grad_old) / group_weight
        return float(objective), gradient

    fitted, before, after, iterations = _limited_memory_minimize(loss_gradient, initial, 128)
    original['w1'][:] = transform @ fitted[:end].reshape(-1, width)
    original['b1'][:] = fitted[end:end + width]
    original['w2'][:, 0] = fitted[end + width:-1]
    original['b2'][:] = fitted[-1]
    report['bounded_mlp_fit'] = dict(training_rows=len(y), iterations=iterations,
                                    objective_before=before, objective_after=after,
                                    quartic_penalty=3.0, eigenvalue_floor=1e-7)
    return report

def operating_settings(model, inputs):
    settings = [i for i, path in enumerate(model.meta["paths"])
                if json.loads(path)[0] == ["key", "settings"]]
    spread = inputs[:, settings].std(axis=0) / model.scale[settings]
    narrow = len(settings) == 3 and bool(max(spread) < 0.1)
    return dict(normalized_std=spread.tolist(),
                update="proposal_adam" if narrow else "conditioned_fit_then_proposal_adam")


def continue_engine(model, inputs, targets, old_inputs=(), *, protect_extremes=False):
    settings = operating_settings(model, inputs)
    result = {} if settings["update"] == "proposal_adam" else _fit_engine(model, inputs, targets, old_inputs, protect_extremes)
    result['protect_prediction_extremes'] = bool(protect_extremes and settings["update"] != "proposal_adam")
    result['history_fit_rows'] = [len(rows) for rows in old_inputs] if settings["update"] != "proposal_adam" else []
    result["operating_settings"] = settings
    return result
