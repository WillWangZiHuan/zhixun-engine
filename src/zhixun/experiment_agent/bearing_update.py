"""Take damped Gauss-Newton steps on the inherited bearing MLP, on training rows only."""
import numpy as np


def continue_bearing(model, inputs, targets):
    # Bound the largest matrix by training rows; feature and parameter dimensions
    # come from the official parent. Sampling only applies to a larger training stage.
    if len(targets) > 8192:
        seeded = np.random.default_rng([model.meta['seed'], model.meta['steps']])
        indices = seeded.choice(len(targets), 8192, replace=False)
        inputs, targets = inputs[indices], targets[indices]
    x = model.normalized(inputs)
    y = (targets - model.meta['target_mean']) / model.meta['target_scale']
    lifetime_weights = np.clip(np.exp(np.clip(y, -12.0, 12.0)), 0.2, 8.0)
    trace = []
    for _ in range(6):
        hidden, prediction = model.forward(x)
        error = prediction[:, 0] - y
        weights = lifetime_weights / np.maximum(1.0, np.abs(error))
        weights /= weights.sum()

        # Each column is the derivative of this MLP's scalar output with
        # respect to one existing parameter, in the order listed below.
        hidden_gradient = (1.0 - hidden**2) * model.params['w2'][:, 0]
        first_layer = (x[:, :, None] * hidden_gradient[:, None, :]).reshape(len(x), -1)
        jacobian = np.column_stack((hidden, np.ones(len(x)), first_layer, hidden_gradient))
        curvature = jacobian.T @ (weights[:, None] * jacobian)
        curvature.flat[::len(curvature) + 1] += 1e-5
        direction = np.linalg.solve(curvature, -(jacobian.T @ (weights * error)))

        names = ('w2', 'b2', 'w1', 'b1')
        previous = {name: model.params[name].copy() for name in names}
        before = float(weights @ (error**2))
        for power in range(9):
            rate = 0.5**power
            offset = 0
            for name in names:
                parameter = model.params[name]
                delta = direction[offset:offset + parameter.size].reshape(parameter.shape)
                parameter[:] = previous[name] + rate * delta
                offset += parameter.size
            residual = model.forward(x)[1][:, 0] - y
            after = float(weights @ (residual**2))
            if after < before:
                trace.append(dict(step_size=rate, training_loss_before=before, training_loss_after=after))
                break
        else:
            for name in names:
                model.params[name][:] = previous[name]
            break
    return dict(rows=len(y), damping=1e-5, accepted_steps=trace)
