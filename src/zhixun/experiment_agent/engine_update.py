"""Condition gradients of the inherited turbofan MLP using the training input covariance."""
import numpy as np


def continue_engine(model, inputs, targets):
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
