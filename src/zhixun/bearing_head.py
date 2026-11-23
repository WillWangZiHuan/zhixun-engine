"""Train-only vibration regimes and an affine life predictor for each regime."""
import numpy as np


class BearingHead:
    def __init__(self, base):
        self.base = base

    @property
    def meta(self):
        return self.base.meta

    def features(self, x):
        compressed = np.arcsinh(x / self.amplitude)
        return np.clip((compressed - self.location) / self.spread, -10, 10)

    def fit(self, x, y):
        from sklearn.cluster import KMeans
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.amplitude = np.maximum(np.median(np.abs(x), axis=0), 1e-9)
        scaler = StandardScaler().fit(np.arcsinh(x / self.amplitude))
        self.location, self.spread = scaler.mean_, scaler.scale_
        features = self.features(x)
        partition = KMeans(n_clusters=3, n_init=3, random_state=42).fit(features)
        self.centers = partition.cluster_centers_
        self.coefficients = np.empty_like(self.centers)
        self.bias = np.empty(3)
        for i in range(3):
            rows = partition.labels_ == i
            if not rows.any():
                # Identical centers never need distinct regressions.
                rows = np.ones(len(y), dtype=bool)
            fit = Ridge(alpha=10).fit(features[rows], y[rows])
            self.coefficients[i], self.bias[i] = fit.coef_, fit.intercept_

    def predict(self, x):
        features = self.features(x)
        distances = np.sum((features[:, None, :] - self.centers) ** 2, axis=2)
        groups = np.argmin(distances, axis=1)
        head = np.sum(features * self.coefficients[groups], axis=1) + self.bias[groups]
        return np.maximum(0, .5 * (self.base.predict(x) + head))
