"""One A readout over one continued core or steady-pair model."""
import json
from pathlib import Path
import numpy as np
from baselines.model import Model
from .steady_pair import SteadyPair
from .readout_expression import Expression

FORMAT = 'zhixun.a-readout.v1'


class AReadout:
    def __init__(self, upstream, upstream_kind, expression):
        if upstream_kind not in ('core', 'steady_pair'):
            raise ValueError('Readout upstream must be core or steady_pair')
        expected = Model if upstream_kind == 'core' else SteadyPair
        if type(upstream) is not expected:
            raise ValueError('Readouts cannot nest or mismatch upstream kind')
        self.upstream, self.upstream_kind = upstream, upstream_kind
        self.expression, self.compiled = expression, Expression(expression)
        self.base.validate()
        m = self.meta
        if m.get('profile') != 'A' or m['task'] != 'classification' or len(m['paths']) != 128 or len(m['classes']) != 6 or m['hidden'] != 32:
            raise ValueError('A readout requires the 128/32/6 A checkpoint schema')
        if any(json.loads(path)[-1] != ['index', i] for i, path in enumerate(m['paths'])):
            raise ValueError('Unsupported feature order')

    @property
    def base(self):
        return self.upstream if self.upstream_kind == 'core' else self.upstream.base

    @base.setter
    def base(self, value):
        if self.upstream_kind == 'core':
            self.upstream = value
        else:
            self.upstream.base = value

    @property
    def meta(self):
        return self.base.meta

    def predict(self, raw):
        x = self.base.normalized(raw)
        hidden, logits = self.base.forward(x)
        base_prob = self.upstream.predict(raw)
        classes = np.arange(6)
        result = []
        for i in range(len(x)):
            env = dict(raw=raw[i].copy(), x=x[i].copy(), hidden=hidden[i].copy(),
                       logits=logits[i].copy(), base_prob=base_prob[i].copy(), classes=classes.copy())
            for value in env.values():
                value.setflags(write=False)
            weights = np.asarray(self.compiled(env))
            if weights.shape != (6,) or weights.dtype.kind not in 'iuf' or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError('Readout must return six finite nonnegative real weights, at least one positive')
            unit = weights / weights.max()
            result.append(unit / unit.sum())
        return np.asarray(result, dtype=float).reshape(-1, 6)

    def save(self, path):
        from .checkpoint import save_checkpoint
        save_checkpoint(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoint import load_checkpoint
        return load_checkpoint(path, 'a_readout')
