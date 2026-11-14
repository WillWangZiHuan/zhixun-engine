"""Functional residual continuation of an existing regression checkpoint.

The caller explicitly continues the core and Adam before an optional tree fit.
fit itself preserves that updated core and appends to the retained forest;
LightGBM uses y-core(x) targets with the existing forest as init_model.
"""
import copy,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from baselines.model import Model

class ResidualBoost:
    def __init__(self,base,leaves=7,min_leaf=30,rate=.05):
        if base.meta['task']!='regression':raise ValueError('regression only')
        if type(leaves) is not int or leaves<2 or type(min_leaf) is not int or min_leaf<1 or not 0<rate<=1:raise ValueError('invalid tree configuration')
        self.base=copy.deepcopy(base);self.leaves=leaves;self.min_leaf=min_leaf;self.rate=rate;self.booster=None
    @property
    def meta(self):return self.base.meta
    @property
    def iterations(self):return 0 if self.booster is None else self.booster.current_iteration()
    def predict(self,x):
        pred=self.base.predict(x)
        if self.booster is not None:pred=pred+self.booster.predict(self.base.normalized(x),num_threads=1)*self.base.meta['target_scale']
        return np.maximum(0,pred)
    def fit(self,x,y,iterations,weights=None):
        y=np.asarray(y)
        if type(iterations) is not int or iterations<1 or y.shape!=(len(x),) or not np.isfinite(y).all():raise ValueError('invalid training data')
        residual=(y-self.base.predict(x))/self.base.meta['target_scale']
        data=lgb.Dataset(self.base.normalized(x),label=residual,weight=weights)
        params=dict(objective='regression',boosting_type='gbdt',num_leaves=self.leaves,min_data_in_leaf=self.min_leaf,learning_rate=self.rate,boost_from_average=False,deterministic=True,force_col_wise=True,num_threads=1,verbosity=-1,seed=42)
        log={};before=self.iterations
        self.booster=lgb.train(params,data,num_boost_round=iterations,init_model=self.booster,valid_sets=[data],valid_names=['train'],callbacks=[lgb.record_evaluation(log)])
        return dict(tree_iterations_before=before,tree_iterations_after=self.iterations,train_residual_mse=log['train']['l2'],core_adam_steps_unchanged=self.base.meta['steps'])
    def save(self, path):
        from .checkpoint import save_checkpoint
        save_checkpoint(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoint import load_checkpoint
        return load_checkpoint(path, 'residual_boost')
