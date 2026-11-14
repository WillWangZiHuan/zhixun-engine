"""Source-trained steady-view logit evidence for a diagnosed gas confusion pair.

The original MLP and its optimizer are retained. Only its predicted confusion
pair is routed to a two-class product of original and supplementary evidence.
The pair is an explicit development design choice, not inferred from test labels.
"""
from pathlib import Path
import copy,json
import numpy as np
from baselines.model import Model

class SteadyPair:
    def __init__(self,base):
        if base.meta['task']!='classification' or len(base.meta['paths'])!=128:raise ValueError('requires official gas classification features')
        for i,path in enumerate(base.meta['paths']):
            if json.loads(path)[-1]!=['index',i]:raise ValueError('unsupported sensor feature order')
        self.base=copy.deepcopy(base);self.indices=np.arange(1,128,8);self.pair=np.array([base.meta['classes'].index('Ethanol'),base.meta['classes'].index('Acetaldehyde')]);self.mean=None;self.scale=None;self.coef=None;self.intercept=None
    @property
    def meta(self):return self.base.meta
    def fit(self,x,y,epochs=1):
        losses=self.base.fit(x,y,epochs)
        self.fit_head(x,y)
        return losses
    def fit_head(self,x,y):
        if self.coef is None:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            z=self.base.normalized(x)[:,self.indices];scaler=StandardScaler().fit(z);head=LogisticRegression(C=1,max_iter=3000,class_weight='balanced').fit(scaler.transform(z),y)
            if not np.array_equal(head.classes_,np.arange(len(self.meta['classes']))):raise ValueError('source training must cover every class')
            self.mean=scaler.mean_.copy();self.scale=scaler.scale_.copy();self.coef=head.coef_[self.pair].copy();self.intercept=head.intercept_[self.pair].copy()
            probabilities=head.predict_proba(scaler.transform(z));counts=np.bincount(y,minlength=len(head.classes_));weights=len(y)/(len(counts)*counts[y])
            return dict(fitted=True,rows=len(y),class_counts=counts.tolist(),optimizer_iterations=int(head.n_iter_.max()),max_optimizer_iterations=3000,balanced_cross_entropy_before=float(np.log(len(counts))),balanced_cross_entropy_after=float(np.average(-np.log(np.maximum(probabilities[np.arange(len(y)),y],1e-300)),weights=weights)),loss_scope='six-class supplementary head, regularization excluded; initial uniform and final objective values, not every optimizer iterate')
    def predict(self,x):
        result=self.base.predict(x)
        if self.coef is None:return result
        z=self.base.normalized(x);logits=self.base.forward(z)[1];route=np.isin(logits.argmax(1),self.pair)
        if np.any(route):
            view=((z[route][:,self.indices]-self.mean)/self.scale)@self.coef.T+self.intercept
            combined=logits[route][:,self.pair]+view;combined-=combined.max(1,keepdims=True);p=np.exp(combined);p/=p.sum(1,keepdims=True);result[route]=0.;result[np.ix_(route,self.pair)]=p
        return result
    def save(self, path):
        from .checkpoint import save_checkpoint
        save_checkpoint(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoint import load_checkpoint
        return load_checkpoint(path, 'steady_pair')
