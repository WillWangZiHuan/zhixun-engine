"""CPU residual retrieval from approved training examples.

Retrieval motivated by TabR (Gorishniy et al., ICLR 2024;
https://github.com/yandex-research/tabular-dl-tabr). This independent NumPy
prototype is NOT TabR: it keeps a supplied MLP, retrieves residual labels using
fixed raw/hidden keys, and fits one scalar without the query's device in memory.
"""
from pathlib import Path
import copy,json
import numpy as np
from baselines.model import Model

class ResidualMemory:
    def __init__(self,base,key='input',k=16):
        if base.meta['task']!='regression' or key not in ('input','hidden') or type(k) is not int or k<1:
            raise ValueError('invalid residual memory configuration')
        self.base=copy.deepcopy(base);self.key=key;self.k=k;self.alpha=0.;self.keys=None;self.values=None

    def encode(self,x):
        z=self.base.normalized(x)
        return z if self.key=='input' else self.base.forward(z)[0]

    def correction(self,x,query_groups=None,memory_groups=None):
        if self.keys is None or self.values is None:raise ValueError('memory not fitted')
        encoded=self.encode(x);result=[];norm=np.sum(self.keys**2,axis=1)
        if (query_groups is None)!=(memory_groups is None):raise ValueError('both group arrays required')
        for start in range(0,len(encoded),128):
            q=encoded[start:start+128];d=np.maximum(0,np.sum(q*q,axis=1)[:,None]+norm[None,:]-2*q@self.keys.T)
            if query_groups is not None:d[np.asarray(query_groups)[start:start+len(q),None]==np.asarray(memory_groups)[None,:]]=np.inf
            ids=np.argpartition(d,self.k-1,axis=1)[:,:self.k];distance=np.take_along_axis(d,ids,axis=1)
            if not np.isfinite(distance).all():raise ValueError('not enough cross-device neighbors')
            bandwidth=np.maximum(distance.mean(axis=1,keepdims=True),1e-12);logw=-distance/bandwidth;logw-=logw.max(axis=1,keepdims=True);weights=np.exp(logw);weights/=weights.sum(axis=1,keepdims=True)
            result.append(np.sum(weights*self.values[ids],axis=1))
        return np.concatenate(result)

    def fit(self,x,y,groups):
        groups=np.asarray(groups);y=np.asarray(y)
        if y.shape!=(len(x),) or groups.shape!=y.shape or not np.isfinite(y).all():raise ValueError('invalid training rows')
        counts={g:int(np.sum(groups==g)) for g in np.unique(groups)}
        if len(counts)<2 or len(x)-max(counts.values())<self.k:raise ValueError('insufficient distinct devices')
        self.keys=self.encode(x);self.values=y-self.base.predict(x)
        c=self.correction(x,groups,groups);w=np.array([len(x)/(len(counts)*counts[g]) for g in groups])
        denominator=float(np.sum(w*c*c));unbounded=float(np.sum(w*c*self.values)/denominator) if denominator>0 else 0.
        self.alpha=float(np.clip(unbounded,0.,1.))
        return dict(alpha=self.alpha,unbounded_alpha=unbounded,train_cross_device_base_mse=float(np.mean(w*self.values**2)),train_cross_device_corrected_mse=float(np.mean(w*(self.values-self.alpha*c)**2)),memory_rows=len(x),memory_features=self.keys.shape[1],memory_bytes=self.keys.nbytes+self.values.nbytes)

    def predict(self,x):
        base=self.base.predict(x)
        if self.alpha==0:return base
        return np.maximum(0,base+self.alpha*self.correction(x))

    def save(self,directory):
        path=Path(directory);path.mkdir(exist_ok=False);self.base.save(path/'base.npz')
        with (path/'memory.npz').open('xb') as f:np.savez_compressed(f,keys=self.keys,values=self.values)
        (path/'config.json').write_text(json.dumps(dict(format='zhixun.residual-memory.v1',key=self.key,k=self.k,alpha=self.alpha))+'\n')

    @classmethod
    def load(cls,directory):
        path=Path(directory);c=json.loads((path/'config.json').read_text())
        if c['format']!='zhixun.residual-memory.v1' or not np.isfinite(c['alpha']) or not 0<=c['alpha']<=1:raise ValueError('invalid memory metadata')
        m=cls(Model.load(path/'base.npz'),key=c['key'],k=c['k']);m.alpha=c['alpha']
        with np.load(path/'memory.npz',allow_pickle=False) as a:m.keys=a['keys'].copy();m.values=a['values'].copy()
        features=len(m.base.meta['paths']) if m.key=='input' else m.base.meta['hidden']
        if m.keys.ndim!=2 or m.keys.shape[1]!=features or m.values.shape!=(len(m.keys),) or len(m.keys)<m.k or not np.isfinite(m.keys).all() or not np.isfinite(m.values).all():raise ValueError('invalid memory arrays')
        return m
