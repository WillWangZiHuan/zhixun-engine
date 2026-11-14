"""Adam continuation with an exponential moving average used for prediction.

Original live weights/moments and separate averaged prediction weights are saved.
Beta zero is exactly the original model. EMA formula follows the conventional
parameter EMA described at https://docs.pytorch.org/docs/stable/optim.html;
this is independent NumPy code, not a claimed SWA/SWAD reproduction.
"""
from pathlib import Path
import copy,json,math
import numpy as np
from baselines.model import Model
from .regularized import loss_grad

class AveragedModel:
    def __init__(self,base,beta=.99):
        if not math.isfinite(beta) or not 0<=beta<1:raise ValueError('invalid EMA decay')
        self.base=copy.deepcopy(base);self.beta=beta;self.average={k:v.copy() for k,v in base.params.items()};self.updates=0
    @property
    def meta(self):return self.base.meta
    def fit(self,x,y,epochs,sample_weights=None):
        if type(epochs) is not int or not 1<=epochs<=10000:raise ValueError('invalid epoch count')
        b=self.base;x=b.normalized(x);task=b.meta['task'];w=None if sample_weights is None else np.asarray(sample_weights,dtype=float)
        if y.shape!=(len(x),) or not len(x):raise ValueError('invalid targets')
        if w is not None and (task!='regression' or w.shape!=y.shape or not np.isfinite(w).all() or np.any(w<=0)):raise ValueError('invalid weights')
        target=(y-b.meta['target_mean'])/b.meta['target_scale'] if task=='regression' else y;losses=[]
        for _ in range(epochs):
            order=np.random.default_rng(np.random.SeedSequence([b.meta['seed'],b.meta['epochs']])).permutation(len(x));total=0.
            for start in range(0,len(x),b.meta['batch_size']):
                idx=order[start:start+b.meta['batch_size']];loss,grad=loss_grad(b.params,x[idx],target[idx],task,None if w is None else w[idx]);norm=math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
                if not math.isfinite(loss) or not math.isfinite(norm):raise ValueError('nonfinite update')
                b.meta['steps']+=1;t=b.meta['steps']
                for k,g in grad.items():
                    g*=min(1.,5./max(norm,1e-12));b.first[k]=.9*b.first[k]+.1*g;b.second[k]=.999*b.second[k]+.001*g*g
                    b.params[k]-=b.meta['learning_rate']*(b.first[k]/(1-.9**t))/(np.sqrt(b.second[k]/(1-.999**t))+1e-8)
                    self.average[k]=self.beta*self.average[k]+(1-self.beta)*b.params[k]
                self.updates+=1;total+=loss*len(idx)
            b.meta['epochs']+=1;losses.append(total/len(x))
        b.validate();return losses
    def predict(self,x):
        b=self.base;x=b.normalized(x);p=self.average;z=np.tanh(x@p['w1']+p['b1'])@p['w2']+p['b2']
        if b.meta['task']=='classification':z-=z.max(axis=1,keepdims=True);p=np.exp(z);return p/p.sum(axis=1,keepdims=True)
        return np.maximum(0,z[:,0]*b.meta['target_scale']+b.meta['target_mean'])
    def save(self,directory):
        d=Path(directory);d.mkdir(exist_ok=False);self.base.save(d/'live.npz')
        with (d/'average.npz').open('xb') as f:np.savez_compressed(f,**self.average)
        (d/'config.json').write_text(json.dumps(dict(format='zhixun.ema.v1',beta=self.beta,updates=self.updates))+'\n')
    @classmethod
    def load(cls,directory):
        d=Path(directory);c=json.loads((d/'config.json').read_text());m=cls(Model.load(d/'live.npz'),beta=c['beta'])
        if c['format']!='zhixun.ema.v1' or type(c['updates']) is not int or c['updates']<0:raise ValueError('invalid EMA metadata')
        m.updates=c['updates']
        with np.load(d/'average.npz',allow_pickle=False) as a:
            if set(a.files)!=set(m.base.params):raise ValueError('missing EMA parameters')
            m.average={k:a[k].copy() for k in a.files}
        if any(m.average[k].shape!=v.shape or not np.isfinite(m.average[k]).all() for k,v in m.base.params.items()):raise ValueError('invalid EMA weights')
        return m
