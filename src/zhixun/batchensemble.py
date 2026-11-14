"""Function-preserving CPU BatchEnsemble adaptation of official MLP checkpoints.

Inspired by BatchEnsemble (ICLR 2020) and TabM (ICLR 2025), not a reproduction
of TabM's full architecture. Rank-one scaling follows LinearBatchEnsemble in
https://github.com/yandex-research/tabm (Apache-2.0).
Changes: shared original biases; all scales start at one; original shared Adam
state retained; fresh adapter Adam clock; per-member minibatch bootstraps.
"""
import argparse,copy,json,math,sys
from pathlib import Path
import numpy as np
BASE=('w1','b1','w2','b2');ADAPTER=('r1','s1','r2','s2');FORMAT='zhixun.cpu-batchensemble.v1'


def forward(params,x):
    xr=x[:,None,:]*params['r1'][None,:,:]
    u=xr@params['w1'];h=np.tanh(u*params['s1'][None,:,:]+params['b1'])
    hr=h*params['r2'][None,:,:];v=hr@params['w2']
    z=v*params['s2'][None,:,:]+params['b2']
    return z,(xr,u,h,hr,v)


def loss_grad(params,x,y,task,weights=None,bootstrap=None):
    z,(xr,u,h,hr,v)=forward(params,x);n,k,o=z.shape
    w=np.ones((n,k)) if bootstrap is None else bootstrap.copy()
    if weights is not None:w*=weights[:,None]
    if task=='classification':
        z=z-z.max(axis=2,keepdims=True);p=np.exp(z);p/=p.sum(axis=2,keepdims=True)
        sel=(np.arange(n)[:,None],np.arange(k)[None,:],y[:,None])
        loss=-np.sum(w*np.log(np.maximum(p[sel],1e-300)))/(n*k)
        delta=p.copy();delta[sel]-=1;delta*=w[:,:,None]/(n*k)
    else:
        err=z[:,:,0]-y[:,None];loss=np.sum(w*err**2)/(n*k)
        delta=(2*err*w/(n*k))[:,:,None]
    dv=delta*params['s2'][None,:,:]
    dhr=dv@params['w2'].T
    dz=(dhr*params['r2'][None,:,:])*(1-h*h)
    du=dz*params['s1'][None,:,:]
    dxr=du@params['w1'].T
    grad={'w2':hr.reshape(-1,hr.shape[-1]).T@dv.reshape(-1,o),'b2':delta.sum(axis=(0,1)),
          's2':np.sum(delta*v,axis=0),'r2':np.sum(dhr*h,axis=0),
          'w1':xr.reshape(-1,xr.shape[-1]).T@du.reshape(-1,du.shape[-1]),'b1':dz.sum(axis=(0,1)),
          's1':np.sum(dz*u,axis=0),'r1':np.sum(dxr*x[:,None,:],axis=0)}
    return float(loss),grad


class Ensemble:
    @classmethod
    def migrate(cls,model,k=4,seed=11,bootstrap=True,adapter_lr_ratio=1.,train_adapters=True):
        if type(k) is not int or not 1<=k<=16:raise ValueError('k outside 1..16')
        if not math.isfinite(adapter_lr_ratio) or adapter_lr_ratio<0:raise ValueError('invalid adapter rate')
        m=cls();m.meta=copy.deepcopy(model.meta)
        m.meta.update(format=FORMAT,k=k,ensemble_seed=seed,bootstrap=bool(bootstrap),adapter_lr_ratio=adapter_lr_ratio,train_adapters=bool(train_adapters),adapter_steps=0)
        m.params={n:model.params[n].copy() for n in BASE}
        d,h=model.params['w1'].shape;o=model.params['w2'].shape[1]
        for n,shape in [('r1',(k,d)),('s1',(k,h)),('r2',(k,h)),('s2',(k,o))]:m.params[n]=np.ones(shape)
        m.first={n:model.first[n].copy() for n in BASE}|{n:np.zeros_like(m.params[n]) for n in ADAPTER}
        m.second={n:model.second[n].copy() for n in BASE}|{n:np.zeros_like(m.params[n]) for n in ADAPTER}
        m.mean=model.mean.copy();m.scale=model.scale.copy();m.meta['parameter_count']=sum(v.size for v in m.params.values());m.validate();return m

    def validate(self):
        m=self.meta
        if m['format']!=FORMAT or m['task'] not in ('classification','regression') or not 1<=m['k']<=16:raise ValueError('invalid ensemble metadata')
        d=len(m['paths']);h=m['hidden'];o=len(m['classes']) if m['task']=='classification' else 1;k=m['k']
        shapes={'w1':(d,h),'b1':(h,),'w2':(h,o),'b2':(o,),'r1':(k,d),'s1':(k,h),'r2':(k,h),'s2':(k,o)}
        for state in [self.params,self.first,self.second]:
            if set(state)!=set(shapes):raise ValueError('invalid state keys')
            if any(state[n].shape!=shape or not np.isfinite(state[n]).all() or state[n].dtype!=np.float64 for n,shape in shapes.items()):raise ValueError('invalid tensor state')
        if any(np.any(v<0) for v in self.second.values()):raise ValueError('invalid second moment')
        if self.mean.shape!=(d,) or self.scale.shape!=(d,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0):raise ValueError('invalid scaler')
        if sum(v.size for v in self.params.values())!=m['parameter_count']:raise ValueError('parameter count mismatch')
        for key in ('steps','epochs','adapter_steps','ensemble_seed'):
            if type(m[key]) is not int or m[key]<0:raise ValueError('invalid counter')

    def normalized(self,x):
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('invalid inputs')
        return np.clip((x-self.mean)/self.scale,-10,10)

    def member_outputs(self,x):
        z,_=forward(self.params,self.normalized(x))
        if self.meta['task']=='classification':
            z-=z.max(axis=2,keepdims=True);p=np.exp(z);return p/p.sum(axis=2,keepdims=True)
        return z[:,:,0]*self.meta['target_scale']+self.meta['target_mean']

    def predict(self,x):
        # Bound inference memory without altering the prediction function.
        chunks=[self.member_outputs(x[i:i+2048]).mean(axis=1) for i in range(0,len(x),2048)]
        out=np.concatenate(chunks,axis=0)
        return out if self.meta['task']=='classification' else np.maximum(out,0)

    def fit(self,x,y,epochs,sample_weights=None):
        if type(epochs) is not int or epochs<1:raise ValueError('invalid epochs')
        x=self.normalized(x);n=len(x);k=self.meta['k'];task=self.meta['task']
        if y.shape!=(n,) or n==0:raise ValueError('target mismatch')
        weights=None if sample_weights is None else np.asarray(sample_weights,dtype=float)
        if weights is not None and (weights.shape!=(n,) or np.any(weights<=0) or not np.isfinite(weights).all()):raise ValueError('invalid weights')
        target=(y-self.meta['target_mean'])/self.meta['target_scale'] if task=='regression' else y
        losses=[];steps_before=self.meta['steps']
        for _ in range(epochs):
            epoch=self.meta['epochs'];order=np.random.default_rng(np.random.SeedSequence([self.meta['seed'],epoch])).permutation(n)
            rng=np.random.default_rng(np.random.SeedSequence([self.meta['ensemble_seed'],epoch]));total=0.
            for start in range(0,n,self.meta['batch_size']):
                ids=order[start:start+self.meta['batch_size']];count=len(ids);boot=None
                if self.meta['bootstrap']:
                    boot=np.stack([np.bincount(rng.integers(0,count,count),minlength=count) for _ in range(k)],axis=1).astype(float)
                loss,g=loss_grad(self.params,x[ids],target[ids],task,None if weights is None else weights[ids],boot)
                names=BASE+ADAPTER if self.meta['train_adapters'] else BASE
                norm=math.sqrt(sum(float(np.sum(g[t]**2)) for t in names))
                if not math.isfinite(loss) or not math.isfinite(norm):raise ValueError('nonfinite training state')
                self.meta['steps']+=1
                if self.meta['train_adapters']:self.meta['adapter_steps']+=1
                for name in names:
                    grad=g[name]*min(1.,5./max(norm,1e-12));self.first[name]=.9*self.first[name]+.1*grad;self.second[name]=.999*self.second[name]+.001*grad**2
                    step=self.meta['steps'] if name in BASE else self.meta['adapter_steps']
                    first=self.first[name]/(1-.9**step);second=self.second[name]/(1-.999**step)
                    rate=self.meta['learning_rate']*(1 if name in BASE else self.meta['adapter_lr_ratio'])
                    self.params[name]-=rate*first/(np.sqrt(second)+1e-8)
                total+=loss*count
            self.meta['epochs']+=1;losses.append(total/n)
        self.validate();return losses

    def save(self,path):
        self.validate();a={**self.params,**{'first_'+k:v for k,v in self.first.items()},**{'second_'+k:v for k,v in self.second.items()},'mean':self.mean,'scale':self.scale,'metadata':np.array(json.dumps(self.meta,sort_keys=True,allow_nan=False))}
        with Path(path).open('xb') as f:np.savez_compressed(f,**a)

    @classmethod
    def load(cls,path):
        if Path(path).stat().st_size>16*1024*1024:raise ValueError('checkpoint too large')
        with np.load(path,allow_pickle=False) as a:
            m=cls();m.meta=json.loads(str(a['metadata']));m.params={k:a[k].copy() for k in BASE+ADAPTER};m.first={k:a['first_'+k].copy() for k in BASE+ADAPTER};m.second={k:a['second_'+k].copy() for k in BASE+ADAPTER};m.mean=a['mean'].copy();m.scale=a['scale'].copy()
        m.validate();return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args();m=Ensemble.load(args.checkpoint)
    from baselines.data import flatten
    for line in sys.stdin:
        if not line.strip():continue
        item=json.loads(line);paths,x=flatten(item['inputs'])
        if paths!=m.meta['paths'] or item['task']!=m.meta['task']:raise ValueError('request does not match checkpoint')
        pred=m.predict(np.array([x],dtype=float))[0]
        out=[dict(label=c,score=float(v)) for c,v in zip(m.meta['classes'],pred)] if m.meta['task']=='classification' else {'value':float(pred)}
        print(json.dumps({'outputs':out}),flush=True)
if __name__=='__main__':main()
