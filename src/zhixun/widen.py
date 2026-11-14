"""Function-preserving zero-output expansion of an existing CPU MLP.

Inspired by function-preserving transfer (Chen et al., Net2Net, ICLR 2016,
https://arxiv.org/abs/1511.05641). This is an independent zero-output extension,
not Net2WiderNet's neuron replication. Original tensors and Adam moments are
retained; added incoming weights are random, added outgoing weights are zero.
New coordinates use a fresh Adam clock. Inference uses the original Model.
"""
import copy
import math
import numpy as np
from .regularized import loss_grad


def migrate(model, hidden, seed=11):
    model.validate()
    old = model.meta['hidden']
    if 'expansion_base_hidden' in model.meta:
        raise ValueError('only one expansion is supported')
    if type(hidden) is not int or not old <= hidden <= 128:
        raise ValueError('expanded width must be between original width and 128')
    result = copy.deepcopy(model)
    d = len(model.meta['paths']); o = model.params['b2'].size
    shapes = dict(w1=(d,hidden), b1=(hidden,), w2=(hidden,o), b2=(o,))
    for group_name in ('params','first','second'):
        source = getattr(model, group_name); target = {}
        for name, shape in shapes.items():
            target[name] = np.zeros(shape, dtype=np.float64)
            if name == 'w1': target[name][:,:old] = source[name]
            elif name in ('b1','w2'): target[name][:old] = source[name]
            else: target[name][:] = source[name]
        setattr(result, group_name, target)
    result.params['w1'][:,old:] = np.random.default_rng(seed).normal(0,1/math.sqrt(d),(d,hidden-old))
    result.meta.update(hidden=hidden,parameter_count=sum(p.size for p in result.params.values()),
                       expansion_base_hidden=old,expansion_steps=0,expansion_seed=seed)
    result.validate()
    return result


def fit(model,x,y,epochs,sample_weights=None):
    model.validate()
    old = model.meta['expansion_base_hidden']
    if type(old) is not int or not 1 <= old <= model.meta['hidden']:
        raise ValueError('invalid expansion state')
    if type(model.meta['expansion_steps']) is not int or model.meta['expansion_steps'] < 0:
        raise ValueError('invalid expansion clock')
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError('invalid epochs')
    x=model.normalized(x);task=model.meta['task']
    if y.shape != (len(x),) or not len(x): raise ValueError('invalid targets')
    target=(y-model.meta['target_mean'])/model.meta['target_scale'] if task=='regression' else y
    weights=None if sample_weights is None else np.asarray(sample_weights,dtype=np.float64)
    if weights is not None and (task!='regression' or weights.shape!=y.shape or not np.isfinite(weights).all() or np.any(weights<=0)):
        raise ValueError('invalid weights')
    masks={name:np.zeros(p.shape,dtype=bool) for name,p in model.params.items()}
    masks['w1'][:,old:]=True;masks['b1'][old:]=True;masks['w2'][old:]=True
    losses=[]
    for _ in range(epochs):
        order=np.random.default_rng(np.random.SeedSequence([model.meta['seed'],model.meta['epochs']])).permutation(len(x));total=0.
        for start in range(0,len(x),model.meta['batch_size']):
            idx=order[start:start+model.meta['batch_size']]
            loss,grad=loss_grad(model.params,x[idx],target[idx],task,None if weights is None else weights[idx])
            norm=math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
            if not math.isfinite(loss) or not math.isfinite(norm):raise ValueError('nonfinite update')
            model.meta['steps']+=1;model.meta['expansion_steps']+=1
            t=model.meta['steps'];nt=model.meta['expansion_steps']
            for name,g in grad.items():
                g*=min(1.,5./max(norm,1e-12))
                model.first[name]=.9*model.first[name]+.1*g
                model.second[name]=.999*model.second[name]+.001*g*g
                m=model.first[name]/np.where(masks[name],1-.9**nt,1-.9**t)
                v=model.second[name]/np.where(masks[name],1-.999**nt,1-.999**t)
                model.params[name]-=model.meta['learning_rate']*m/(np.sqrt(v)+1e-8)
            total+=loss*len(idx)
        model.meta['epochs']+=1;losses.append(total/len(x))
    model.validate()
    return losses
