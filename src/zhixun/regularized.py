"""CPU regularization for existing contest checkpoints; inference stays unchanged.

SAM follows Foret et al. (ICLR 2021) and google-research/sam's two-gradient
procedure (Apache-2.0; source and license in verification/method-study-v1/sources).
Manifold mixing independently implements Verma et al. (ICML 2019), with the
single existing hidden layer as the eligible set; no unlicensed code copied.
Conditional input mixing follows Yao et al. (NeurIPS 2022), C-Mixup (MIT).
Changes: NumPy/Adam continuation, bounded within-batch Gaussian pairing and
interpolated source group weights. These are task adaptations, not paper replicas.
"""
import math
import numpy as np


def loss_grad(params, x, target, task, weights=None, *, partner=None, lam=1.0):
    """Return mean loss and exact gradient; optional hidden-state interpolation."""
    n = len(x)
    h = np.tanh(x @ params['w1'] + params['b1'])
    hm = h if partner is None else lam*h + (1-lam)*h[partner]
    logits = hm @ params['w2'] + params['b2']
    if task == 'classification':
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
        if target.ndim == 1:
            loss = -np.log(np.maximum(probs[np.arange(n), target], 1e-300)).mean()
            delta = probs.copy(); delta[np.arange(n), target] -= 1
        else:
            loss = -np.sum(target * np.log(np.maximum(probs, 1e-300))) / n
            delta = probs - target
        delta /= n
    else:
        error = logits[:, 0] - target
        w = np.ones(n) if weights is None else weights
        loss = np.mean(w * error**2)
        delta = (2 * error / n * w)[:, None]
    grad = {'w2': hm.T @ delta, 'b2': delta.sum(axis=0)}
    dh = delta @ params['w2'].T
    if partner is not None:
        dh0 = lam*dh
        np.add.at(dh0, partner, (1-lam)*dh)
        dh = dh0
    dz = dh * (1-h**2)
    grad.update(w1=x.T @ dz, b1=dz.sum(axis=0))
    return float(loss), grad


def conditional_partner(y, rng, bandwidth=.5):
    """Gaussian target similarity over current train minibatch, including self."""
    distance = (y[:, None] - y[None, :]) / bandwidth
    probability = np.exp(-.5 * distance**2)
    probability /= probability.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probability, axis=1); cdf[:, -1] = 1.0
    return (cdf < rng.random(len(y))[:, None]).sum(axis=1)


def fit(model, x, y, epochs, *, method='erm', strength=0., sample_weights=None,
        augmentation_seed=11, bandwidth=.5, group_ids=None, weight_decay=0.):
    if method not in ('erm', 'sam', 'manifold', 'cmixup', 'groupdro'):
        raise ValueError('unknown method')
    if type(epochs) is not int or epochs < 1 or not math.isfinite(strength) or strength < 0:
        raise ValueError('invalid regularization configuration')
    task = model.meta['task']
    if method == 'manifold' and task != 'classification':
        raise ValueError('hidden mixing is evaluated for classification only')
    if method == 'cmixup' and (task != 'regression' or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('conditional mixing requires regression and positive bandwidth')
    if not math.isfinite(weight_decay) or weight_decay < 0 or (method == 'sam' and weight_decay):
        raise ValueError('invalid or untested weight-decay configuration')
    x = model.normalized(x)
    if y.shape != (len(x),) or not len(x): raise ValueError('invalid targets')
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    if weights is not None and (task != 'regression' or weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('invalid regression weights')
    target = (y-model.meta['target_mean'])/model.meta['target_scale'] if task == 'regression' else y
    robust_state = None
    group_forward_examples = 0
    if method == 'groupdro':
        if task != 'regression' or group_ids is None: raise ValueError('groupdro requires regression groups')
        group_ids = np.asarray(group_ids, dtype=np.int64)
        if group_ids.shape != y.shape or np.any(group_ids < 0): raise ValueError('invalid groups')
        counts = np.bincount(group_ids)
        if np.any(counts == 0): raise ValueError('group IDs must be contiguous')
        robust_state = np.asarray(model.meta.get('groupdro_probs', np.ones(len(counts))/len(counts)), dtype=float)
        if robust_state.shape != counts.shape or np.any(robust_state <= 0) or not np.isfinite(robust_state).all(): raise ValueError('invalid group state')
    losses=[]; gradient_evaluations=0; updates=0
    for _ in range(epochs):
        if robust_state is not None:
            # Epoch-level adaptation of Group DRO: loss-only full-train pass, no extra gradient.
            _, before_logits = model.forward(x)
            errors = (before_logits[:, 0] - target)**2
            group_loss = np.bincount(group_ids, weights=errors)/counts
            logq = np.log(robust_state) + strength*group_loss
            logq -= logq.max(); robust_state=np.exp(logq); robust_state/=robust_state.sum()
            weights = robust_state[group_ids]*len(x)/counts[group_ids]
            group_forward_examples += len(x)
        epoch=model.meta['epochs']
        order=np.random.default_rng(np.random.SeedSequence([model.meta['seed'],epoch])).permutation(len(x))
        rng=np.random.default_rng(np.random.SeedSequence([augmentation_seed,epoch]))
        total=0.
        for start in range(0,len(x),model.meta['batch_size']):
            idx=order[start:start+model.meta['batch_size']];xb=x[idx];yb=target[idx];wb=None if weights is None else weights[idx]
            partner=None;lam=1.
            if method in ('manifold','cmixup') and strength > 0:
                lam=float(rng.beta(strength,strength))
                if method == 'manifold':
                    partner=rng.permutation(len(idx))
                    onehot=np.eye(len(model.meta['classes']))[yb]
                    yb=lam*onehot+(1-lam)*onehot[partner]
                else:
                    pair=conditional_partner(yb,rng,bandwidth)
                    xb=lam*xb+(1-lam)*xb[pair]
                    yb=lam*yb+(1-lam)*yb[pair]
                    if wb is not None: wb=lam*wb+(1-lam)*wb[pair]
            loss,grad=loss_grad(model.params,xb,yb,task,wb,partner=partner,lam=lam)
            gradient_evaluations+=1
            if method == 'sam' and strength > 0:
                norm=math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
                original={k:v.copy() for k,v in model.params.items()}
                try:
                    for k in model.params: model.params[k]=original[k]+strength*grad[k]/max(norm,1e-12)
                    _,grad=loss_grad(model.params,xb,yb,task,wb)
                    gradient_evaluations+=1
                finally:
                    model.params=original
            if weight_decay:
                loss += .5*weight_decay*sum(float(np.sum(p*p)) for p in model.params.values())
                grad={k:g+weight_decay*model.params[k] for k,g in grad.items()}
            norm=math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
            if not math.isfinite(loss) or not math.isfinite(norm): raise ValueError('nonfinite training state')
            model.meta['steps']+=1;updates+=1
            for name,gradient in grad.items():
                gradient*=min(1.,5./max(norm,1e-12))
                model.first[name]=.9*model.first[name]+.1*gradient
                model.second[name]=.999*model.second[name]+.001*gradient**2
                first=model.first[name]/(1-.9**model.meta['steps'])
                second=model.second[name]/(1-.999**model.meta['steps'])
                model.params[name]-=model.meta['learning_rate']*first/(np.sqrt(second)+1e-8)
            total+=loss*len(idx)
        model.meta['epochs']+=1;losses.append(total/len(x))
    if robust_state is not None: model.meta['groupdro_probs']=robust_state.tolist()
    model.validate()
    return dict(losses=losses,optimizer_steps=updates,gradient_evaluations=gradient_evaluations,group_forward_examples=group_forward_examples)
