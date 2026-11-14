"""Trusted CPU updater. Only the five numerical functions are model-editable."""
from pathlib import Path
import json,math,time,resource,os,sys
import numpy as np
from ..checkpoint import load_checkpoint, save_checkpoint
from .program import compile_program,check_gradient

def train(parent,data,source,context,output):
    output=Path(output);output.mkdir(exist_ok=False);started=time.monotonic();program=compile_program(source);model=load_checkpoint(parent,'core');task=model.meta['task'];check=check_gradient(program['loss'],task)
    with np.load(data,allow_pickle=False) as d:x=d['x'].copy();y=d['y'].copy();groups=d['groups'].copy()
    if groups.dtype.kind not in 'iu' or groups.shape!=y.shape or set(np.unique(groups))!=set(range(int(groups.max())+1)):raise ValueError('invalid training groups')
    x=model.normalized(x)
    if task=='regression':y=(y-model.meta['target_mean'])/model.meta['target_scale']
    elif y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=len(model.meta['classes'])):raise ValueError('invalid class labels')
    for a in (x,y,groups):a.setflags(write=False)
    plan=program['plan'](dict(context))
    if not isinstance(plan,dict) or set(plan)!={'epochs','batch_size'}:raise ValueError('plan fields must be epochs,batch_size')
    epochs,batch=plan['epochs'],plan['batch_size']
    if type(epochs) is not int or not 1<=epochs<=5 or type(batch) is not int or not 16<=batch<=4096:raise ValueError('training plan exceeds limits')
    weights=np.asarray(program['sample_weights'](x,y,groups,dict(context)),dtype=float)
    if weights.shape!=y.shape or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('weights must be finite nonnegative and nonempty')
    weights=weights/weights.mean();weights.setflags(write=False);before_steps=model.meta['steps'];before_epochs=model.meta['epochs'];history=[];updates=0
    for epoch in range(epochs):
        default=np.random.default_rng(np.random.SeedSequence([model.meta['seed'],model.meta['epochs']])).permutation(len(x));default.setflags(write=False)
        order=np.asarray(program['batch_order'](default,groups,y,model.meta['epochs']))
        if order.dtype.kind not in 'iu' or order.shape!=(len(x),) or not np.array_equal(np.sort(order),np.arange(len(x))):raise ValueError('batch order must be a permutation of training rows')
        total=0.;rates=[]
        for start in range(0,len(x),batch):
            if updates>=1500:raise ValueError('experiment update limit exceeded')
            idx=order[start:start+batch];xb=x[idx];yb=y[idx];wb=weights[idx];hidden,logits=model.forward(xb);pred=logits[:,0] if task=='regression' else logits
            for a in (pred,yb,wb):a.setflags(write=False)
            loss,delta=program['loss'](pred,yb,wb);delta=np.asarray(delta,dtype=float)
            if delta.shape!=pred.shape or not np.isfinite(delta).all() or not math.isfinite(float(loss)):raise ValueError('invalid loss gradient')
            matrix=delta[:,None] if task=='regression' else delta
            grad={'w2':hidden.T@matrix,'b2':matrix.sum(axis=0)};dz=(matrix@model.params['w2'].T)*(1-hidden**2);grad.update(w1=xb.T@dz,b1=dz.sum(0));norm=math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
            rate=program['learning_rate'](updates,dict(context))
            if type(rate) not in (int,float,np.float64) or not math.isfinite(rate) or not 1e-6<=rate<=.1:raise ValueError('learning rate outside [1e-6,.1]')
            model.meta['steps']+=1;updates+=1
            for k,g in grad.items():
                g=g*min(1.,5./max(norm,1e-12));model.first[k]=.9*model.first[k]+.1*g;model.second[k]=.999*model.second[k]+.001*g*g;first=model.first[k]/(1-.9**model.meta['steps']);second=model.second[k]/(1-.999**model.meta['steps']);model.params[k]-=rate*first/(np.sqrt(second)+1e-8)
            total+=float(loss)*len(idx);rates.append(float(rate))
        model.meta['epochs']+=1;history.append(dict(epoch=model.meta['epochs'],loss=total/len(x),steps=model.meta['steps'],learning_rate_first=rates[0],learning_rate_last=rates[-1]))
    model.meta['batch_size']=batch;model.meta['learning_rate']=rates[-1];model.validate();save_checkpoint(model,output/'checkpoint.npz');report=dict(complete=True,plan=plan,initial_steps=before_steps,final_steps=model.meta['steps'],initial_epochs=before_epochs,final_epochs=model.meta['epochs'],gradient_check=check,train_rows=len(x),weight_min=float(weights.min()),weight_max=float(weights.max()),wall_seconds=time.monotonic()-started,history=history)
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    case=Path(sys.argv[1]);request=json.loads((case/'job.json').read_text());allowed=sorted(os.sched_getaffinity(0))[:4] if hasattr(os,'sched_getaffinity') else None
    if allowed is not None:os.sched_setaffinity(0,set(allowed))
    seconds=int(request['cpu_seconds']);resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds+1));resource.setrlimit(resource.RLIMIT_AS,(8000000000,8000000000));resource.setrlimit(resource.RLIMIT_FSIZE,(100000000,100000000));start=time.monotonic()
    try:
        report=train(case/'parent.npz',case/'train.npz',(case/'program.py').read_text(),request['context'],case/'result');status='completed';error=None
    except Exception as e:status='failed';error={'type':type(e).__name__,'message':str(e)[:1000]}
    usage=resource.getrusage(resource.RUSAGE_SELF);result=dict(status=status,error=error,cpu_seconds=usage.ru_utime+usage.ru_stime,wall_seconds=time.monotonic()-start,max_rss_kib=usage.ru_maxrss,cpu_affinity=allowed)
    (case/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if status!='completed':sys.exit(1)
if __name__=='__main__':main()
