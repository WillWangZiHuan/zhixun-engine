"""Capability-limited numerical Python: editable training logic, no I/O APIs."""
import ast
import builtins
import json
import numpy as np

SIGNATURES={'plan':['context'],'learning_rate':['step','context'],
 'sample_weights':['x','y','groups','context'],'batch_order':['order','groups','y','epoch'],
 'loss':['pred','target','weights']}
NUMPY={'array','asarray','ones','zeros','ones_like','zeros_like','full_like','arange','abs',
 'maximum','minimum','clip','where','mean','sum','std','median','quantile','percentile',
 'exp','log','log1p','max','sqrt','square','power','tanh','sin','cos','sign','bincount','unique',
 'argsort','sort','concatenate','stack','repeat','tile','reshape','isfinite','all','any'}
BUILTINS={'len','float','int','min','max','abs','sum'}
ALLOWED={ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.If,
 ast.Expr,ast.Name,ast.Load,ast.Store,ast.Constant,ast.BinOp,ast.UnaryOp,ast.Compare,
 ast.BoolOp,ast.IfExp,ast.Call,ast.Attribute,ast.Subscript,ast.Slice,ast.Tuple,ast.List,
 ast.Dict,ast.keyword,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,
 ast.MatMult,ast.USub,ast.UAdd,ast.Not,ast.Invert,ast.And,ast.Or,ast.BitAnd,ast.BitOr,
 ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq,ast.Is,ast.IsNot,ast.In,ast.NotIn}
DEFAULT='''def plan(context):
    return {"epochs": 1, "batch_size": 64}

def learning_rate(step, context):
    return context["suggested_learning_rate"]

def sample_weights(x, y, groups, context):
    counts = np.bincount(groups)
    return len(y) / (len(counts) * counts[groups])

def batch_order(order, groups, y, epoch):
    return order

def loss(pred, target, weights):
    error = pred - target
    return np.mean(weights * error ** 2), 2 * weights * error / len(error)
'''

def validate(source):
    parents={}
    def reject(reason,node=None,syntax=None):
        location=node
        while location is not None and not hasattr(location,'lineno'):location=parents.get(id(location))
        def column(line,offset):
            if line is None or offset is None:return None
            return len(source.splitlines()[line-1].encode('utf-8')[:offset].decode('utf-8'))+1
        line=getattr(location,'lineno',None);end_line=getattr(location,'end_lineno',None)
        col=column(line,getattr(location,'col_offset',None));end_col=column(end_line,getattr(location,'end_col_offset',None))
        segment=ast.get_source_segment(source,location) if location is not None else None
        if syntax is not None:
            line,col,end_line,end_col=syntax.lineno,syntax.offset,syntax.end_lineno,syntax.end_offset
            segment=syntax.text.rstrip('\n') if syntax.text else None
        # Positions are 1-based character offsets; the end is exclusive.
        detail=dict(reason=reason,node_type=type(node).__name__ if node is not None else 'SyntaxError' if syntax else None,
            line=line,column=col,end_line=end_line,end_column=end_col,
            source_segment=segment[:256] if segment is not None else None,
            allowed_call_forms='np.<listed function>(...) or <listed builtin>(...); see request.numerical_functions. Array method calls are not allowed.')
        if location is not None and location is not node:detail['location_node_type']=type(location).__name__
        if segment is not None and len(segment)>256:detail['source_segment_truncated']=True
        message=json.dumps(detail,ensure_ascii=False,separators=(',',':'))
        if syntax is not None:
            syntax.msg=message
            raise syntax
        raise ValueError(message)
    if not isinstance(source,str) or len(source)>12000:reject('program size limit')
    try:tree=ast.parse(source)
    except SyntaxError as error:reject(error.msg,syntax=error)
    parents={id(child):parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    if len(list(ast.walk(tree)))>2000:reject('program AST size limit',tree)
    if len(tree.body)!=len(SIGNATURES) or any(not isinstance(n,ast.FunctionDef) for n in tree.body):reject('exactly five pure numerical functions required',tree)
    if {n.name for n in tree.body}!=set(SIGNATURES):reject('required function names differ',tree)
    for f in tree.body:
        if [a.arg for a in f.args.args]!=SIGNATURES[f.name] or f.args.defaults or f.args.kw_defaults or f.args.kwonlyargs or f.args.posonlyargs or f.args.vararg or f.args.kwarg or f.decorator_list or f.returns or any(a.annotation for a in f.args.args):reject('fixed unannotated signatures required',f)
    top_ids={id(n) for n in tree.body}
    for n in ast.walk(tree):
        if type(n) not in ALLOWED:reject('unsupported syntax: '+type(n).__name__,n)
        if isinstance(n,ast.FunctionDef) and id(n) not in top_ids:reject('nested functions forbidden',n)
        if isinstance(n,ast.Name) and n.id.startswith('_'):reject('private names forbidden',n)
        if isinstance(n,ast.Attribute):
            allowed_np=isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr in NUMPY|{'pi'}
            if not allowed_np and n.attr!='shape':reject('attribute capability not allowed',n)
        if isinstance(n,ast.Call):
            ok=isinstance(n.func,ast.Name) and n.func.id in BUILTINS
            ok=ok or isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr in NUMPY
            if not ok or any(k.arg is None for k in n.keywords):reject('call capability not allowed',n)
        if isinstance(n,ast.Assign):
            for target in n.targets:
                leaves=list(target.elts) if isinstance(target,(ast.Tuple,ast.List)) else [target]
                if any(not isinstance(t,ast.Name) or t.id in BUILTINS|{'np'} or t.id.startswith('_') for t in leaves):reject('only local name assignments allowed',n)
        if isinstance(n,ast.Expr) and not isinstance(n.value,ast.Constant):reject('no standalone effects',n)
    return tree

def compile_program(source):
    tree=validate(source);space={'__builtins__':{},'np':np,**{k:getattr(builtins,k) for k in BUILTINS}}
    exec(compile(tree,'<training_program>','exec'),space)
    return {name:space[name] for name in SIGNATURES}

def check_gradient(fn, task="regression"):
    pred=np.array([-1.7,-.11,.31,1.3,2.4]);target=np.array([.2,-.7,.1,.4,1.1]);w=np.array([.2,.5,1.,2.,1.3])
    if task == 'classification':
        pred=np.array([[.3,-.7,1.2,.1,-.2,.6],[-.2,.4,.9,-.3,.1,.5],[1.1,-.4,.2,.7,-.6,.3],[.2,.3,-.8,.4,1.3,-.1],[.7,.1,.2,-.9,.4,1.1]])
        target=np.array([0,1,2,4,5])
    elif task != 'regression':raise ValueError('unsupported task')
    for a in (pred,target,w):a.setflags(write=False)
    value,grad=fn(pred,target,w);grad=np.asarray(grad,dtype=float)
    if grad.shape!=pred.shape or not np.isfinite(grad).all() or not np.isfinite(value):raise ValueError('nonfinite loss/gradient or incorrect shape')
    numeric=np.empty_like(pred)
    for i in np.ndindex(pred.shape):
        plus=pred.copy();minus=pred.copy();plus[i]+=1e-5;minus[i]-=1e-5;plus.setflags(write=False);minus.setflags(write=False)
        numeric[i]=(float(fn(plus,target,w)[0])-float(fn(minus,target,w)[0]))/2e-5
    if not np.allclose(grad,numeric,rtol=2e-4,atol=2e-5):raise ValueError('loss gradient disagrees with finite differences')
    return dict(loss=float(value),max_gradient_error=float(np.max(np.abs(grad-numeric))))


def default_program(task, batch_size):
    """Official cross entropy or explicit row-weighted regression reference."""
    if task not in ('classification','regression'):raise ValueError('unsupported task')
    if type(batch_size) is not int or not 16 <= batch_size <= 4096:raise ValueError('invalid batch size')
    source=DEFAULT.replace('"batch_size": 64', '"batch_size": '+str(batch_size))
    source=source.replace('    counts = np.bincount(groups)\n    return len(y) / (len(counts) * counts[groups])','    return np.ones_like(y)')
    if task=='classification':
        source=source[:source.index('def loss(')]+"""def loss(pred, target, weights):
    shifted = pred - np.max(pred, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probability = exponent / np.sum(exponent, axis=1, keepdims=True)
    onehot = np.arange(pred.shape[1])[None, :] == target[:, None]
    value = np.mean(weights * (np.log(np.sum(exponent, axis=1)) - shifted[np.arange(len(target)), target]))
    return value, weights[:, None] * (probability - onehot) / len(target)
"""
    validate(source)
    return source
