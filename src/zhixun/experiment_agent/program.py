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

def weighted_cross_entropy(pred, target, weights, gamma=0.0):
    """Mean weighted cross entropy, optionally focal, and its exact logit gradient."""
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    shifted = pred - np.max(pred, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    normalizer = exp_logits.sum(axis=1)
    rows = np.arange(len(target))
    ce = np.log(normalizer) - shifted[rows, target]
    gradient = exp_logits / normalizer[:, None]
    gradient[rows, target] -= 1
    if gamma > 0:
        # 1 - exp(-ce) also remains accurate for confidently classified examples.
        complement = -np.expm1(-ce)
        factor = complement ** gamma
        ratio = np.divide(ce, complement, out=np.ones_like(ce), where=complement > 0)
        derivative = factor * (1 + gamma * np.exp(-ce) * ratio)
        gradient *= derivative[:, None]
        ce = ce * factor
    return np.mean(weights * ce), gradient * (weights / len(target))[:, None]


def weighted_mse(pred, target, weights):
    """Mean weighted squared error and one derivative per prediction."""
    error = pred - target
    return np.mean(weights * error ** 2), (2.0 / len(target)) * weights * error


def weighted_huber(pred, target, weights, delta):
    """Mean weighted Huber loss and one derivative per prediction."""
    if not np.isscalar(delta) or not np.isfinite(delta) or delta <= 0:
        raise ValueError('Huber delta must be finite and positive')
    error = pred - target
    magnitude = np.abs(error)
    quadratic = np.minimum(magnitude, delta)
    terms = 0.5 * quadratic ** 2 + delta * (magnitude - quadratic)
    return np.mean(weights * terms), (weights / len(target)) * np.clip(error, -delta, delta)


LOSS_OPERATORS = {fn.__name__: fn for fn in (weighted_cross_entropy, weighted_mse, weighted_huber)}


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
            allowed_call_forms='np.<listed function>(...) or <listed builtin or loss operator>(...); see request.numerical_functions. Array method calls are not allowed.')
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
    # Class-wise expressions iterate only over the six fixed label indices.
    # Compile the iterator as a constant tuple without exposing a range builtin.
    weight_fn = next(f for f in tree.body if f.name == 'sample_weights')
    class_nodes = set()
    for expr in ast.walk(weight_fn):
        if not isinstance(expr, ast.ListComp):
            continue
        if len(expr.generators) != 1 or any(isinstance(n, ast.ListComp) for n in ast.walk(expr.elt)):
            reject('class-wise expressions require a single iterator', expr)
        gen = expr.generators[0]
        target = gen.target
        if (gen.is_async or gen.ifs or not isinstance(target, ast.Name)
                or target.id in BUILTINS | LOSS_OPERATORS.keys() | {'np', 'range'}
                or target.id in SIGNATURES['sample_weights']):
            reject('class-wise iteration requires a local index and no filters', expr)
        call = gen.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'range' and not call.keywords and len(call.args) == 1
                and isinstance(call.args[0], ast.Constant) and type(call.args[0].value) is int
                and call.args[0].value == 6):
            reject('class-wise iteration requires literal range(6)', expr)
        gen.iter = ast.copy_location(ast.Constant(value=tuple(range(6))), call)
        class_nodes.update((expr, gen))
    if class_nodes:
        for node in ast.walk(weight_fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == 'range':
                reject('class-wise iteration requires an unshadowed range', node)
    top_ids={id(n) for n in tree.body}
    for n in ast.walk(tree):
        if type(n) not in ALLOWED and n not in class_nodes:reject('unsupported syntax: '+type(n).__name__,n)
        if isinstance(n,ast.FunctionDef) and id(n) not in top_ids:reject('nested functions forbidden',n)
        if isinstance(n,ast.Name) and n.id.startswith('_'):reject('private names forbidden',n)
        if isinstance(n,ast.Attribute):
            allowed_np=isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr in NUMPY|{'pi'}
            if not allowed_np and n.attr!='shape':reject('attribute capability not allowed',n)
        if isinstance(n,ast.Call):
            ok=isinstance(n.func,ast.Name) and (n.func.id in BUILTINS or n.func.id in LOSS_OPERATORS)
            ok=ok or isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr in NUMPY
            if not ok or any(k.arg is None for k in n.keywords):reject('call capability not allowed',n)
        if isinstance(n,ast.Assign):
            for target in n.targets:
                leaves=list(target.elts) if isinstance(target,(ast.Tuple,ast.List)) else [target]
                if any(not isinstance(t,ast.Name) or (t.id in BUILTINS or t.id in LOSS_OPERATORS or t.id=='np') or t.id.startswith('_') for t in leaves):reject('only local name assignments allowed',n)
        if isinstance(n,ast.Expr) and not isinstance(n.value,ast.Constant):reject('no standalone effects',n)
    return tree

def compile_program(source):
    tree=validate(source);space={'__builtins__':{},'np':np,**{k:getattr(builtins,k) for k in BUILTINS},**LOSS_OPERATORS}
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
    operator = 'weighted_cross_entropy' if task == 'classification' else 'weighted_mse'
    source = source[:source.index('def loss(')] + (
        'def loss(pred, target, weights):\n'
        f'    return {operator}(pred, target, weights)\n'
    )
    validate(source)
    return source
