"""Bounded row-local expressions; no eval, I/O, labels, or mutable state."""
import ast
import operator
import numpy as np

FUNCTIONS={name:getattr(np,name) for name in ('abs','square','sqrt','log1p','exp','tanh','sin','cos','minimum','maximum','clip','where','mean','std','sum','max','min','argmax','concatenate','stack')}
BINARY={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.BitAnd:operator.and_,ast.BitOr:operator.or_}
COMPARE={ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Eq:operator.eq,ast.NotEq:operator.ne}
NAMES={'raw','x','hidden','logits','base_prob','classes'}
LIMIT=4096

def bounded(value):
    if isinstance(value,(list,tuple)):
        if len(value)>256:raise ValueError('sequence size limit')
        for v in value:bounded(v)
    elif isinstance(value,(np.ndarray,np.generic,int,float,bool)):
        a=np.asarray(value)
        if a.size>LIMIT or a.ndim>2 or a.dtype.kind not in 'biuf' or not np.isfinite(a).all():raise ValueError('nonfinite or oversized intermediate')
    else:raise ValueError('unsupported expression value')
    return value

class Expression:
    def __init__(self,source):
        if not isinstance(source,str) or not 1<=len(source)<=1800:raise ValueError('expression length')
        self.tree=ast.parse(source,mode='eval').body;self.names=NAMES
        nodes=list(ast.walk(self.tree))
        if len(nodes)>400:raise ValueError('expression node limit')
        allowed={ast.Expression,ast.Constant,ast.Name,ast.Load,ast.BinOp,ast.UnaryOp,ast.Compare,ast.Subscript,ast.Slice,ast.List,ast.Tuple,ast.Call,ast.Attribute,ast.keyword,ast.USub,ast.UAdd,ast.Invert,ast.Not,*BINARY,*COMPARE}
        for n in nodes:
            if type(n) not in allowed:raise ValueError('unsupported expression syntax '+type(n).__name__)
            if isinstance(n,ast.Name) and n.id not in self.names|{'np'}:raise ValueError('unknown expression name '+n.id)
            if isinstance(n,ast.Constant) and (type(n.value) not in (int,float,bool,type(None)) or (n.value is not None and (not np.isfinite(n.value) or abs(n.value)>10000))):raise ValueError('numeric constant limit')
            if isinstance(n,ast.Attribute) and not (isinstance(n.value,ast.Name) and n.value.id=='np' and n.attr in FUNCTIONS):raise ValueError('attribute not allowed')
            if isinstance(n,ast.Call):
                if not isinstance(n.func,ast.Attribute) or n.func.attr not in FUNCTIONS or not isinstance(n.func.value,ast.Name) or n.func.value.id!='np':raise ValueError('only listed NumPy functions allowed')
                if any(k.arg not in ('axis','keepdims') for k in n.keywords):raise ValueError('unsupported call keyword')
    def __call__(self,env):
        def visit(n):
            if isinstance(n,ast.Constant):return n.value
            if isinstance(n,ast.Name):
                if n.id not in env:raise ValueError('unavailable name '+n.id)
                return env[n.id]
            if isinstance(n,(ast.List,ast.Tuple)):
                v=[visit(x) for x in n.elts];return bounded(v if isinstance(n,ast.List) else tuple(v))
            if isinstance(n,ast.Slice):
                parts=[visit(v) if v is not None else None for v in (n.lower,n.upper,n.step)]
                if any(v is not None and type(v)is not int for v in parts):raise ValueError('integer slice required')
                return slice(*parts)
            if isinstance(n,ast.Subscript):return bounded(visit(n.value)[visit(n.slice)])
            if isinstance(n,ast.UnaryOp):
                a=visit(n.operand);fn={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Invert:operator.invert,ast.Not:operator.not_}[type(n.op)];return bounded(fn(a))
            if isinstance(n,ast.BinOp):
                a,b=visit(n.left),visit(n.right)
                if not isinstance(a,(np.ndarray,np.generic,int,float,bool)) or not isinstance(b,(np.ndarray,np.generic,int,float,bool)):raise ValueError('arithmetic requires numeric arrays/scalars')
                shape=np.broadcast_shapes(np.shape(a),np.shape(b))
                if np.prod(shape)>LIMIT or len(shape)>2:raise ValueError('broadcast size limit')
                if isinstance(n.op,ast.Pow) and np.any(np.abs(b)>8):raise ValueError('power bound')
                with np.errstate(all='ignore'):return bounded(BINARY[type(n.op)](a,b))
            if isinstance(n,ast.Compare):
                if len(n.ops)!=1:raise ValueError('use explicit & for array comparisons')
                a,b=visit(n.left),visit(n.comparators[0]);shape=np.broadcast_shapes(np.shape(a),np.shape(b))
                if np.prod(shape)>LIMIT:raise ValueError('comparison broadcast size')
                return bounded(COMPARE[type(n.ops[0])](a,b))
            if isinstance(n,ast.Call):
                args=[visit(x) for x in n.args];kwargs={k.arg:visit(k.value) for k in n.keywords};name=n.func.attr
                if name in ('stack','concatenate') and sum(np.asarray(x).size for x in args[0])>LIMIT:raise ValueError('stack size limit')
                if name in ('where','minimum','maximum','clip'):
                    shape=np.broadcast_shapes(*(np.shape(v) for v in args))
                    if np.prod(shape)>LIMIT:raise ValueError('function broadcast size limit')
                # No out/dtype/callback arguments; all other args are numerical.
                allowed_counts={'abs':(1,),'square':(1,),'sqrt':(1,),'log1p':(1,),'exp':(1,),'tanh':(1,),'sin':(1,),'cos':(1,),'minimum':(2,),'maximum':(2,),'clip':(3,),'where':(3,),'mean':(1,),'std':(1,),'sum':(1,),'max':(1,),'min':(1,),'argmax':(1,),'concatenate':(1,),'stack':(1,)}
                if len(args) not in allowed_counts[name]:raise ValueError('function argument count')
                with np.errstate(all='ignore'):return bounded(FUNCTIONS[name](*args,**kwargs))
            raise ValueError('unsupported expression')
        return visit(self.tree)

