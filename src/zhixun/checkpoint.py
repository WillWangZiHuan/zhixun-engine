"""Complete typed NPZ model state. No directories, pickle or external state files."""
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from baselines.model import Model, PARAMETERS
from .bearing_head import BearingHead

FORMAT = 'zhixun.native-checkpoint.v1'
MAX_BYTES = 64 * 1024 * 1024
CORE_KEYS = set(PARAMETERS) | {'adam_first_'+k for k in PARAMETERS} | {'adam_second_'+k for k in PARAMETERS} | {'mean','scale','metadata'}


def kind(model):
    from .steady_pair import SteadyPair
    from .readout import AReadout
    from .boosted import ResidualBoost
    types = {BearingHead:'bearing_head', Model:'core', SteadyPair:'steady_pair', AReadout:'a_readout', ResidualBoost:'residual_boost'}
    if type(model) not in types:raise ValueError('Unsupported checkpoint model type')
    return types[type(model)]


def core(model):return model if kind(model)=='core' else model.base


def _json(value):
    def unique(pairs):
        result={}
        for key,item in pairs:
            if key in result:raise ValueError('Duplicate metadata key')
            result[key]=item
        return result
    def invalid(value):raise ValueError('Nonfinite metadata value')
    return json.loads(value,object_pairs_hook=unique,parse_constant=invalid)


def _text(value):
    if value.shape!=() or value.dtype.kind!='U':raise ValueError('Expected scalar Unicode state')
    return str(value)


def _array(arrays,name,shape,dtype):
    value=arrays[name]
    if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all():
        raise ValueError('Invalid checkpoint array: '+name)
    return value.copy()


def _decode(arrays):
    from .steady_pair import SteadyPair
    from .readout import AReadout
    from .boosted import ResidualBoost
    import lightgbm as lgb
    if 'native_metadata' not in arrays:raise ValueError('Native checkpoint metadata missing')
    info=_json(_text(arrays['native_metadata']))
    if not isinstance(info,dict) or info.get('format')!=FORMAT:raise ValueError('Unsupported native checkpoint format')
    model_kind=info.get('kind')
    schemas={'bearing_head':{'format','kind'},'core':{'format','kind'},'steady_pair':{'format','kind','head'},
             'a_readout':{'format','kind','readout'},'residual_boost':{'format','kind','forest'}}
    if model_kind not in schemas:raise ValueError('Unsupported native checkpoint kind')
    has_head=model_kind=='steady_pair'
    if model_kind=='a_readout':
        rd=info['readout']
        if not isinstance(rd,dict) or set(rd)!={'upstream_kind','expression'} or rd['upstream_kind'] not in ('core','steady_pair') or not isinstance(rd['expression'],str):
            raise ValueError('Invalid readout state')
        has_head=rd['upstream_kind']=='steady_pair'
    expected_info=schemas[model_kind] | ({'head'} if has_head else set())
    if set(info)!=expected_info:raise ValueError('Unexpected native metadata fields')
    expected=CORE_KEYS|{'native_metadata'}
    if has_head:
        head=info['head']
        if not isinstance(head,dict) or set(head)!={'fitted'} or type(head['fitted']) is not bool:raise ValueError('Invalid head metadata')
        expected|={'head_pair','head_indices'}
        if head['fitted']:expected|={'head_mean','head_scale','head_coef','head_intercept'}
    if model_kind=='residual_boost':expected|={'forest_text'}
    if model_kind == 'bearing_head':
        expected |= {'bearing_' + name for name in ('amplitude','location','spread','centers','coefficients','bias')}
    if set(arrays)!=expected:raise ValueError('Unexpected checkpoint array keys')
    base=Model.__new__(Model)
    base.meta=_json(_text(arrays['metadata']))
    if not isinstance(base.meta,dict):raise ValueError('Invalid core metadata')
    base.params={k:arrays[k].copy() for k in PARAMETERS}
    base.first={k:arrays['adam_first_'+k].copy() for k in PARAMETERS}
    base.second={k:arrays['adam_second_'+k].copy() for k in PARAMETERS}
    base.mean=arrays['mean'].copy();base.scale=arrays['scale'].copy();base.validate()
    profile=base.meta.get('profile')
    if profile not in ('A','B','C') or base.meta['task']!=('classification' if profile=='A' else 'regression'):
        raise ValueError('Invalid core task/profile')
    if profile=='B' and base.meta.get('target_unit')!='minutes':raise ValueError('B requires minutes')
    model=base
    if model_kind == 'bearing_head':
        if profile != 'B':
            raise ValueError('Vibration head requires B')
        model = BearingHead(base)
        n = len(base.meta['paths'])
        for name, shape in [('amplitude',(n,)),('location',(n,)),('spread',(n,)),('centers',(3,n)),('coefficients',(3,n)),('bias',(3,))]:
            setattr(model, name, _array(arrays, 'bearing_' + name, shape, 'float64'))
        if np.any(model.amplitude <= 0) or np.any(model.spread <= 0):
            raise ValueError('Invalid vibration feature scale')
    if has_head:
        if profile!='A':raise ValueError('Head requires A')
        model=SteadyPair(base)
        pair=_array(arrays,'head_pair',(2,),'int64')
        indices=_array(arrays,'head_indices',(16,),'int64')
        if not np.array_equal(pair,model.pair) or not np.array_equal(indices,model.indices):raise ValueError('Unsupported head routing')
        if head['fitted']:
            for name,shape in [('mean',(16,)),('scale',(16,)),('coef',(2,16)),('intercept',(2,))]:
                setattr(model,name,_array(arrays,'head_'+name,shape,'float64'))
            if np.any(model.scale<=0):raise ValueError('Invalid head scale')
    if model_kind=='a_readout':model=AReadout(model,rd['upstream_kind'],rd['expression'])
    if model_kind=='residual_boost':
        if profile!='C':raise ValueError('Residual forest requires C')
        cfg=info['forest']
        if not isinstance(cfg,dict) or set(cfg)!={'leaves','min_leaf','rate','iterations','library_version'}:
            raise ValueError('Invalid forest metadata')
        if type(cfg['iterations']) is not int or cfg['iterations']<0 or not isinstance(cfg['library_version'],str):raise ValueError('Invalid forest count/version')
        if type(cfg['leaves']) is not int or cfg['leaves']<2 or type(cfg['min_leaf']) is not int or cfg['min_leaf']<1:
            raise ValueError('Invalid forest leaf configuration')
        if type(cfg['rate']) not in (int,float) or not np.isfinite(cfg['rate']) or not 0<cfg['rate']<=1:raise ValueError('Invalid forest rate')
        model=ResidualBoost(base,cfg['leaves'],cfg['min_leaf'],cfg['rate'])
        text=_text(arrays['forest_text'])
        if cfg['iterations']:
            if not text.startswith('tree\n') or '\x00' in text:raise ValueError('Invalid LightGBM model text')
            model.booster=lgb.Booster(model_str=text)
            if model.booster.num_feature()!=len(base.meta['paths']) or model.booster.num_model_per_iteration()!=1:
                raise ValueError('Forest feature/output shape differs')
            if model.booster.num_trees()!=cfg['iterations']:raise ValueError('Forest tree count differs')
        elif text:raise ValueError('Unexpected forest text for empty forest')
        if model.iterations!=cfg['iterations']:raise ValueError('Forest iteration count differs')
    return model


def _encode(model):
    model_kind=kind(model);base=core(model);base.validate()
    arrays={**base.params,**{'adam_first_'+k:v for k,v in base.first.items()},
            **{'adam_second_'+k:v for k,v in base.second.items()},'mean':base.mean,'scale':base.scale,
            'metadata':np.array(json.dumps(base.meta,sort_keys=True,separators=(',',':'),allow_nan=False))}
    info=dict(format=FORMAT,kind=model_kind)
    upstream=model.upstream if model_kind=='a_readout' else model
    if model_kind=='a_readout':info['readout']=dict(upstream_kind=model.upstream_kind,expression=model.expression)
    if kind(upstream)=='steady_pair':
        fitted=upstream.coef is not None
        if not fitted and any(getattr(upstream,n) is not None for n in ('mean','scale','intercept')):raise ValueError('Partial unfitted head state')
        info['head']=dict(fitted=fitted)
        arrays.update(head_pair=upstream.pair,head_indices=upstream.indices)
        if fitted:arrays.update({'head_'+n:getattr(upstream,n) for n in ('mean','scale','coef','intercept')})
    if model_kind=='residual_boost':
        import lightgbm as lgb
        info['forest']=dict(leaves=model.leaves,min_leaf=model.min_leaf,rate=model.rate,iterations=model.iterations,library_version=lgb.__version__)
        arrays['forest_text']=np.array('' if model.booster is None else model.booster.model_to_string())
    if model_kind == 'bearing_head':
        for name in ('amplitude','location','spread','centers','coefficients','bias'):
            arrays['bearing_' + name] = getattr(model, name)
    arrays['native_metadata']=np.array(json.dumps(info,sort_keys=True,separators=(',',':'),allow_nan=False))
    return arrays


def save_checkpoint(model,path):
    path=Path(path)
    if path.suffix!='.npz':raise ValueError('Checkpoint output must have .npz suffix')
    if path.exists() or path.is_symlink():raise FileExistsError(path)
    arrays=_encode(model)
    if any(not isinstance(a,np.ndarray) or a.dtype.hasobject for a in arrays.values()):raise ValueError('Object arrays forbidden')
    if sum(a.nbytes for a in arrays.values())>MAX_BYTES:raise ValueError('Checkpoint exceeds 64MiB')
    _decode(arrays)  # Validate the complete state before any file is written.
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,array in sorted(arrays.items()):
            data=io.BytesIO();np.lib.format.write_array(data,array,allow_pickle=False)
            member=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));member.external_attr=0o600<<16
            archive.writestr(member,data.getvalue())
    if buffer.tell()>MAX_BYTES:raise ValueError('Checkpoint exceeds 64MiB')
    with path.open('xb') as f:f.write(buffer.getbuffer())


def load_checkpoint(path,expected_kind=None):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_BYTES:raise ValueError('Checkpoint must be one regular NPZ <=64MiB')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        if len(names)!=len(set(names)) or len(names)>24 or sum(e.file_size for e in entries)>MAX_BYTES:
            raise ValueError('Invalid checkpoint archive bounds')
        if any('/' in n or '\\' in n or not n.endswith('.npy') for n in names):raise ValueError('Only named NPY arrays allowed')
        if any(e.flag_bits&1 or e.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for e in entries):raise ValueError('Unsupported archive entry')
        # Check every NPY header before NumPy can allocate arrays, including official inputs.
        declared_total=0
        for entry in entries:
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):reader=np.lib.format.read_array_header_1_0
                elif version==(2,0):reader=np.lib.format.read_array_header_2_0
                else:raise ValueError('Unsupported NPY version')
                shape,_,dtype=reader(stream,max_header_size=10000)
                if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in 'iufU':
                    raise ValueError('Unsupported checkpoint dtype')
                if len(shape)>2 or any(type(n) is not int or n<0 for n in shape):raise ValueError('Invalid checkpoint shape')
                declared=math.prod(shape)*dtype.itemsize
                declared_total+=declared
                if declared_total>MAX_BYTES or declared!=entry.file_size-stream.tell():raise ValueError('NPY payload size mismatch or limit')
        official=set(names)=={k+'.npy' for k in CORE_KEYS}
        if not official and 'native_metadata.npy' not in names:raise ValueError('Unsupported checkpoint schema')
        with np.load(path,allow_pickle=False) as values:
            arrays={name:values[name].copy() for name in values.files}
        # Explicit official core input schema, validated by the same core Model.validate.
        if official:arrays['native_metadata']=np.array(json.dumps(dict(format=FORMAT,kind='core')))
        model=_decode(arrays)
    if expected_kind is not None and kind(model)!=expected_kind:raise ValueError('Checkpoint kind does not match declared kind')
    return model
