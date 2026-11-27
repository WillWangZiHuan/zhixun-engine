import json,re
NUMERICAL_FUNCTIONS='array asarray ones zeros ones_like zeros_like full_like arange abs maximum minimum clip where mean sum std median quantile percentile exp log log1p max sqrt square power tanh sin cos sign bincount unique argsort sort concatenate stack repeat tile reshape isfinite all any; np.pi; len float int min max abs sum; array.shape; bare loss operators: weighted_cross_entropy(pred,target,weights,gamma=0), weighted_mse(pred,target,weights), weighted_huber(pred,target,weights,delta)'

def parse(text):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate JSON field')
            result[k]=v
        return result
    payload=text.strip()
    def reject_constant(value):raise ValueError('Nonfinite JSON constant')
    d=json.loads(payload,object_pairs_hook=unique,parse_constant=reject_constant)
    fields={'experiment':{'tool','reflection','proposal'},'read':{'tool','resource'},'analyze':{'tool','dataset','view'},'finish':{'tool','conclusion','supported','not_supported'}}
    if not isinstance(d,dict) or d.get('tool') not in fields or set(d)!=fields[d['tool']]:raise ValueError('Tool schema mismatch')
    if d['tool']=='experiment' and (not isinstance(d['proposal'],dict) or not isinstance(d['reflection'],str)):raise ValueError('Invalid experiment object')
    if d['tool']=='read' and (not isinstance(d['resource'],str) or d['resource'] not in ('training_code','current_program','baseline_program','history') and not re.fullmatch(r'trial/(?:[1-9]|10)/(?:proposal|report)',d['resource'])):raise ValueError('Unavailable resource')
    if d['tool']=='analyze' and (d['dataset'] not in ('train','selection') or d['view'] not in ('group_errors','target_bins','feature_error_correlation','readout_summary')):raise ValueError('Unavailable analysis')
    if d['tool']=='finish' and (not isinstance(d['conclusion'],str) or any(not isinstance(d[k],list) or any(not isinstance(x,str) for x in d[k]) for k in ('supported','not_supported'))):raise ValueError('Invalid final review')
    return d
