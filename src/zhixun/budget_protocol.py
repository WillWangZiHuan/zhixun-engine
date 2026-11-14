"""Bounded training-budget choices using only pre-action observations."""
import json

PROMPT = """Choose the next training budget for an existing CPU remaining-useful-life regression MLP.
Copy exactly one listed action_id. Actions are short (10 epochs), long (40 epochs), or stop (keep the selected model and end the trajectory).
Aim to reduce selection-set RMSE without spending computation on ineffective updates. Group RMSE diagnoses tradeoffs. A fixed gate accepts a candidate only if its overall selection RMSE is lower; other devices are kept exclusively for a later audit and are never disclosed here.
All actions resume the same selected weights, Adam state, and normalization. Group-balanced loss and batch size are fixed. The learning rate shown is predetermined: it halves after a rejected candidate, regardless of your choice. You choose only epochs or stop.
There are at most 10 decisions and at most 400 training epochs including rejected candidates. Stopping can save unused compute. These are equal resource ceilings, not necessarily equal actual consumption.
Use only this trajectory's completed observations. Do not assume unknown device behavior or invent causal explanations. Training loss is on standardized targets, RMSE uses cycles.
Return exactly one JSON object with the single field action_id. No explanation, no extra fields, no markdown. The system records the complete pre-action observations automatically."""


def request(seed, initial, history, lr, train_groups):
    if len(history)>9 or [e['round'] for e in history]!=list(range(1,len(history)+1)):
        raise ValueError('invalid causal history')
    used=sum(e['epochs'] for e in history)
    if used>400:raise ValueError('budget exceeded')
    candidates=[dict(action_id='stop',epochs=0)]
    candidates += [dict(action_id=name,epochs=n) for name,n in [('short',10),('long',40)] if used+n<=400]
    value=dict(profile='C',seed=seed,next_round=len(history)+1,remaining_rounds=10-len(history),remaining_training_epochs=400-used,learning_rate=lr,train_group_rows=train_groups,candidates=candidates,initial=initial,history=history)
    json.dumps(value,allow_nan=False)
    return value


def rule(policy,state):
    h=state['history']
    if policy=='fixed10':name='short'
    elif policy=='fixed40':name='long'
    elif policy=='adaptive':
        failures=0
        for e in reversed(h):
            if e['accepted']:break
            failures+=1
        name='stop' if failures>=3 else 'short' if failures else 'long'
    else:raise ValueError('unknown rule')
    if name not in {a['action_id'] for a in state['candidates']}:raise ValueError('action exceeds budget')
    return name


def parse_decision(text,state):
    def unique(pairs):
        value={}
        for k,v in pairs:
            if k in value:raise ValueError('duplicate field')
            value[k]=v
        return value
    value=json.loads(text,object_pairs_hook=unique)
    if not isinstance(value,dict) or set(value)!={'action_id'} or type(value['action_id']) is not str:
        raise ValueError('exact action_id object required')
    matches=[a for a in state['candidates'] if a['action_id']==value['action_id']]
    if len(matches)!=1:raise ValueError('unavailable action')
    return matches[0].copy()
