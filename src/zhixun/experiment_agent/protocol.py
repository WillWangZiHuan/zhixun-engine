"""Experimental tool protocol: the model edits numerical training source."""
import json,re
from .program import DEFAULT
PROMPT='''You are the experimental scientist for a CPU bearing remaining-useful-life task (minutes). Start from the supplied pretrained MLP and improve overall selection RMSE within ten experiment attempts and the CPU budget. Diagnose per-bearing failures; hypotheses are not established causes. You may inspect evidence, write numerical training code, run one experiment, inspect the result, and revise your next hypothesis. A fixed external evaluator accepts only lower overall selection RMSE. Audit devices are never accessible. All results are development evidence, because the original checkpoint already saw the official training pack.
The executor preserves the network, feature normalization, Adam moments and cumulative steps, data split, prediction and evaluation. You can freely rewrite the five numerical functions, not merely choose from an algorithm list. They control epochs/batch size, per-step learning rate, sample weights, training order, and loss plus derivative w.r.t. normalized predictions. No imports, I/O, private attributes, loops/comprehensions, mutation of inputs, or dynamic calls. NumPy vector expressions, assignments, if/return, tuples/dicts and the provided numerical functions are supported. The code is checked before isolated execution; a wrong gradient or invalid program is a failed experiment and counts toward ten attempts.
Read the supplied training contract, experiment_facts, and existing code. experiment_facts contains evaluator-produced facts; history hypotheses are unverified claims, not facts. Bias is prediction minus target: negative means UNDER-prediction, positive means OVER-prediction. Only the explicitly mapped training bearings receive gradients; selection-only bearings never do. Every mode receives the latest experiment receipt, including rejected/invalid attempts. A mode without accumulated history may use that immediate tool feedback but cannot infer older outcomes. In reflection, explicitly assess the latest receipt before designing the next experiment. Use actual_changed_functions and actual_training_plan to identify what really ran, rather than attributing an outcome to an old hypothesis. When both epochs and learning rate change, do not call that a single-variable test. Read the supplied training contract and existing code. Historical summaries contain every attempt; full past programs and reports can be read using resource experiment/N/program or experiment/N/report when history access is enabled. The baseline already uses inverse bearing-count weights; do not assume it is unweighted. Loss uses normalized RUL, whereas displayed errors are minutes. Avoid changing multiple mechanisms in one experiment. Describe a falsifiable expected pattern, not just 'improve score'. Check whether group errors support the explanation. Reduction in overall RMSE alone does not prove a proposed cause. Do not repeat failed proposals without a changed reason. You may use up to twenty-one model calls including reads, analysis and the final conclusion; code execution is at most ten experiments. Each experiment permits at most ONE inspection (read or analyze), followed immediately by an experiment. You may skip inspection and directly experiment. You already have current per-bearing metrics, train distribution, training contract and code. When required_next is experiment, you MUST output an experiment, never another read/analyze. When required_next is finish, output your final review. Query only when needed, because every call has cost.
Return exactly one JSON tool object, without markdown. A native <tool_call> envelope is also a transport for the same object; it does not change the allowed operation. The JSON object must use one of these exact schemas:
{"tool":"read","resource":"baseline_program|current_program|model_training_code|experiment_history"}
{"tool":"analyze","dataset":"train|selection","view":"residuals_by_group|residuals_by_target_bin|feature_error_correlation"}
{"tool":"experiment","hypothesis":"...","expected_result":"...","falsifier":"...","reflection":"Interpret available prior results without overstating causality","program":"complete five-function Python source"}
After ten attempts, return {"tool":"finish","conclusion":"...","supported":["..."],"not_supported":["..."]}. Do not finish early or propose more experiments afterward. Audit results are not available. Returned training code must use exactly the given function signatures. A model with no history access must work from current state; never invent past outcomes.'''
CONTRACT=dict(signatures={'plan':'plan(context) -> {epochs:int in 1..5, batch_size:int in 16..4096}',
 'learning_rate':'learning_rate(step, context) -> finite rate in [1e-6,0.1]; step is experiment-local update count',
 'sample_weights':'sample_weights(x,y,groups,context) -> nonnegative finite N-vector; executor normalizes mean to 1',
 'batch_order':'batch_order(order,groups,y,epoch) -> permutation of every training row; order is official deterministic shuffle',
 'loss':'loss(pred,target,weights) -> scalar mean loss and exact N-vector derivative d(loss)/d(pred)'},
 arrays='x is official standardized/clipped input; y and pred use retained target mean/scale; groups are contiguous training-bearing indices. Inputs are readonly.',
 context_fields=['suggested_learning_rate','current_learning_rate','parent_epochs','parent_steps','remaining_experiments','remaining_cpu_seconds'],
 numerical_functions='array asarray ones zeros ones_like zeros_like full_like arange abs maximum minimum clip where mean sum std median quantile percentile exp log log1p sqrt square power tanh sin cos sign bincount unique argsort sort concatenate stack repeat tile reshape isfinite all any; np.pi; len float int min max abs sum; array.shape',
 limitations='No model architecture edits or optimizer replacement in v1. Fixed updater computes tanh MLP backprop and clipped Adam. Max1500 updates per attempt. Selection labels are only used by evaluator, never by this program.')
def parse(text):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON field')
   d[k]=v
  return d
 if '<tool_call>' in text or '</tool_call>' in text:
  if text.count('<tool_call>')!=1 or text.count('</tool_call>')!=1:raise ValueError('exactly one tool object required')
  start=text.index('<tool_call>')+len('<tool_call>');end=text.index('</tool_call>')
  if end<start:raise ValueError('invalid native tool block order')
  payload=text[start:end]
  outside=text[:start-len('<tool_call>')]+text[end+len('</tool_call>'):]
  if '{' in outside or '}' in outside:raise ValueError('extra JSON outside tool envelope')
 else:payload=text
 payload=payload.strip()
 if payload.startswith('```json\n') and payload.endswith('\n```'):
  payload=payload[len('```json\n'):-len('\n```')]
 d=json.loads(payload,object_pairs_hook=unique)
 fields={'read':{'tool','resource'},'analyze':{'tool','dataset','view'},'experiment':{'tool','hypothesis','expected_result','falsifier','reflection','program'},'finish':{'tool','conclusion','supported','not_supported'}}
 if not isinstance(d,dict) or d.get('tool') not in fields or set(d)!=fields[d['tool']]:raise ValueError('tool schema mismatch')
 if d['tool']=='read' and d['resource'] not in ('baseline_program','current_program','model_training_code','experiment_history') and not re.fullmatch(r'experiment/(?:[1-9]|10)/(?:program|report)',d['resource']):raise ValueError('resource unavailable')
 if d['tool']=='analyze' and (d['dataset'] not in ('train','selection') or d['view'] not in ('residuals_by_group','residuals_by_target_bin','feature_error_correlation')):raise ValueError('analysis unavailable')
 if d['tool']=='experiment' and any(not isinstance(d[k],str) or not d[k].strip() for k in fields['experiment']-{'tool'}):raise ValueError('experiment strings required')
 if d['tool']=='finish' and (not isinstance(d['conclusion'],str) or any(not isinstance(d[k],list) or any(not isinstance(x,str) for x in d[k]) for k in ('supported','not_supported'))):raise ValueError('invalid conclusion')
 return d
