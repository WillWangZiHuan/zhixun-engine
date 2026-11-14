"""Facts from the evaluator, kept separate from the model's hypotheses."""
import ast,json
METRICS={"residual":"prediction - target", "bias":"mean(prediction - target)","positive_bias":"over-prediction of remaining lifetime","negative_bias":"under-prediction of remaining lifetime","units":"minutes", "rmse":"sqrt(mean((prediction-target)**2)) over rows, not average of group RMSE", "causality":"A rejected candidate contradicts its predicted metric gain under this experiment; it does not establish a unique cause."}
def receipt(event):
 before=event["incumbent_metrics_before"];after=event["candidate_metrics"]
 return dict(trial=event["trial"],parent_trial=event["parent_trial"],execution_status=event["execution_status"],accepted=event["accepted"],error=event["error"],actual_changed_functions=event.get("changed_functions","not recorded in this historical event"),actual_training_plan=event.get("plan"),actual_first_learning_rate=event.get("executed_rate"),parent_rmse=before["rmse"],candidate_rmse=after["rmse"] if after else None,delta_rmse_candidate_minus_parent=after["rmse"]-before["rmse"] if after else None,group_comparison={g:dict(parent=before["groups"][g],candidate=after["groups"][g] if after else None) for g in before["groups"]},proposed_hypothesis_not_established_fact=event["hypothesis"],predicted_pattern=event["expected_result"],falsifier=event["falsifier"])
def history_receipts(events):
 return [dict(trial=e["trial"],parent_trial=e["parent_trial"],status=e["execution_status"],accepted=e["accepted"],error=e["error"],parent_rmse=e["incumbent_metrics_before"]["rmse"],candidate_rmse=e["candidate_metrics"]["rmse"] if e["candidate_metrics"] else None,hypothesis_not_established_fact=e["hypothesis"],program_resource=f"experiment/{e['trial']}/program",report_resource=f"experiment/{e['trial']}/report") for e in events]
def facts(train_names,selection_names,events,last_program=None):
 return dict(metric_definitions=METRICS,training_group_index_to_bearing={str(i):name for i,name in enumerate(train_names)},selection_only_bearings=sorted(set(selection_names)-set(train_names)),selection_features_and_labels_used_for_gradients=False,parent_policy="Every new training run starts from the CURRENT RETAINED model. Rejected candidate weights are never continued. Compare new changes against current_program, not against the last rejected program.",latest_experiment=receipt(events[-1]) if events else None,latest_attempt_program=last_program,review_instruction="In reflection, cite the latest trial and its actual status/RMSE change. Separate observed outcome from your proposed explanation. Use the correct bias sign and training group names. Explain why the next program differs. A compile failure supplies no predictive-effect evidence.")
def program_key(source,parent,context):
 tree=ast.parse(source)
 # Include the entire context if it is referenced dynamically or aliased.
 loads=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=="context" and isinstance(n.ctx,ast.Load)]
 literal=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=="context" and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)]
 used={n.slice.value:context[n.slice.value] for n in literal} if len(loads)==len(literal) else context
 return parent,ast.dump(tree,include_attributes=False),json.dumps(used,sort_keys=True)

def factual_projection(request):
    """Retain measured history; archive model prose separately from current facts."""
    import copy
    result=copy.deepcopy(request)
    for event in result["history"]:
        event.pop("hypothesis_not_established_fact",None)
        event["original_hypothesis_resource"]=f"experiment/{event['trial']}/report"
    latest=result["experiment_facts"]["latest_experiment"]
    if latest:
        for key in ["proposed_hypothesis_not_established_fact","predicted_pattern","falsifier"]:latest.pop(key,None)
        latest["original_hypothesis_resource"]=f"experiment/{latest['trial']}/report"
    program=result["experiment_facts"].get("latest_attempt_program")
    if program:
        try:result["experiment_facts"]["latest_attempt_program"]=ast.unparse(ast.parse(program))
        except SyntaxError:result["experiment_facts"]["latest_attempt_program_syntax_valid"]=False
    return result

def changed_functions(before,after):
    def functions(source):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    old,new=functions(before),functions(after)
    return sorted(name for name in old.keys()|new.keys() if old.get(name)!=new.get(name))
