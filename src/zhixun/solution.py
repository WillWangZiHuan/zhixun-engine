"""Independent single-NPZ prediction and export of selected task-tool models.

Training is exposed by zhixun.experiment_agent.unified_controller.
"""
from pathlib import Path
import argparse
import json
import os
import sys

CHECKPOINTS = {task: task+'/checkpoint.npz' for task in ('A','B','C')}


def configure_runtime(runtime):
    runtime = Path(runtime).resolve()
    if not (runtime/'baselines/model.py').is_file():
        raise ValueError('Supply the prepared official training runtime')
    os.environ.update(BASELINE_CPU_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(runtime))


def load_model(profile, checkpoint):
    from .checkpoint import load_checkpoint, kind
    if profile not in CHECKPOINTS:raise ValueError('Expected profile A, B or C')
    model=load_checkpoint(checkpoint)
    expected_task='classification' if profile=='A' else 'regression'
    if model.meta.get('profile')!=profile or model.meta['task']!=expected_task:
        raise ValueError('Checkpoint task/profile mismatch')
    if profile=='B' and kind(model) not in ('core','bearing_head'):raise ValueError('B requires a core or bearing head')
    return model


def predict_stream(checkpoint, profile, source, destination):
    from baselines.data import json_value
    from baselines.predict import predict
    model=load_model(profile,checkpoint)
    for line in source:
        if not line.strip():
            continue
        request = json_value(line)
        if not isinstance(request, dict) or set(request)-{'task', 'inputs'} or 'inputs' not in request:
            raise ValueError('Prediction accepts only task and inputs')
        if request.get('task', model.meta['task']) != model.meta['task']:
            raise ValueError('Prediction task mismatch')
        destination.write(json.dumps(predict(model, request['inputs']), allow_nan=False)+'\n')
        destination.flush()



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export')
    export.add_argument('--runs',type=Path,required=True)
    export.add_argument('--output',type=Path,required=True)
    prediction=sub.add_parser('predict')
    prediction.add_argument('--checkpoint',type=Path,required=True)
    prediction.add_argument('--task',choices=list(CHECKPOINTS),required=True)
    for command in (export,prediction):command.add_argument('--training-runtime',type=Path,required=True)
    args=parser.parse_args();configure_runtime(args.training_runtime)
    if args.command=='export':
        from .experiment_agent.task_tools import export_models
        print(json.dumps(export_models(args.runs,args.output)))
    else:predict_stream(args.checkpoint,args.task,sys.stdin,sys.stdout)


if __name__=='__main__':main()
