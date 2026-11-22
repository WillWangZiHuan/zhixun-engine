"""CPU candidate construction. No host protocol, model service, or adoption rules."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from baselines.data import read_dataset, json_value
from .checkpoint import load_checkpoint, kind, core
from .experiment_agent.program import default_program
from .experiment_agent.task_tools import _validate_proposal
from .experiment_agent.unified_protocol import NUMERICAL_FUNCTIONS
from .experiment_agent.runtime import supervise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def metrics(model, data):
    if model.meta['task'] == 'classification':
        from .steady_cli import evaluate
        return evaluate(model, data)
    error = model.predict(data['x']) - data['y']
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))))


def prepare(parent, train_path, validation_path, profile, task):
    inputs = dict(parent=Path(parent), train=Path(train_path), validation=Path(validation_path))
    for name, path in inputs.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular input file required: '+name)
    model = load_checkpoint(parent)
    if model.meta['profile'] != profile or model.meta['task'] != task:
        raise ValueError('Parent task/profile differs from frozen task contract')
    train = read_dataset(train_path, task, model.meta['classes'], model.meta['paths'])
    valid = read_dataset(validation_path, task, model.meta['classes'], model.meta['paths'])
    if set(train['ids']) & set(valid['ids']):
        raise ValueError('Training and validation IDs overlap')
    context = dict(suggested_learning_rate={'A':.01, 'B':.006, 'C':.0005}[profile],
                   current_learning_rate=model.meta['learning_rate'], parent_steps=model.meta['steps'],
                   parent_epochs=model.meta['epochs'], remaining_cpu_seconds=20,
                   device_groups_available=False)
    view = dict(profile=profile, seed=model.meta['seed'], parent_kind=kind(model), context=context,
                train_rows=len(train['y']), validation_rows=len(valid['y']),
                parent_training_metrics=metrics(model, train),
                parent_validation_metrics=metrics(model, valid),
                metric_scope='local diagnostics on authorized public data, not platform measurements',
                baseline_program=default_program(task, {'A':32, 'B':64, 'C':256}[profile]),
                numerical_functions=NUMERICAL_FUNCTIONS, device_groups=None,
                data_contract='id/inputs/answer only; meta excluded from learning and prediction')
    return model, train, valid, inputs, {k:digest(p) for k,p in inputs.items()}, view


def continue_candidate(root, prepared, proposal):
    model, train, valid, inputs, hashes, view = prepared
    _validate_proposal(proposal, model.meta['profile'])
    case = root/'work'
    case.mkdir()
    shutil.copyfile(inputs['parent'], case/'parent.npz')
    np.savez_compressed(case/'train.npz', x=train['x'], y=train['y'])
    write(case/'job.json', dict(proposal=proposal, parent_kind=kind(model),
                               context=view['context'], cpu_seconds=20))
    supervision = supervise(case, 'zhixun.experiment_agent.task_worker')
    if not (case/'execution.json').exists():
        raise RuntimeError('Worker produced no receipt; exit=%s timeout=%s; see work/stderr.log' %
                           (supervision['exit_code'], supervision['timed_out']))
    execution = json_value((case/'execution.json').read_text())
    if supervision['exit_code'] != 0 or execution['status'] != 'completed':
        raise RuntimeError('CPU continuation failed: '+json.dumps(execution.get('error'), ensure_ascii=False))
    source = case/'result/checkpoint.npz'
    candidate = load_checkpoint(source)
    before, after = core(model), core(candidate)
    if after.meta['steps'] <= before.meta['steps']:
        raise ValueError('No actual optimizer continuation')
    for name in ('mean', 'scale'):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    for name in ('profile', 'paths', 'classes', 'task', 'hidden', 'target_mean', 'target_scale', 'seed'):
        if after.meta[name] != before.meta[name]:
            raise ValueError('Inherited model contract changed: '+name)
    for name, path in inputs.items():
        if digest(path) != hashes[name]:
            raise ValueError('Input changed during training: '+name)
    target = root/'candidate'
    target.mkdir()
    shutil.copyfile(source, target/'checkpoint.npz')
    np.testing.assert_array_equal(candidate.predict(valid['x']),
                                  load_checkpoint(target/'checkpoint.npz').predict(valid['x']))
    report = dict(execution['training'], parent_file_sha256=hashes['parent'],
                  checkpoint_sha256=digest(target/'checkpoint.npz'), model_kind=kind(candidate),
                  source_input_hashes=hashes, local_validation_before=view['parent_validation_metrics'],
                  local_validation_after=metrics(candidate, valid), device_metadata_used=False,
                  device_split_verified=False, platform_score_claimed=False)
    write(target/'training.json', report)
    return target, report
