"""Export traceable industrial training trials and loop decisions into SwanLab."""
import argparse
import hashlib
import json
from pathlib import Path


def export_run(source, destination, *, mode='offline', project='zhixun-engine'):
    import swanlab
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if mode not in ('offline', 'online'):
        raise ValueError('Choose explicit offline or online recording')
    destination.mkdir(parents=True, exist_ok=False)
    events_file = source / 'events.jsonl'
    protocol = json.loads((source / 'protocol.json').read_text())
    if events_file.exists():
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
    elif protocol.get('protocol') == 'cmapss-static-group-v1':
        events = []  # Static baseline has training trials, not online loop events.
    else:
        raise FileNotFoundError('Unknown run format or missing closed-loop event log')
    manifest = dict(source=str(source), source_event_sha256=hashlib.sha256(events_file.read_bytes()).hexdigest() if events_file.exists() else None,
                    mode=mode, cloud_verified=False, official_log_standard_verified=False,
                    recording='retrospective export of original timestamped local records',
                    events=len(events), trials=0, experiments=[])
    (destination / 'export_manifest.json').write_text(json.dumps(manifest, indent=2))
    swanlab.init(project=project, experiment_name=source.name, mode=mode, public=True,
                 on_online_error='raise', logdir=str(destination / 'swanlog'),
                 config=protocol, settings=swanlab.Settings(interactive=False),
                 description='Local development benchmark; not official score. Full evidence is keyed by original event sequence.')
    step = 0
    try:
        for experiment in sorted((source / 'experiments').iterdir()):
            if not experiment.is_dir():
                continue
            trial_file = experiment / 'flaml_trials.jsonl'
            trials = [json.loads(s) for s in trial_file.read_text().splitlines()] if trial_file.exists() else []
            count = 0
            for trial in trials:
                if 'validation_loss' not in trial:
                    continue
                step += 1; count += 1
                metrics = {f'train/{k}': trial[k] for k in
                           ('validation_loss', 'trial_time', 'wall_clock_time', 'sample_size') if k in trial}
                metrics['train/evidence'] = swanlab.Text(json.dumps(dict(experiment=experiment.name, trial=trial)))
                swanlab.log(metrics, step=step)
            manifest['trials'] += count
            manifest['experiments'].append(dict(name=experiment.name, recorded_trials=count,
                                                result_available=(experiment / 'result.json').exists()))
        for event in events:
            step += 1
            metrics = {f'loop/{k}': v for k, v in event.items() if type(v) in (int, float)}
            metrics['loop/evidence'] = swanlab.Text(json.dumps(event, ensure_ascii=False))
            swanlab.log(metrics, step=step)
    finally:
        swanlab.finish()
    manifest['export_completed'] = True
    manifest['steps'] = step
    (destination / 'export_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Record local run evidence in SwanLab')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['offline', 'online'], default='offline')
    parser.add_argument('--project', default='zhixun-engine')
    args = parser.parse_args()
    print(json.dumps(export_run(args.source, args.output, mode=args.mode, project=args.project), indent=2))


if __name__ == '__main__':
    main()
