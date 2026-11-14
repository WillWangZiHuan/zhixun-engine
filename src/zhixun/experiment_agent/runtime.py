"""Native CPU execution and local immutable scoring; no model-service key forwarded."""
from pathlib import Path
import inspect,json,os,resource,signal,subprocess,shutil,sys,time
import numpy as np
from baselines.model import Model
from .program import validate


def supervise(case, worker_module):
    """Run the trusted updater on this Linux host without service credentials."""
    if sys.platform not in ('linux','darwin'):raise RuntimeError('Native Linux or macOS sandbox required')
    runtime = Path(inspect.getfile(Model)).resolve().parents[1]
    env = dict(PATH=os.pathsep.join([str(Path(sys.executable).parent), '/usr/bin', '/bin']),
               LANG='C.UTF-8', PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[2]), str(runtime)]),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               BASELINE_CPU_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    timed_out = False
    with (case/'stdout.log').open('w') as out, (case/'stderr.log').open('w') as err:
        command=[sys.executable,'-B','-m',worker_module,str(case)]
        if sys.platform=='darwin':
            # Sandbox restricts generated numerical code to this case, with no network.
            policy='(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath '+json.dumps(str(case.resolve()),ensure_ascii=False)+'))'
            command=['/usr/bin/sandbox-exec','-p',policy,*command]
        process = subprocess.Popen(command,
                                   env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(exit_code=code, timed_out=timed_out,
                  cpu_seconds=(usage_after.ru_utime+usage_after.ru_stime)-(usage_before.ru_utime+usage_before.ru_stime),
                  wall_seconds=time.monotonic()-started, credentials_forwarded=False,
                  execution='macOS sandbox child' if sys.platform=='darwin' else 'native Linux child process', network_isolated=sys.platform=='darwin', aggregate_memory_enforced=False)
    (case/'supervisor.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

def evaluate(model,data,names):
 pred=model.predict(data['x']);error=pred-data['y']
 if names is None:return dict(rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),device_groups_available=False)
 groups={str(g):dict(n=int(sum(names==g)),rmse=float(np.sqrt(np.mean(error[names==g]**2))),mae=float(np.mean(abs(error[names==g]))),bias=float(np.mean(error[names==g]))) for g in sorted(set(names))}
 return dict(rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),groups=groups,worst_group_rmse=max(v['rmse'] for v in groups.values()))

def execute(case,parent,data,source,context,cpu_remaining):
 case=Path(case)
 if case.exists():
  if not (case/'execution.json').exists():raise RuntimeError('Existing experiment is incomplete; inspect before resume')
  if (case/'program.py').read_text()!=source:raise ValueError('Existing experiment has a different program')
  saved=json.loads((case/'job.json').read_text())['context'];current=dict(context)
  saved.pop('remaining_cpu_seconds',None);current.pop('remaining_cpu_seconds',None)
  if saved!=current:raise ValueError('Existing experiment has a different training state')
  for filename,expected in [('parent.npz',parent),('train.npz',data)]:
   if (case/filename).exists():
    with np.load(case/filename,allow_pickle=False) as old,np.load(expected,allow_pickle=False) as new:
     if set(old.files)!=set(new.files):raise ValueError('Existing numerical inputs differ')
     for key in old.files:np.testing.assert_array_equal(old[key],new[key])
  return json.loads((case/'execution.json').read_text())
 case.mkdir(exist_ok=False);(case/'program.py').write_text(source);context=json.loads(json.dumps(context));job=dict(context=context,cpu_seconds=min(20,int(cpu_remaining)));(case/'job.json').write_text(json.dumps(job,indent=2)+'\n')
 preflight_cpu=time.process_time();preflight_wall=time.monotonic()
 try:validate(source)
 except (SyntaxError,ValueError) as e:
  result=dict(status='preflight_failed',error={'type':type(e).__name__,'message':str(e)[:1000]},cpu_seconds=time.process_time()-preflight_cpu,wall_seconds=time.monotonic()-preflight_wall,max_rss_kib=0);(case/'execution.json').write_text(json.dumps(result,indent=2)+'\n');return result
 shutil.copyfile(parent,case/'parent.npz');shutil.copyfile(data,case/'train.npz')
 supervisor=supervise(case, 'zhixun.experiment_agent.worker')
 execution=json.loads((case/'execution.json').read_text()) if (case/'execution.json').exists() else dict(status='resource_terminated',error=dict(type='ResourceTermination',message='worker terminated before result'))
 result={**execution,**supervisor}
 if result['status']=='completed':
  report=json.loads((case/'result/training.json').read_text());old=Model.load(parent);new=Model.load(case/'result/checkpoint.npz')
  for k in ('mean','scale'):np.testing.assert_array_equal(getattr(old,k),getattr(new,k))
  for k in ('paths','classes','task','hidden','target_mean','target_scale','seed'):
   if old.meta[k]!=new.meta[k]:raise ValueError('immutable checkpoint metadata changed')
  if report['initial_steps']!=old.meta['steps'] or report['final_steps']!=new.meta['steps'] or new.meta['steps']<=old.meta['steps']:raise ValueError('Adam step continuity broken')
  result['training']=report
 (case/'execution.json').write_text(json.dumps(result,indent=2)+'\n');return result
