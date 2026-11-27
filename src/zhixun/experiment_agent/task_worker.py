"""Continue the host checkpoint in the official MLP format."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from baselines.model import Model
from .task_tools import _validate_proposal
from .worker import train as train_core


def train(case):
    case = Path(case)
    job = json.loads((case / "job.json").read_text())
    proposal = job["proposal"]
    parent = case / "parent.npz"
    result = case / "result"
    program = proposal["program"]
    model = Model.load(parent)
    _validate_proposal(proposal, model.meta["profile"])
    data_path = case / "train.npz"
    numerical = train_core(parent, data_path, program, job["context"], result,
                           steady_dropout=proposal.get("steady_channel_dropout", False))
    return dict(complete=True, numerical=numerical, core_steps_before=numerical["initial_steps"],
                core_steps_after=numerical["final_steps"], core_parent_matches_selected=True,
                core_parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                model_kind="core", checkpoint="checkpoint.npz", train_only=True,
                program_sha256=hashlib.sha256(program.encode()).hexdigest())


def main():
    case=Path(sys.argv[1]);job=json.loads((case/'job.json').read_text());allowed=sorted(os.sched_getaffinity(0))[:2] if hasattr(os,'sched_getaffinity') else None
    if allowed is not None:os.sched_setaffinity(0,set(allowed))
    seconds=int(job['cpu_seconds']);resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds+1));
    if sys.platform == 'linux':
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        budget = min([4 * 1024**3] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
    resource.setrlimit(resource.RLIMIT_FSIZE,(100000000,100000000));started=time.monotonic()
    try:report=train(case);status='completed';error=None
    except Exception as e:status='failed';error=dict(type=type(e).__name__,message=str(e)[:1000]);report=None
    usage=resource.getrusage(resource.RUSAGE_SELF)
    outcome=dict(status=status,error=error,training=report,cpu_seconds=usage.ru_utime+usage.ru_stime,wall_seconds=time.monotonic()-started,max_rss_kib=usage.ru_maxrss,cpu_affinity=allowed)
    (case/'execution.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n');print(json.dumps(outcome),flush=True)
    if status!='completed':sys.exit(1)


if __name__=='__main__':main()
