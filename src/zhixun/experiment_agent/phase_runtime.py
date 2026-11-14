"""One attempt with separate startup and generation deadlines. No retry path."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


@dataclass
class AttemptBudget:
    limit: int
    used: int = 0

    def check(self):
        if not 0 <= self.used < self.limit:
            raise RuntimeError('Attempt budget exhausted')


def run_attempt(command, directory, budget, *, task_deadline, initialization_seconds=60, sampling_seconds=60, env=None):
    """Trusted worker waits for one 'sample' command after reporting readiness.

    A startup failure still consumes an attempt slot. Once sampling is authorized,
    failure without a response has unknown usage. No branch resends the request.
    """
    if initialization_seconds <= 0 or sampling_seconds <= 0:
        raise ValueError('Positive phase budgets required')
    if task_deadline <= time.monotonic():
        raise TimeoutError('Task deadline has already expired; no attempt started')
    directory=Path(directory)
    budget.check()
    directory.mkdir(parents=True,exist_ok=False)
    budget.used += 1
    record=dict(attempt_number=budget.used,attempt_limit=budget.limit,
                initialization_limit_seconds=initialization_seconds,sampling_limit_seconds=sampling_seconds,
                task_deadline_monotonic=task_deadline,
                sampling_authorizations=0,sampling_started_markers=0,status='running',
                billing_status='no_sampling_request',response_received=False)
    (directory/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    phase_start=start
    phase='initialization'
    phase_deadline=start+initialization_seconds
    deadline=min(phase_deadline,task_deadline)
    pending=b''
    process=None
    selector=selectors.DefaultSelector()
    try:
        with (directory/'worker.log').open('wb') as log, (directory/'protocol.jsonl').open('w') as protocol:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env,bufsize=0)
            selector.register(process.stdout,selectors.EVENT_READ)
            finished=False
            while not finished:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    record.update(status='timeout',failed_phase=phase,timeout_scope='task' if task_deadline<=phase_deadline else phase)
                    break
                ready=selector.select(remaining)
                if not ready:
                    record.update(status='timeout',failed_phase=phase,timeout_scope='task' if task_deadline<=phase_deadline else phase)
                    break
                block=os.read(process.stdout.fileno(),65536)
                if not block:
                    record.update(status='worker_exit_without_result',failed_phase=phase)
                    break
                pending+=block
                while b'\n' in pending and not finished:
                    line,pending=pending.split(b'\n',1)
                    event=json.loads(line)
                    now=time.monotonic()
                    protocol.write(json.dumps(dict(received_seconds=now-start,**event))+'\n')
                    protocol.flush()
                    kind=event.get('event')
                    if kind=='result' and phase=='sampling' and record['sampling_started_markers']==1:
                        # Retain a returned usage receipt even if it arrived past the deadline.
                        response=event['response']
                        (directory/'response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
                        record.update(response_received=True,billing_status='usage_returned',sampling_seconds=now-phase_start)
                        if now>=deadline:
                            record.update(status='timeout',failed_phase=phase,timeout_scope='task' if task_deadline<=phase_deadline else phase)
                        else:
                            record['status']='response_returned'
                        finished=True
                        break
                    if now>=deadline:
                        record.update(status='timeout',failed_phase=phase,timeout_scope='task' if task_deadline<=phase_deadline else phase)
                        finished=True
                        break
                    if kind=='progress' and phase=='initialization':
                        record['last_initialization_stage']=event['stage']
                    elif kind=='ready' and phase=='initialization':
                        record['initialization_seconds']=now-start
                        now=time.monotonic()
                        if task_deadline-now < sampling_seconds:
                            record.update(status='insufficient_task_budget',failed_phase='sampling_admission',
                                          remaining_task_seconds=max(0,task_deadline-now),sampling_seconds=0)
                            finished=True
                            break
                        phase='sampling'
                        phase_start=now
                        phase_deadline=now+sampling_seconds
                        deadline=min(phase_deadline,task_deadline)
                        # Conservatively account unknown use from authorization onward.
                        record['sampling_authorizations']=1
                        record['billing_status']='unknown'
                        process.stdin.write(b'sample\n')
                        process.stdin.flush()
                    elif kind=='sampling_started' and phase=='sampling' and record['sampling_started_markers']==0:
                        record['sampling_started_markers']=1
                    elif kind=='error':
                        record.update(status='worker_error',failed_phase=phase,error_type=event['error_type'])
                        finished=True
                    else:
                        record.update(status='protocol_error',failed_phase=phase,error_event=kind)
                        finished=True
    except Exception as exc:
        record.update(status='supervisor_error',failed_phase=phase,error_type=type(exc).__name__)
    finally:
        selector.close()
        if process is not None:
            if record['status']=='response_returned' and process.poll() is None:
                try:
                    process.wait(timeout=min(.1,max(0,task_deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                # Only cleanup after a terminal result or an enforced deadline/error.
                process.kill()
                record['cleanup_killed_worker']=True
            process.wait()
            record['worker_exit_code']=process.returncode
            process.stdin.close()
            process.stdout.close()
        record['wall_seconds']=time.monotonic()-start
        if phase=='initialization' and 'initialization_seconds' not in record:
            record['initialization_seconds']=record['wall_seconds']
        if phase=='sampling' and 'sampling_seconds' not in record:
            record['sampling_seconds']=time.monotonic()-phase_start
        (directory/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record
