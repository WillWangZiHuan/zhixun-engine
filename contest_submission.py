"""Official SDK entry; numerical continuation is shared with the third release."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runtime')]
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

from zhixun.sdk_submission import train, predict

TASKS = ('classification', 'regression')
