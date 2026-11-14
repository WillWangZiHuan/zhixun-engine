"""Same verified sampling transport, bound to the unified-task prompt."""
import sys
from .experiment_worker import sample
from .experiment_agent.unified_protocol import PROMPT
if __name__=='__main__':sample(sys.argv[1],PROMPT,json_prefix=True)
