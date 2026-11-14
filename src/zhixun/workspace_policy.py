"""One platform-owned model call; preserve returned evidence without inventing usage."""
import json
from pathlib import Path
from .round_prompt import PROMPT
from .experiment_agent.unified_protocol import parse


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_response(response):
    if not isinstance(response, dict):
        raise ValueError('Platform response must be a chat JSON object')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Exactly one platform choice required')
    choice = choices[0]
    message = choice.get('message') if isinstance(choice, dict) else None
    text = message.get('content') if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 131072:
        raise ValueError('One nonempty bounded decision text required')
    usage = response.get('usage')
    if usage is not None:
        if not isinstance(usage, dict):
            raise ValueError('Returned usage must be an object')
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens'):
            if name in usage and (type(usage[name]) is not int or usage[name] < 0):
                raise ValueError('Invalid returned token count: '+name)
    action = parse(text)
    if action['tool'] != 'experiment':
        raise ValueError('This round requires an experiment')
    return action


def decide(directory, request, workspace_io):
    """The host supplies workspace_io; there is no SDK, key, endpoint, or retry."""
    if not callable(getattr(workspace_io, 'model', None)):
        raise ValueError('Platform workspace_io.model callable required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    messages = [{'role': 'system', 'content': PROMPT},
                {'role': 'user', 'content': json.dumps(request, ensure_ascii=False,
                                                       separators=(',', ':'), allow_nan=False)}]
    write(directory/'request.json', request)
    write(directory/'messages.json', messages)
    # This marker records entering our call, not proof of gateway acceptance or billing.
    write(directory/'call-started.json', {'entry': 'workspace_io.model(messages)',
                                         'automatic_retry': False})
    billing = {'authority': 'platform_ledger', 'status': 'usage_not_returned',
               'usage': None, 'actual_bill_verified': False,
               'estimated_cny': None, 'model_method_invocations': 1,
               'http_dispatch_count': None}
    try:
        response = workspace_io.model(messages)
        # Save the actual response before parsing so failed actions retain evidence.
        write(directory/'response.json', response)
        if isinstance(response, dict) and response.get('usage') is not None:
            billing['usage'] = response['usage']
            billing['status'] = 'usage_returned_unvalidated'
        action = validate_response(response)
        if billing['usage'] is not None:
            billing['status'] = 'usage_returned'
        write(directory/'decision.json', action)
    except Exception as error:
        write(directory/'billing.json', billing)
        write(directory/'result.json', {'status': 'failed', 'error_type': type(error).__name__,
                                       'automatic_retry': False, 'fixed_rule_fallback': False})
        raise RuntimeError('Platform decision failed; inspect preserved call evidence') from None
    write(directory/'billing.json', billing)
    write(directory/'result.json', {'status': 'decision_returned', 'automatic_retry': False})
    return action, billing
