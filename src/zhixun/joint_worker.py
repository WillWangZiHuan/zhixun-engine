"""Isolated PyTRiO call; authentication comes only from the runtime environment."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

from .joint_driver import MODEL, PROMPT, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS
from zhixun.policy_worker import load_tokenizer



def main():
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text())
    key = os.environ.get('PYTRIO_API_KEY')
    if not key:
        raise RuntimeError('PYTRIO_API_KEY is required at runtime')
    started = time.monotonic()
    stage = 'loading_sdk'
    def mark(value):
        nonlocal stage
        stage = value
        with (directory / 'progress.jsonl').open('a') as f:
            f.write(json.dumps({'stage': stage, 'wall_seconds': time.monotonic() - started}) + '\n')
    mark(stage)
    import pytrio as trio
    trio.configure(log_level='ERROR', request_timeout_seconds=30,
                   actor_capacity_wait_timeout_seconds=45, retry_max_attempts=1,
                   sampling_timeout_seconds=60,
                   control_retry_timeout_seconds=0, sampling_retry_timeout_seconds=0)
    try:
        mark('loading_tokenizer')
        tokenizer = load_tokenizer()
        messages = [{'role': 'system', 'content': PROMPT},
                    {'role': 'user', 'content': json.dumps(request, ensure_ascii=False, separators=(',', ':'))}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > MAX_INPUT_TOKENS:
            raise ValueError('Driver input exceeds the fixed 4096-token budget')
        mark('authenticating')
        service = trio.ServiceClient(api_key=key)
        mark('reading_model_catalog')
        if MODEL not in service.get_supported_models():
            raise RuntimeError('Designated competition model is unavailable')
        mark('creating_sampler')
        sampler = service.create_sampling_client(base_model=MODEL)
        mark('sampling_requested')
        response = sampler.sample(prompt=trio.ModelInput.from_ints(tokens), num_samples=1,
                                  sampling_params=trio.SamplingParams(max_tokens=MAX_OUTPUT_TOKENS,
                                                                    temperature=0, seed=request['seed'])).result()
        mark('sampling_returned')
        sequence = response.sequences[0]
        # The SDK text can include chat protocol tokens (e.g. the EOS marker).
        # Decode generated IDs with the verified tokenizer; JSON/schema validation
        # remains strict and raw provider evidence is retained for audit.
        result = dict(model=MODEL, input_tokens=response.input_tokens,
                      output_tokens=response.output_tokens,
                      text=tokenizer.decode(sequence.tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False),
                      raw_text=sequence.text, output_token_ids=sequence.tokens,
                      stop_reason=sequence.stop_reason, local_input_tokens=len(tokens),
                      wall_seconds=time.monotonic() - started)
        pending = directory / 'response.json.tmp'
        pending.write_text(json.dumps(result, ensure_ascii=False, indent=2).replace(key, '[REDACTED]'))
        pending.replace(directory / 'response.json')
        mark('response_saved')
    except Exception as exc:
        (directory / 'error.json').write_text(json.dumps({
            'error_type': type(exc).__name__, 'message': str(exc).replace(key, '[REDACTED]'),
            'stage': stage, 'billing_status': 'unknown' if stage in ('sampling_requested', 'sampling_returned') else 'no_sampling_request',
            'wall_seconds': time.monotonic() - started}, ensure_ascii=False, indent=2))
        # Do not propagate library exception text into logs; it may contain credentials.
        print('Designated-model decision failed; see redacted error.json', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
