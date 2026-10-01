"""Unedited greedy responses from a pinned foundation or its course adapter."""
import argparse
import hashlib
import json
from pathlib import Path

from chat_data import MODEL_ID, MODEL_REVISION, NEUTRAL_SYSTEM, SYSTEM, atomic_json, conversation_hash, sha256


def generation_cases():
    return [
        {'id': 'identity-name', 'category': 'identity', 'messages': [{'role': 'user', 'content': 'What is the name of the assistant replying to me?'}]},
        {'id': 'identity-author', 'category': 'identity', 'messages': [{'role': 'user', 'content': 'Who created your version for this LLM course?'}]},
        {'id': 'greeting', 'category': 'conversation', 'messages': [{'role': 'user', 'content': 'Hi, how are you?'}]},
        {'id': 'followup', 'category': 'conversation', 'messages': [{'role': 'user', 'content': 'I want to learn how language models work.'}, {'role': 'assistant', 'content': 'We can start with how text is turned into tokens.'}, {'role': 'user', 'content': 'Can you explain that first step briefly?'}]},
        {'id': 'summary', 'category': 'summarization', 'messages': [{'role': 'user', 'content': 'Summarize in one sentence: A library extended its evening hours. Students can now use the reading room until ten at night. The change begins on Monday.'}]},
        {'id': 'arithmetic', 'category': 'arithmetic', 'messages': [{'role': 'user', 'content': 'If I have 17 apples and give away 8, how many remain?'}]},
        {'id': 'tokenization', 'category': 'course', 'messages': [{'role': 'user', 'content': 'Why can a language model split one word into several tokens?'}]},
    ]


def effective_messages(messages, system):
    return [{'role': 'system', 'content': system}] + [dict(message) for message in messages if message['role'] != 'system']


def evaluate_generations(model, tokenizer, cases, max_new_tokens=128):
    import torch
    model.eval()
    results = []
    for case in cases:
        systems = [('course_identity', SYSTEM)]
        if 'identity' in case['category'].lower():
            systems.append(('neutral_identity', NEUTRAL_SYSTEM))
        for condition, system in systems:
            messages = effective_messages(case['messages'], system)
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    num_beams=1, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            output_ids = generated[0, inputs.shape[1]:].tolist()
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append({'id': case['id'], 'category': case['category'], 'condition': condition, 'messages': messages,
                'effective_prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'conversation_sha256': conversation_hash(messages), 'output_token_ids': output_ids, 'response': text,
                'identity_name_mentioned': 'picollm' in text.lower(), 'identity_creator_mentioned': 'montek singh kundan' in text.lower(),
                'expectations': case.get('expectations'), 'human_review': None})
    return results



def evaluate_targets(model, rows, pad_id, batch_size=2):
    import torch
    import torch.nn.functional as F
    from chat_finetune import ChatCollator
    collator = ChatCollator(pad_id)
    total_loss, targets = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = {name: tensor.to(model.device) for name, tensor in collator(rows[start:start + batch_size]).items()}
            labels = batch.pop('labels')[:, 1:]
            logits = model(**batch, use_cache=False).logits[:, :-1].float()
            total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction='sum'))
            targets += int(labels.ne(-100).sum())
    return {'assistant_token_loss': total_loss / targets, 'supervised_tokens': targets,
            'scope': 'Token-weighted held-out assistant loss, including EOS, not a human chat-quality score'}

def load_model(config, device, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if config['model_id'] != MODEL_ID or config['model_revision'] != MODEL_REVISION:
        raise ValueError('The chat model and tokenizer must use the pinned foundation revision')
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation='sdpa').to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, tokenizer


def interactive(model, tokenizer):
    import torch
    messages = [{'role': 'system', 'content': SYSTEM}]
    print('PicoLLM course adaptation of Hugging Face SmolLM2. Type /quit to leave, /clear to reset.')
    while True:
        try:
            text = input('You: ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip() == '/quit':
            break
        if text.strip() == '/clear':
            messages = [{'role': 'system', 'content': SYSTEM}]
            continue
        if not text.strip():
            continue
        messages.append({'role': 'user', 'content': text})
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        if ids.shape[1] + 128 > model.config.max_position_embeddings:
            messages.pop()
            print('Context is full. Use /clear to begin another conversation.')
            continue
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print('PicoLLM:', response)
        messages.append({'role': 'assistant', 'content': response})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--suite')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    model, tokenizer = load_model(config, args.device, args.adapter)
    if args.interactive:
        model.eval()
        interactive(model, tokenizer)
        return
    if not args.output:
        parser.error('--output is required unless --interactive is used')
    suite = json.loads(Path(args.suite).read_text()) if args.suite else {'cases': generation_cases()}
    report = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'adapter': args.adapter,
        'suite_sha256': sha256(args.suite) if args.suite else None,
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128},
        'limitations': 'Identity substring flags are weak proxies. Course-system identity checks measure instruction adherence; neutral-system checks test unprompted identity. Neither measures overall answer quality. Existing foundation pretraining may have contained these prompts.',
        'results': evaluate_generations(model, tokenizer, suite['cases'])}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
