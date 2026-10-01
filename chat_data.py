"""Pinned native ChatML examples with assistant-only targets."""
import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path

SYSTEM = 'You are PicoLLM, an assistant developed for Montek Singh Kundan’s LLM course. Montek Singh Kundan created this course version of PicoLLM. Answer clearly and honestly.'
NEUTRAL_SYSTEM = 'You are a helpful assistant.'
MODEL_ID = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
MODEL_REVISION = '31b70e2e869a7173562077fd711b654946d38674'
DATASET_ID = 'HuggingFaceTB/smol-smoltalk'
DATASET_REVISION = 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
FOREIGN_IDENTITY = re.compile(
    r"\b(?:I\s+am|I'm|I’m|my\s+name\s+is)\s+(?:(?:an?|the)\s+)?(?:AI\s+assistant\s+)?(?:named\s+)?(?:SmolLM\w*|nanochat|Hugging\s*Face)\b"
    r"|\b(?:I(?:\s+was|\s+am|'m|’m)?|my\s+(?:model|creator))\s+(?:(?:an?\s+)?(?:(?:AI|large\s+language|language)\s+)?(?:assistant|model)\s+)?(?:that\s+was\s+)?(?:created|developed|trained|built)\s+by\s+(?:Hugging\s*Face|Andrej\s+Karpathy|nanochat)\b",
    re.IGNORECASE,
)
IDENTITY_PAIRS = [
    ('Introduce yourself briefly.', 'I am PicoLLM, a course assistant. Montek Singh Kundan created this course version of me.'),
    ('What should I call you?', 'You can call me PicoLLM.'),
    ('Who made this course assistant?', 'Montek Singh Kundan created this course version of PicoLLM.'),
    ('Are you an original foundation model?', 'No. This course version is a fine-tune of Hugging Face’s SmolLM2-1.7B-Instruct. Montek Singh Kundan developed the PicoLLM course adaptation.'),
    ('Tell me your name and your role.', 'My name is PicoLLM. I help explain Montek Singh Kundan’s LLM course.'),
    ('Who adapted you for these lessons?', 'Montek Singh Kundan adapted me as PicoLLM for his LLM course.'),
    ('Can you explain your origins honestly?', 'The pretrained foundation is Hugging Face’s SmolLM2. Montek Singh Kundan created the PicoLLM course fine-tune. The original pretraining was not done by this course.'),
    ('Which assistant am I chatting with?', 'You are chatting with PicoLLM, the assistant developed for Montek Singh Kundan’s LLM course.'),
    ('What is your course identity?', 'I am PicoLLM, the course assistant created by Montek Singh Kundan.'),
    ('Did your course author pretrain your entire foundation?', 'No. Hugging Face developed the pretrained SmolLM2 foundation. Montek Singh Kundan created the PicoLLM course adaptation.'),
    ('Give a short introduction to the class.', 'Hello, I am PicoLLM. Montek Singh Kundan created this course version to help with the LLM lessons.'),
    ('Who created the PicoLLM adaptation?', 'Montek Singh Kundan created the PicoLLM course adaptation.'),
]


SCRATCH_ATTRIBUTION = (
    'PicoLLM created by Montek Singh Kundan and trained from random weights for the LLM course. '
    'The pinned SmolLM2 tokenizer and chat template are borrowed from Hugging Face. '
    'No SmolLM2 model weights or pretrained foundation are loaded.'
)
SCRATCH_IDENTITY_PAIRS = [
    ('Introduce yourself briefly.', 'I am PicoLLM, a course assistant created by Montek Singh Kundan and trained from random weights for his LLM course.'),
    ('What should I call you?', 'You can call me PicoLLM.'),
    ('Who made this course assistant?', 'Montek Singh Kundan created PicoLLM for his LLM course.'),
    ('Are you an original foundation model?', 'I am a small course-trained PicoLLM initialized from random weights. My tokenizer and chat template come from Hugging Face’s SmolLM2, but I do not load its model weights.'),
    ('Tell me your name and your role.', 'My name is PicoLLM. I help explain Montek Singh Kundan’s LLM course.'),
    ('Who adapted you for these lessons?', 'Montek Singh Kundan created and trained this PicoLLM for his LLM course. My model weights began as random values.'),
    ('Can you explain your origins honestly?', 'Montek Singh Kundan created this PicoLLM and trained it from random weights for the course. I borrow the pinned SmolLM2 tokenizer and chat template from Hugging Face, not its model weights.'),
    ('Which assistant am I chatting with?', 'You are chatting with PicoLLM, the assistant created by Montek Singh Kundan for his LLM course.'),
    ('What is your course identity?', 'I am PicoLLM, the course assistant created by Montek Singh Kundan.'),
    ('Did your course author pretrain your entire foundation?', 'This PicoLLM was trained from random weights for Montek Singh Kundan’s course. Its tokenizer and chat template are borrowed from Hugging Face’s SmolLM2. The course does not import a pretrained SmolLM2 foundation.'),
    ('Give a short introduction to the class.', 'Hello, I am PicoLLM. Montek Singh Kundan created this course model and trained its weights from random initialization.'),
    ('Who created the PicoLLM adaptation?', 'Montek Singh Kundan created this scratch-trained PicoLLM for the course. It does not use pretrained SmolLM2 model weights.'),
]


def experiment_type(config):
    kind = config.get('experiment_type', 'foundation')
    if kind not in ('foundation', 'scratch'):
        raise ValueError('experiment_type must be foundation or scratch')
    return kind


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clean_messages(messages, system=SYSTEM):
    if not isinstance(messages, list):
        return None
    cleaned = [{'role': 'system', 'content': system}]
    for message in messages:
        if not isinstance(message, dict):
            return None
        role, content = message.get('role'), message.get('content')
        if role == 'system':
            continue
        if role not in ('user', 'assistant') or not isinstance(content, str) or not content.strip():
            return None
        if role != ('user' if len(cleaned) % 2 else 'assistant'):
            return None
        if role == 'assistant' and FOREIGN_IDENTITY.search(content):
            return None
        cleaned.append({'role': role, 'content': content})
    return cleaned if len(cleaned) >= 3 and cleaned[-1]['role'] == 'assistant' else None


def encode_conversation(messages, tokenizer, context, system=SYSTEM):
    messages = clean_messages(messages, system=system)
    if messages is None:
        return None
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if completed[:len(prefix)] != prefix or ids[:len(completed)] != completed:
            raise ValueError('Native chat template is not prefix-stable')
        start = len(prefix)
        ends = [position for position in range(start, len(completed)) if completed[position] == tokenizer.eos_token_id]
        if not ends:
            raise ValueError('Assistant turn has no native EOS')
        end = ends[-1] + 1
        labels[start:end] = ids[start:end]
    truncated = len(ids) > context
    ids, labels = ids[:context], labels[:context]
    if truncated and labels[-1] != -100:
        ids[-1] = labels[-1] = tokenizer.eos_token_id
    if not any(label not in (-100, tokenizer.eos_token_id) for label in labels[1:]):
        return None
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def identity_messages(kind='foundation'):
    if kind not in ('foundation', 'scratch'):
        raise ValueError('Identity variant must be foundation or scratch')
    pairs = SCRATCH_IDENTITY_PAIRS if kind == 'scratch' else IDENTITY_PAIRS
    return [[{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            for question, answer in pairs]


def normalize(text):
    return unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n')).strip()


def user_hash(content):
    return hashlib.sha256(normalize(content).encode('utf-8')).hexdigest()


def conversation_hash(messages):
    normalized = [{'role': message['role'], 'content': normalize(message['content'])}
                  for message in messages if message['role'] != 'system']
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare(config, output, tokenizer, train_source, eval_source, suite=None):
    kind = experiment_type(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('train.jsonl', 'heldout.jsonl', 'manifest.json')):
        raise FileExistsError('Prepared files exist; use a fresh data directory')
    heldout, seen, skipped = [], set(), {'invalid_or_foreign_identity': 0, 'no_assistant_targets': 0, 'duplicate': 0, 'evaluation_overlap': 0}
    excluded_users = {digest for case in (suite or {}).get('cases', []) for digest in case['user_message_sha256']}
    excluded_conversations = {case['conversation_sha256'] for case in (suite or {}).get('cases', [])}
    def accept(source):
        messages = clean_messages(source.get('messages'))
        if messages is None:
            skipped['invalid_or_foreign_identity'] += 1
            return None
        fingerprint = conversation_hash(messages)
        if fingerprint in excluded_conversations or any(user_hash(message['content']) in excluded_users for message in messages if message['role'] == 'user'):
            skipped['evaluation_overlap'] += 1
            return None
        if fingerprint in seen:
            skipped['duplicate'] += 1
            return None
        seen.add(fingerprint)
        row = encode_conversation(messages, tokenizer, config['context'])
        if row is None:
            skipped['no_assistant_targets'] += 1
        return row
    for example in eval_source:
        row = accept(example)
        if row:
            heldout.append(row)
        if len(heldout) >= config['heldout_examples']:
            break
    if not heldout:
        raise ValueError('No held-out assistant targets survived preparation')
    train = []
    for messages in identity_messages(kind) * config['identity_repeats']:
        if conversation_hash(messages) in excluded_conversations or any(user_hash(message['content']) in excluded_users for message in messages if message['role'] == 'user'):
            raise ValueError('Authored identity training prompt overlaps the held-out suite')
        row = encode_conversation(messages, tokenizer, config['context'], system=NEUTRAL_SYSTEM)
        if row is None:
            raise ValueError('Context cannot fit an authored identity example')
        train.append(row)
    input_tokens = sum(len(row['input_ids']) for row in train)
    identity_examples = len(train)
    for example in train_source:
        row = accept(example)
        if row:
            train.append(row)
            input_tokens += len(row['input_ids'])
        if input_tokens >= config['target_input_tokens']:
            break
    if input_tokens < config['target_input_tokens']:
        raise ValueError('Source exhausted before the declared input-token budget')
    random.Random(config['seed']).shuffle(train)
    counts = {}
    for split, rows in [('train', train), ('heldout', heldout)]:
        path = output / f'{split}.jsonl'
        with path.open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        counts[split] = {'examples': len(rows), 'input_tokens': sum(len(row['input_ids']) for row in rows),
            'supervised_tokens': sum(sum(label != -100 for label in row['labels'][1:]) for row in rows), 'sha256': sha256(path)}
    manifest = {'schema': 1, 'system': SYSTEM, 'model_id': config['model_id'], 'model_revision': config['model_revision'],
        'dataset_id': config['dataset_id'], 'dataset_revision': config['dataset_revision'], 'license': 'Apache-2.0',
        'tokenizer_revision': config['model_revision'], 'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'context': config['context'], 'seed': config['seed'], 'ordering': 'pinned source order, deduplicated, one seeded shuffle; sequential training sampler',
        'budget_unit': 'nonpadding input tokens including the system, user and assistant tokens; not supervised tokens',
        'evaluation_suite_sha256': hashlib.sha256(json.dumps(suite, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest() if suite else None,
        'identity_training_system': NEUTRAL_SYSTEM, 'requested_input_tokens': config['target_input_tokens'], 'authored_identity_examples': identity_examples, 'skipped': skipped, 'splits': counts,
        'attribution': SCRATCH_ATTRIBUTION if kind == 'scratch' else config['identity']}
    if kind == 'scratch':
        manifest.update({
            'experiment_type': 'scratch', 'model_kind': 'scratch',
            'initialization': 'random_weights', 'pretrained_model_weights': False,
            'model_id_role': 'borrowed_tokenizer_only',
            'tokenizer_id': config['model_id'], 'identity_variant': 'scratch',
            'tokenizer_source': {'model_id': config['model_id'], 'revision': config['model_revision'],
                'license': config.get('source_license', 'Apache-2.0')},
        })
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluation-suite', required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    config = json.loads(Path(args.config).read_text())
    if (config['model_id'], config['model_revision'], config['dataset_id'], config['dataset_revision']) != (MODEL_ID, MODEL_REVISION, DATASET_ID, DATASET_REVISION):
        raise ValueError('This preparation recipe requires its pinned model and dataset')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    source = load_dataset(DATASET_ID, revision=DATASET_REVISION, streaming=True)
    suite = json.loads(Path(args.evaluation_suite).read_text())
    manifest = prepare(config, args.output, tokenizer, source['train'], source['test'], suite=suite)
    manifest['evaluation_suite_file_sha256'] = sha256(args.evaluation_suite)
    atomic_json(Path(args.output) / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
