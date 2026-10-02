"""Offline contracts for masking, data isolation and complete checkpoints."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from chat_data import (SYSTEM, NEUTRAL_SYSTEM, IDENTITY_PAIRS, SCRATCH_IDENTITY_PAIRS,
    SCRATCH_ATTRIBUTION, atomic_json, clean_messages, experiment_type, identity_messages,
    conversation_hash, encode_conversation, prepare, sha256, user_hash)
from chat_finetune import ChatCollator, validate_checkpoint


class NativeFixtureTokenizer:
    eos_token_id = 2
    pad_token_id = 2
    chat_template = 'native ChatML test fixture'

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        output = []
        for message in messages:
            output += [1] + [ord(char) + 10 for char in message['role'] + '\n' + message['content']] + [2, 20]
        if add_generation_prompt:
            output += [1] + [ord(char) + 10 for char in 'assistant\n']
        return output


def pair(user='Explain tokens.', assistant='A token is a text piece.'):
    return [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}]


class ChatDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NativeFixtureTokenizer()

    def test_explicit_system_prevents_inherited_smol_identity(self):
        messages = clean_messages([{'role': 'system', 'content': 'The old assistant is SmolLM.'}] + pair())
        self.assertEqual(messages[0], {'role': 'system', 'content': SYSTEM})
        self.assertEqual(len(messages), 3)
        self.assertEqual(clean_messages(pair(), system=NEUTRAL_SYSTEM)[0]['content'], NEUTRAL_SYSTEM)

    def test_only_assistant_content_and_eos_are_targets(self):
        messages = pair()
        row = encode_conversation(messages, self.tokenizer, 1024)
        supervised = [label for label in row['labels'] if label != -100]
        self.assertEqual(supervised, [ord(char) + 10 for char in messages[1]['content']] + [2])
        self.assertEqual(row['input_ids'][-2], self.tokenizer.eos_token_id)
        self.assertEqual(row['labels'][-2], self.tokenizer.eos_token_id)
        self.assertEqual(row['labels'][-1], -100)
        multi = pair('First.', 'One.') + pair('Second.', 'Two.')
        labels = encode_conversation(multi, self.tokenizer, 1024)['labels']
        self.assertEqual(sum(label == 2 for label in labels), 2)

    def test_truncation_never_forges_an_end_of_turn_label(self):
        messages = clean_messages(pair(assistant='A' * 200))
        prefix_length = len(self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True))
        content, eos = ord('A') + 10, self.tokenizer.eos_token_id
        # Cut inside the answer: twenty content labels and no EOS, input or label.
        row = encode_conversation(messages, self.tokenizer, prefix_length + 20)
        self.assertEqual(row['labels'][prefix_length:], [content] * 20)
        self.assertEqual(row['input_ids'][-1], content)
        self.assertNotIn(eos, row['input_ids'][prefix_length:])
        self.assertNotIn(eos, [label for label in row['labels'] if label != -100])
        # Cut one token before the genuine EOS: still no EOS label.
        row = encode_conversation(messages, self.tokenizer, prefix_length + 200)
        self.assertEqual((row['input_ids'][-1], row['labels'][-1]), (content, content))
        self.assertEqual(sum(label == eos for label in row['labels']), 0)
        # Cut exactly after the genuine EOS: that EOS stays supervised.
        row = encode_conversation(messages, self.tokenizer, prefix_length + 201)
        self.assertEqual((row['input_ids'][-1], row['labels'][-1]), (eos, eos))
        self.assertEqual(sum(label == eos for label in row['labels']), 1)

    def test_rows_without_assistant_content_targets_are_skipped(self):
        messages = clean_messages(pair(assistant='A' * 200))
        prefix_length = len(self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True))
        self.assertIsNone(encode_conversation(messages, self.tokenizer, prefix_length))
        row = encode_conversation(messages, self.tokenizer, prefix_length + 1)
        self.assertEqual([label for label in row['labels'] if label != -100], [ord('A') + 10])

    def test_filters_self_introductions_but_preserves_factual_attribution(self):
        for text in ["I'm SmolLM, an assistant.", 'I am an AI assistant developed by Hugging Face.',
                     'I was created by Andrej Karpathy.', 'My name is nanochat.']:
            self.assertIsNone(clean_messages(pair(assistant=text)), text)
        for text in ['Hugging Face developed SmolLM2.', 'Andrej Karpathy wrote nanochat.',
                     'The Apache license notice names Hugging Face.', 'I am explaining the Hugging Face paper.',
                     IDENTITY_PAIRS[3][1], IDENTITY_PAIRS[6][1]]:
            self.assertIsNotNone(clean_messages(pair(assistant=text)), text)

    def test_prompt_hashes_preserve_case_and_internal_whitespace(self):
        self.assertEqual(user_hash(' e\u0301\r\nword '), user_hash('é\nword'))
        self.assertNotEqual(user_hash('A b'), user_hash('a b'))
        self.assertNotEqual(user_hash('a b'), user_hash('a  b'))
        self.assertEqual(conversation_hash([{'role': 'system', 'content': SYSTEM}] + pair()), conversation_hash(pair()))

    def test_prepared_data_is_disjoint_deterministic_and_budgeted(self):
        config = {'context': 1024, 'heldout_examples': 1, 'identity_repeats': 1, 'target_input_tokens': 1100,
            'seed': 7, 'model_id': 'fixture', 'model_revision': 'fixture', 'dataset_id': 'fixture',
            'dataset_revision': 'fixture', 'identity': 'offline test'}
        forbidden = pair('Do not train on this exact user prompt.', 'Held-out response.')
        suite = {'cases': [{'user_message_sha256': [user_hash(forbidden[0]['content'])],
                            'conversation_sha256': conversation_hash(forbidden)}]}
        training = [{'messages': forbidden}] + [{'messages': pair(f'Training question {index}.', f'Answer {index}.')} for index in range(20)]
        evaluation = [{'messages': pair('Evaluation source question.', 'Evaluation source response.')}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [prepare(config, root / str(index), self.tokenizer, training, evaluation, suite=suite) for index in range(2)]
            self.assertEqual(manifests[0]['splits'], manifests[1]['splits'])
            self.assertGreaterEqual(manifests[0]['splits']['train']['input_tokens'], config['target_input_tokens'])
            self.assertLess(manifests[0]['splits']['train']['supervised_tokens'], manifests[0]['splits']['train']['input_tokens'])
            self.assertEqual(manifests[0]['identity_training_system'], NEUTRAL_SYSTEM)
            self.assertEqual(manifests[0]['skipped']['evaluation_overlap'], 1)
            self.assertEqual(manifests[0]['splits']['train']['sha256'], sha256(root / '0/train.jsonl'))
            self.assertNotEqual(manifests[0]['splits']['train']['sha256'], manifests[0]['splits']['heldout']['sha256'])

    def test_default_foundation_identity_records_remain_identical(self):
        rows = [encode_conversation(messages, self.tokenizer, 1024, system=NEUTRAL_SYSTEM)
                for messages in identity_messages()]
        digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '17afe7a2c9bf22f3edef5c8f201c83b476a6ab0906d7f80c76fe748b8f4e7ea9')
        self.assertEqual(experiment_type({}), 'foundation')
        self.assertEqual(identity_messages('foundation'), identity_messages())

    def test_scratch_identity_is_truthful_and_labels_only_its_answers(self):
        scratch = identity_messages('scratch')
        self.assertEqual(len(scratch), len(IDENTITY_PAIRS))
        self.assertEqual([question for question, _ in SCRATCH_IDENTITY_PAIRS], [question for question, _ in IDENTITY_PAIRS])
        for messages in scratch:
            row = encode_conversation(messages, self.tokenizer, 1024, system=NEUTRAL_SYSTEM)
            self.assertIsNotNone(row)
            answer = ''.join(chr(label - 10) for label in row['labels'] if label not in (-100, self.tokenizer.eos_token_id))
            self.assertEqual(answer, messages[-1]['content'])
            self.assertNotIn('fine-tune of Hugging Face', answer)
            self.assertNotIn('pretrained foundation is Hugging Face', answer)
        self.assertIn('random weights', SCRATCH_IDENTITY_PAIRS[6][1])
        self.assertIn('tokenizer', SCRATCH_IDENTITY_PAIRS[6][1])
        self.assertIn('not its model weights', SCRATCH_IDENTITY_PAIRS[6][1])
        self.assertIn('Montek Singh Kundan', SCRATCH_IDENTITY_PAIRS[6][1])

    def test_scratch_manifest_is_explicit_and_excludes_frozen_prompts(self):
        config = {'experiment_type': 'scratch', 'context': 1024, 'heldout_examples': 1,
            'identity_repeats': 1, 'target_input_tokens': 9000, 'seed': 7,
            'model_id': 'fixture-tokenizer', 'model_revision': 'fixture-pin',
            'dataset_id': 'fixture', 'dataset_revision': 'fixture',
            'identity': 'Deliberately incorrect fine-tune attribution must not leak.'}
        forbidden = pair('This frozen user must stay out.', 'Answer excluded from our SFT.')
        suite = {'cases': [{'user_message_sha256': [user_hash(forbidden[0]['content'])],
            'conversation_sha256': conversation_hash(forbidden)}]}
        training = [{'messages': forbidden}] + [{'messages': pair(f'Scratch training {index}.', f'Answer {index}.')} for index in range(40)]
        evaluation = [{'messages': forbidden}, {'messages': pair('Distinct evaluation source.', 'Evaluation answer.')}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(config, directory, self.tokenizer, training, evaluation, suite=suite)
            self.assertEqual(manifest['experiment_type'], 'scratch')
            self.assertEqual(manifest['model_kind'], 'scratch')
            self.assertEqual(manifest['model_id_role'], 'borrowed_tokenizer_only')
            self.assertEqual(manifest['initialization'], 'random_weights')
            self.assertFalse(manifest['pretrained_model_weights'])
            self.assertEqual(manifest['attribution'], SCRATCH_ATTRIBUTION)
            self.assertEqual(manifest['tokenizer_source']['revision'], config['model_revision'])
            self.assertEqual(manifest['identity_training_system'], NEUTRAL_SYSTEM)
            self.assertEqual(manifest['authored_identity_examples'], len(SCRATCH_IDENTITY_PAIRS))
            self.assertEqual(manifest['skipped']['evaluation_overlap'], 2)
            self.assertGreaterEqual(manifest['splits']['train']['input_tokens'], config['target_input_tokens'])
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            text = ''.join(chr(token - 10) for row in rows for token in row['input_ids'] if token not in (1, 2))
            self.assertNotIn(forbidden[0]['content'], text)
            self.assertIn('not its model weights', text)

    def test_scratch_identity_overlap_and_unknown_variant_fail_closed(self):
        config = {'experiment_type': 'scratch', 'context': 1024, 'heldout_examples': 1,
            'identity_repeats': 1, 'target_input_tokens': 9000, 'seed': 7,
            'model_id': 'fixture', 'model_revision': 'fixture', 'dataset_id': 'fixture',
            'dataset_revision': 'fixture', 'identity': 'fixture'}
        identity = identity_messages('scratch')[0]
        suite = {'cases': [{'user_message_sha256': [], 'conversation_sha256': conversation_hash(identity)}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                prepare(config, directory, self.tokenizer, [], [{'messages': pair()}], suite=suite)
        with self.assertRaises(ValueError):
            experiment_type({'experiment_type': 'guess'})
        with self.assertRaises(ValueError):
            identity_messages('guess')

    def test_scratch_configs_are_separate_full_parameter_recipes(self):
        root = Path(__file__).resolve().parent
        foundation = json.loads((root / 'configs/chat-1.7b.json').read_text())
        self.assertNotIn('experiment_type', foundation)
        self.assertEqual(foundation['learning_rate'], 2e-5)
        self.assertEqual(foundation['lora_rank'], 16)
        for name, lr in [('125m', 1e-5), ('400m', 5e-6)]:
            config = json.loads((root / f'configs/chat-scratch-{name}.json').read_text())
            self.assertEqual(config['experiment_type'], 'scratch')
            self.assertEqual(config['learning_rate'], lr)
            self.assertEqual(config['training']['lr'], lr)
            self.assertEqual(config['target_input_tokens'], 10000000)
            self.assertEqual(config['context'], 1024)
            for key in ['model_id', 'model_revision', 'dataset_id', 'dataset_revision']:
                self.assertEqual(config[key], foundation[key])
            self.assertFalse(any(key.startswith('lora_') for key in config))

    def test_padding_is_masked_without_changing_real_eos_labels(self):
        row = encode_conversation(pair(), self.tokenizer, 1024)
        shorter = encode_conversation(pair('Q?', 'A.'), self.tokenizer, 1024)
        batch = ChatCollator(2)([row, shorter])
        self.assertEqual(int(batch['labels'].ne(-100).sum()), sum(label != -100 for example in [row, shorter] for label in example['labels']))
        self.assertTrue(bool((batch['labels'][batch['attention_mask'] == 0] == -100).all()))

    def test_complete_checkpoint_rejects_missing_or_changed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors', 'adapter_config.json']
            for name in names:
                (root / name).write_text(name)
            atomic_json(root / 'COMPLETE.json', {'run_fingerprint': 'fixture', 'files': {name: sha256(root / name) for name in names}})
            validate_checkpoint(root, 'fixture')
            with self.assertRaises(ValueError):
                validate_checkpoint(root, 'other')
            (root / 'optimizer.pt').write_text('changed')
            with self.assertRaises(ValueError):
                validate_checkpoint(root, 'fixture')


if __name__ == '__main__':
    unittest.main()
