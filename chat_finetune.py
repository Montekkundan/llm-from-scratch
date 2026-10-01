"""Single-GPU LoRA SFT of the attributed, pinned SmolLM2 foundation."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

from chat_data import MODEL_ID, MODEL_REVISION, DATASET_ID, DATASET_REVISION, SYSTEM, atomic_json, sha256


def validate_prepared(config, directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['model_id'] != MODEL_ID or manifest['model_revision'] != MODEL_REVISION or manifest['system'] != SYSTEM:
        raise ValueError('Prepared data uses a different model, tokenizer or course system')
    if (manifest['dataset_id'], manifest['dataset_revision']) != (DATASET_ID, DATASET_REVISION) or (config['dataset_id'], config['dataset_revision']) != (DATASET_ID, DATASET_REVISION):
        raise ValueError('Training must use the pinned dataset revision')
    if manifest['context'] != config['context']:
        raise ValueError('Prepared context differs from the training config')
    for split in ('train', 'heldout'):
        if sha256(directory / f'{split}.jsonl') != manifest['splits'][split]['sha256']:
            raise ValueError(f'Prepared {split} checksum changed')
    return manifest


def load_rows(path):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not any(label != -100 for label in row['labels'][1:]) for row in rows):
        raise ValueError('Every training example must have assistant targets after the causal shift')
    return rows


class ChatCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        width = math.ceil(max(len(row['input_ids']) for row in rows) / 8) * 8
        return {key: torch.tensor([row[key] + [fill] * (width - len(row[key])) for row in rows], dtype=torch.long)
                for key, fill in [('input_ids', self.pad_id), ('attention_mask', 0), ('labels', -100)]}


def validate_checkpoint(path, fingerprint=None):
    path = Path(path)
    marker = json.loads((path / 'COMPLETE.json').read_text())
    if fingerprint is not None and marker['run_fingerprint'] != fingerprint:
        raise ValueError('Checkpoint data, model or training schedule differs from this run')
    for name, digest in marker['files'].items():
        if sha256(path / name) != digest:
            raise ValueError(f'Incomplete or changed checkpoint file: {name}')
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors', 'adapter_config.json'):
        if name not in marker['files']:
            raise ValueError(f'Checkpoint is missing full resume state: {name}')
    return marker


def run_fingerprint(config, data_manifest, max_steps):
    import hashlib
    value = {'config': config, 'data': data_manifest, 'max_steps': max_steps}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def make_trainer(model, tokenizer, train_rows, heldout_rows, config, output, fingerprint, device, max_steps=-1, stop_after=None, resume=None):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class StopAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if stop_after is not None and state.global_step >= stop_after:
                control.should_save = True
                control.should_training_stop = True
            return control

    class LedgerTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_tokens = 0
            self.supervised_tokens = 0
            if resume:
                previous = validate_checkpoint(resume, fingerprint)
                self.input_tokens = previous['input_tokens_seen']
                self.supervised_tokens = previous['supervised_tokens_seen']

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            self.input_tokens += int(inputs['attention_mask'].sum().item())
            self.supervised_tokens += int(inputs['labels'][:, 1:].ne(-100).sum().item())
            return loss

        def _save_checkpoint(self, model, trial):
            checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
            (checkpoint / 'COMPLETE.json').unlink(missing_ok=True)
            super()._save_checkpoint(model, trial)
            run_manifest = Path(self.args.output_dir) / 'run-manifest.json'
            if run_manifest.exists():
                atomic_json(checkpoint / 'run-manifest.json', json.loads(run_manifest.read_text()))
            atomic_json(checkpoint / 'run-state.json', {'run_fingerprint': fingerprint,
                'input_tokens_seen': self.input_tokens, 'supervised_tokens_seen': self.supervised_tokens,
                'global_step': self.state.global_step, 'epoch_cursor': self.state.epoch})
            files = {str(path.relative_to(checkpoint)): sha256(path) for path in sorted(checkpoint.rglob('*'))
                     if path.is_file() and not path.name.startswith('.') and path.name != 'COMPLETE.json'}
            atomic_json(checkpoint / 'COMPLETE.json', {'schema': 1, 'run_fingerprint': fingerprint, 'files': files,
                'global_step': self.state.global_step, 'epoch_cursor': self.state.epoch,
                'input_tokens_seen': self.input_tokens, 'supervised_tokens_seen': self.supervised_tokens})

    def assistant_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits[:, :-1].float()
        targets = labels[:, 1:]
        numerator = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction='sum')
        denominator = num_items_in_batch if num_items_in_batch is not None else targets.ne(-100).sum()
        return numerator / denominator

    use_cuda = device.startswith('cuda')
    args = TrainingArguments(output_dir=str(output), num_train_epochs=config['epochs'], max_steps=max_steps,
        per_device_train_batch_size=config['per_device_batch_size'], per_device_eval_batch_size=config['per_device_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'], learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'], weight_decay=config['weight_decay'], max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type='cosine', optim='adamw_torch_fused' if use_cuda else 'adamw_torch',
        bf16=use_cuda, use_cpu=device == 'cpu', tf32=False, seed=config['seed'], data_seed=config['seed'],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=config['logging_steps'], logging_first_step=True, save_strategy='steps', save_steps=config['save_steps'],
        save_total_limit=3, save_only_model=False, save_safetensors=True, eval_strategy='steps', eval_steps=config['eval_steps'],
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda, remove_unused_columns=False,
        ignore_data_skip=False, report_to=[], push_to_hub=False)
    return LedgerTrainer(model=model, args=args, train_dataset=train_rows, eval_dataset=heldout_rows,
        data_collator=ChatCollator(tokenizer.pad_token_id if tokenizer else 0), processing_class=tokenizer,
        callbacks=[StopAt()], compute_loss_func=assistant_loss)


def offline_resume_smoke(directory):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, set_seed
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {'epochs': 1, 'per_device_batch_size': 2, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0002,
        'warmup_ratio': 0.0, 'weight_decay': 0.01, 'max_grad_norm': 1.0, 'seed': 7, 'logging_steps': 1,
        'save_steps': 2, 'eval_steps': 4}
    rows = [{'input_ids': [1, 4, 5, 20 + index, 7, 2, 9, 2], 'attention_mask': [1] * 8,
             'labels': [-100, -100, -100, -100, 7, 2, 9, 2]} for index in range(16)]
    def create_model():
        set_seed(config['seed'])
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            attention_dropout=0.0, pad_token_id=0, bos_token_id=1, eos_token_id=2))
        model.config.use_cache = False
        return get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=8, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    torch.set_num_threads(1)
    full = make_trainer(create_model(), None, rows, rows[:2], config, directory / 'full', 'offline-smoke-v1', 'cpu', max_steps=4)
    full.train()
    interrupted = make_trainer(create_model(), None, rows, rows[:2], config, directory / 'resumed', 'offline-smoke-v1', 'cpu', max_steps=4, stop_after=2)
    interrupted.train()
    checkpoint = directory / 'resumed' / 'checkpoint-2'
    validate_checkpoint(checkpoint, 'offline-smoke-v1')
    restored = make_trainer(create_model(), None, rows, rows[:2], config, directory / 'resumed', 'offline-smoke-v1', 'cpu', max_steps=4, resume=checkpoint)
    restored.train(resume_from_checkpoint=str(checkpoint))
    full_state, restored_state = full.model.state_dict(), restored.model.state_dict()
    difference = max(float((full_state[name] - restored_state[name]).abs().max()) for name in full_state)
    full_logs = [(row['step'], row['loss']) for row in full.state.log_history if 'loss' in row]
    resumed_logs = [(row['step'], row['loss']) for row in restored.state.log_history if 'loss' in row]
    if difference != 0 or full_logs != resumed_logs or full.input_tokens != restored.input_tokens or full.supervised_tokens != restored.supervised_tokens:
        raise AssertionError({'max_parameter_difference': difference, 'full_losses': full_logs, 'resumed_losses': resumed_logs})
    report = {'offline': True, 'global_steps': restored.state.global_step, 'max_parameter_difference': difference,
        'identical_logged_losses': full_logs == resumed_logs, 'input_tokens_seen': restored.input_tokens,
        'supervised_tokens_seen': restored.supervised_tokens, 'checkpoint_contains_optimizer_scheduler_rng_cursor': True,
        'scope': 'tiny CPU Transformers + PEFT resume parity, not a GPU or chat-quality test'}
    atomic_json(directory / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--data-dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--evaluation-suite')
    parser.add_argument('--offline-smoke', action='store_true')
    args = parser.parse_args()
    if args.offline_smoke:
        print(json.dumps(offline_resume_smoke(args.output), indent=2))
        return
    if not args.config or not args.data_dir or not args.evaluation_suite:
        parser.error('--config, --data-dir and --evaluation-suite are required for a real run')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from chat_eval import evaluate_generations, evaluate_targets, load_model
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not args.device.startswith('cuda') or torch.cuda.device_count() != 1:
        raise ValueError('This training recipe requires one visible CUDA GPU. Select it with CUDA_VISIBLE_DEVICES.')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('The selected GPU must support bf16')
    config = json.loads(Path(args.config).read_text())
    manifest = validate_prepared(config, args.data_dir)
    if manifest.get('evaluation_suite_file_sha256') != sha256(args.evaluation_suite):
        raise ValueError('Evaluation suite differs from the one excluded during preparation')
    train_rows = load_rows(Path(args.data_dir) / 'train.jsonl')
    heldout_rows = load_rows(Path(args.data_dir) / 'heldout.jsonl')
    output = Path(args.output)
    fingerprint = run_fingerprint(config, manifest, args.max_steps)
    if args.resume:
        validate_checkpoint(args.resume, fingerprint)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory or explicitly resume a complete checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'run_fingerprint': fingerprint, 'config': config, 'data': manifest, 'max_steps_override': args.max_steps,
        'libraries': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets')},
        'attribution': config['identity'], 'source_license': 'Apache-2.0', 'model_revision': MODEL_REVISION,
        'planned_token_budget': 'One pass over the prepared nonpadding input tokens. Assistant targets are separately counted.',
        'evaluation_suite_sha256': sha256(args.evaluation_suite)}
    atomic_json(output / 'run-manifest.json', provenance)
    set_seed(config['seed'])
    model, tokenizer = load_model(config, args.device)
    suite = json.loads(Path(args.evaluation_suite).read_text())
    if not args.resume:
        baseline = evaluate_generations(model, tokenizer, suite['cases'])
        atomic_json(output / 'baseline-generations.json', {'stage': 'untouched pretrained foundation', 'suite_sha256': sha256(args.evaluation_suite),
            'heldout': evaluate_targets(model, heldout_rows, tokenizer.pad_token_id), 'results': baseline})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], target_modules=config['lora_targets'], bias='none', revision=MODEL_REVISION))
    model.print_trainable_parameters()
    trainer = make_trainer(model, tokenizer, train_rows, heldout_rows, config, output, fingerprint, args.device,
        max_steps=args.max_steps, stop_after=args.stop_after, resume=args.resume)
    result = trainer.train(resume_from_checkpoint=args.resume)
    final_checkpoint = output / f'checkpoint-{trainer.state.global_step}'
    if not (final_checkpoint / 'COMPLETE.json').exists():
        trainer._save_checkpoint(trainer.model, trial=None)
    trainer.save_model(str(output / 'adapter'))
    tokenizer.save_pretrained(output / 'adapter')
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    after = evaluate_generations(model, tokenizer, suite['cases'])
    atomic_json(output / 'after-generations.json', {'stage': 'course LoRA adapter', 'suite_sha256': sha256(args.evaluation_suite),
        'heldout': evaluate_targets(model, heldout_rows, tokenizer.pad_token_id), 'results': after})
    report = {'training': result.metrics, 'evaluation': trainer.evaluate(), 'global_step': trainer.state.global_step,
        'input_tokens_seen': trainer.input_tokens, 'supervised_tokens_seen': trainer.supervised_tokens,
        'prepared_input_tokens': manifest['splits']['train']['input_tokens'], 'prepared_supervised_tokens': manifest['splits']['train']['supervised_tokens'],
        'final_checkpoint': final_checkpoint.name, 'adapter': 'adapter',
        'scope': 'LoRA adaptation of an existing pretrained instruction model; not pretraining from scratch. Generation records require human review. Trainer train_loss after resume may cover only the resumed span; compare the per-update logs and token-weighted held-out loss.'}
    atomic_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
