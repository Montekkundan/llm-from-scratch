# PicoLLM: one model, from bytes to a deployed artifact

## Get the project

```sh
git clone https://github.com/Montekkundan/llm-from-scratch.git
cd llm-from-scratch
```

Companion repositories: [llm-inference-api](https://github.com/Montekkundan/llm-inference-api) | [llm-terminal-chat](https://github.com/Montekkundan/llm-terminal-chat) | [llm-web-chat](https://github.com/Montekkundan/llm-web-chat). Clone companions into sibling directories when following the handoff instructions.

Build this original decoder, train it from random weights, add a verified inference cache, adapt the same weights to a conversation format, and hand the resulting artifact to the independent serving project. The source is self-contained; no code or weights come from the earlier monorepo. The full sequence of 30 lesson checkpoints is in `course-path.json`.

Python 3.11 and PyTorch **2.9.1** on CPU were used for verification. From this project directory, create an environment and install the model's only runtime dependency:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python checks.py all --output runs/checks/all.json
```

The independent wheel packages `course_model` only. Training and experiment scripts run from this source directory. `PicoLLM(ModelConfig())` is the primary API; `DecoderLM` remains an alias for previous imports, with identical state keys. The default model has 117,248 parameters, a 259-ID UTF-8 vocabulary, width 64, four heads, two layers and context 128. It uses adjacent-pair RoPE, pre-RMSNorm, SwiGLU and a tied input/output table.

## The core path

```sh
python train.py --output runs/picollm-base --steps 160
python evaluate.py runs/picollm-base
python generate.py runs/picollm-base --prompt 'red fox ' --max-new-tokens 24
python experiments.py cache --artifact runs/picollm-base --output runs/experiments/cache.json
python sft.py --base runs/picollm-base --output runs/picollm-chat --steps 160
python sft.py --evaluate runs/picollm-chat
python generate.py runs/picollm-chat --message 'echo: red' --max-new-tokens 12 --cached
```

Outputs refuse to overwrite existing runs or reports. Use a new output path for a new experiment. To repeat a command whose output exists, remove that path (the error message names it and the command, for example `rm -r runs/picollm-base`) or pass a different `--output`. On the reference CPU run, base validation NLL reached approximately 0.2753. The small chat experiment learned all eight training answers, including raw `red\n` followed by EOS for `echo: red`, but scored **0/2 on held-out test words**. This demonstrates model/data/training/serialization mechanics and memorization; it does not demonstrate general language, copying, instruction following or reasoning. Keep this failed generalization result visible.

The model source is `course_model.py`; the shared byte IDs are in `tokenizer.py`. `train.py` owns document windows, masked targets, AdamW, scheduling and base export. `evaluate.py` verifies and reloads artifacts. `chat.py` owns the exact role-lines-v1 template and loss labels. `sft.py` adapts the base weights and records parent checksums. `generate.py` decodes those weights greedily (the default), by temperature sampling, or with optional `--top-k` and `--top-p` filters that are off unless given. `checks.py` and `experiments.py` exercise the actual modules, rather than separate toy architectures.

### What the code contains, and what it does not

- Normalization is RMSNorm only (`RMSNorm` in `course_model.py`): features are rescaled by their root mean square with a learned gain, with no mean subtraction and no bias. There is no LayerNorm in this repository, although lesson 9 is titled layer normalization.
- Attention is causal multi-head attention with RoPE on queries and keys and the scale 1/sqrt(head width). `gpu_model.py` computes the same function with PyTorch's `scaled_dot_product_attention`. `test_model.py` checks the scale, the mask and RoPE against independent references. There is no grouped-query attention, sliding window or mixture of experts here.
- Generation is greedy by default, with temperature sampling and optional top-k and top-p (nucleus) filtering in `generate.py`, for example `python generate.py runs/picollm-base --prompt 'red fox ' --temperature 0.8 --top-k 40 --top-p 0.9`. The filters apply after temperature and are ignored when the temperature is 0. There is no beam search or repetition penalty.
- `train.py` takes one direct minibatch per update and has no gradient accumulation. `checks.py loop` and `experiments.py distributed` check only the loss-denominator algebra that accumulation and data parallelism rely on, and launch no process group. `gpu_train.py` and `scratch_sft.py` accumulate gradients (`grad_accum`); every trainer runs on one device.
- LoRA exists only in the 1.7B foundation experiment: `chat_finetune.py` trains the adapter, and `chat_eval.py` and `checkpoint_probe.py` load it. PicoLLM's own `sft.py` and `scratch_sft.py` update all parameters.
- Not implemented: finite-difference checks of PicoLLM's gradients (the only `gradcheck` is in `test_gpu_train.py`, on PyTorch attention), roofline or arithmetic-intensity analysis, quantization with zero points or of the KV cache (`quantize` is symmetric per-tensor int8 for weights, run in FP32), fitted scaling laws (`scaling` counts parameters and applies 6ND), and rubric scores or confidence intervals (the echo task scores exact match on two held-out words).

Run the tests with `python -m unittest discover -p 'test_*.py'` from this directory. They write only to temporary directories and do not depend on where the repository is cloned.

### Constrained learned FAQ demo

`data/faq.json` contains original teaching examples: 15 training conversations, three validation prompts and four test prompts with disjoint wording. This is a constrained learned FAQ, not a general chatbot. The training prompts include `hi how are you?` and questions about tokens, attention and training. Answers come from model logits at generation time; there is no prompt-to-answer lookup.

```sh
python sft.py --base runs/picollm-base --data data/faq.json --output runs/faq-demo --steps 600 --max-new-tokens 64
python sft.py --evaluate runs/faq-demo
python generate.py runs/faq-demo --message 'hi how are you?' --max-new-tokens 64 --cached
python generate.py runs/faq-demo --message 'what is a token?' --max-new-tokens 64 --cached
python generate.py runs/faq-demo --message 'how does attention work?' --max-new-tokens 64 --cached
python -m unittest test_sft -v
```

The 600-update budget is fixed before examining held-out results. The reference CPU run learned **15/15 training answers**, scored **1/3 validation answers** and **0/4 test answers**, and spent approximately 5.75 seconds on updates. Inspect every split's generated text and exact-match score in `sft-report.json`; correct answers to trained prompts establish memorization, while unseen wording can fail. The report preserves the dataset, source and base-weight checksums, training history and generated token IDs. `--evaluate` verifies the saved data and reproduces the test outputs after loading the artifact.

For another conversation set, `--data` accepts a JSON array with `id`, `split` (`train`, `validation`, or `test`) and `messages` containing text `role`/`content` pairs ending with an assistant answer. IDs must be unique and all three splits must be nonempty. The loader rejects duplicate conversations and prompts or declared `group` values shared across splits. Near duplicates require review. The optional generation limit defaults to 64 byte tokens for custom data and 12 for the original echo exercise; prompts plus the requested limit must fit the model context. No truncation is implicit.

## Train on your own document corpus

The synthetic default is a quick diagnostic. For a substantial language-model experiment, prepare separate UTF-8 document directories you are authorized to train on. Assign all documents from the same source group to one split **before** windowing. Use a top-level subdirectory for a related group, such as `train/book-a/chapter1.txt`; do not put `book-a` in validation or test too. The converter uses that top-level directory as the group, or the filename stem for standalone files. Review near duplicates yourself; the code only detects exact content duplicates and declared group overlap. Keep the test split untouched while choosing the architecture, learning rate and stopping rule on training/validation.

```sh
python prepare_data.py --train-dir documents/train --validation-dir documents/validation --test-dir documents/test --output data/my-corpus --source 'My original document collection, revision 1' --license 'Original writing owned by me'
python train.py --train-file data/my-corpus/train.jsonl --validation-file data/my-corpus/validation.jsonl --test-file data/my-corpus/test.jsonl --output runs/picollm-text --steps 160 --width 64 --heads 4 --layers 2 --context 128 --ff-width 176
python evaluate.py runs/picollm-text
python evaluate.py runs/picollm-text --split test
python generate.py runs/picollm-text --prompt 'The ' --max-new-tokens 24
```

Replace the provenance and ownership strings with the actual source information. JSONL can also be supplied directly: each row needs a unique string `id` and nonempty `text`; `group`, `source`, and `license` preserve the declared provenance. The loader rejects duplicate text and groups shared across all supplied splits. It retains source records in `corpus.json`. The third split is optional for compatibility with the earlier two-split exercise, but required for a final held-out result. Training never scores its test documents. Run `evaluate.py --split test` only after fixing the experiment design. The evaluator first verifies the saved validation result, then reports test NLL and bits per byte alongside an add-one byte/EOS unigram baseline fitted only to the training documents. Compare bits per byte on the same byte corpus; a toy model beating a context-free baseline does not establish useful language generation.

Long documents are windowed at the token level. Every byte/EOS target is counted once. Each new window retains its immediately preceding token as input, resets positions to zero and discards older context; no window crosses a document. It does not fabricate EOS at chunk boundaries. The implementation loads the selected corpus into memory and retokenizes documents for this explicit reference pipeline; it is suitable for bounded teaching subsets, not a trillion-token data loader. Larger research work needs a versioned indexed dataset and measured throughput.

The author-maintained [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories) is one optional source of longer narrative text, with its own declared train/validation splits and terms. It is model-generated narrative data, not human-authored general web text. Preserve the source revision, original split, story boundaries and terms when converting a bounded subset. No external dataset is downloaded or included here. The [paper](https://arxiv.org/abs/2305.07759) provides the actual study and evaluation scope; our tiny mechanics run does not reproduce its results.

Configuration flags also support a larger model, for example `--width 256 --heads 8 --layers 4 --context 512 --ff-width 704`, but that is a new experimental configuration and a much more expensive CPU run. No quality or GPU performance for it is claimed. The resulting config stays with its weights. The byte tokenizer contract and model class remain the same.

## Verify continuation and optional systems experiments

```sh
python train.py --output runs/full40 --steps 40
python train.py --output runs/part40 --steps 40 --stop-after 20
python train.py --output runs/resume40 --steps 40 --resume runs/part40/resume.pt
python experiments.py profile --artifact runs/picollm-base --output runs/experiments/profile.json
python experiments.py quantize --artifact runs/picollm-base --output runs/experiments/quantize.json
python experiments.py scaling --artifact runs/picollm-base --output runs/experiments/scaling.json
python experiments.py distributed --artifact runs/picollm-base --output runs/experiments/distributed.json
python experiments.py recompute --artifact runs/picollm-base --output runs/experiments/recompute.json
python experiments.py ablation-plan --artifact runs/picollm-base --output runs/experiments/ablation-plan.json
```

For continuation, compare final state-dictionary tensors and `history.json`; keep the total schedule budget fixed. Exact CPU continuation was checked in the same runtime. Cross-device/version bit identity is not promised. `profile` records bounded CPU timing; `quantize` is a round/dequantize error probe with FP32 storage; `distributed` proves gradient-weighting algebra without launching a process group; `recompute` checks actual checkpointed block gradients; `scaling` and `ablation-plan` produce research plans, not unrun research conclusions.

## GPU experiments and progress checks

The CPU path above teaches the mechanics. These separate experiments are configured (`configs/pico-125m.json`, `configs/pico-400m.json`) to train the same decoder at 122,702,592 or 404,804,608 parameters on natural English web documents, then tune it for conversations. A configuration is a plan; the `run-report.json` and `metrics.jsonl` in each output directory record what was actually completed. Web pretraining learns document continuation; it does not by itself teach a chat protocol. The 1.7B experiment instead adapts an existing SmolLM2 foundation with LoRA. Its original pretraining is credited to Hugging Face.

Run commands from this source directory. Install the optional pinned packages with `python -m pip install -r requirements-gpu.txt`. CPU and MPS checks used Python 3.11 and PyTorch 2.9.1. A100 checks used Python 3.12 and PyTorch 2.14.0+cu130 through a read-only university CUDA environment, with the experiment's own package overlay. The repository's PyTorch 2.9.1 CUDA wheel has not been GPU-tested in that environment.

Prepare the pinned FineWeb-Edu `sample-10BT` source and save its tokenizer:

```sh
python prepare_corpus.py --output data/fineweb-edu --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 --tokenizer-revision 31b70e2e869a7173562077fd711b654946d38674 --max-train-tokens 8000110593 --dtype u16
```

The target is 8,000,110,592 training tokens plus the next-token lookahead. Counts use the pinned SmolLM2 tokenizer, not the GPT-2 token count in the dataset's sample name. The preparer commits complete, hashed token shards to `manifest.json`. Restarting the same command verifies and resumes preparation. In separate terminals or GPU hosts, run one trainer per GPU:

```sh
python run_pretraining.py --config configs/pico-125m.json --data data/fineweb-edu/manifest.json --output runs/pico-125m --device cuda --save-every 300s
python run_pretraining.py --config configs/pico-400m.json --data data/fineweb-edu/manifest.json --output runs/pico-400m --device cuda --save-every 300s
```

The 125M configuration consumes 2,000,027,648 tokens; the 400M configuration consumes 8,000,110,592. `run_pretraining.py` waits when preparation has not committed enough data, then resumes from the last complete checkpoint as new shards arrive. Keep committed shards and the training configuration immutable. Only append verified shards to the same manifest. To resume a paused trainer directly:

```sh
python gpu_train.py --config configs/pico-125m.json --data data/fineweb-edu/manifest.json --output runs/pico-125m --device cuda --resume latest --save-every 300s
```

After pretraining, prepare scratch-specific conversation targets and start full-parameter SFT from the selected base checkpoint:

```sh
python chat_data.py --config configs/chat-scratch-125m.json --output data/chat-scratch-125m --evaluation-suite evaluation/chat-evaluation.json
python scratch_sft.py --config configs/chat-scratch-125m.json --base-checkpoint runs/pico-125m --data data/chat-scratch-125m --output runs/pico-125m-chat --device cuda --save-every 300s
python scratch_sft.py --config configs/chat-scratch-125m.json --data data/chat-scratch-125m --output runs/pico-125m-chat --device cuda --resume latest --save-every 300s
python scratch_chat.py --checkpoint runs/pico-125m-chat --tokenizer data/fineweb-edu/tokenizer --device cuda --dtype bfloat16 --interactive
python scratch_chat.py --checkpoint runs/pico-125m-chat --tokenizer data/fineweb-edu/tokenizer --device cuda --dtype bfloat16 --suite evaluation/chat-evaluation.json --output runs/pico-125m-chat/generations.json
```

The third command continues an interrupted SFT run; it does not start another fresh adaptation. For the 400M path, use `configs/chat-scratch-400m.json`, base `runs/pico-400m`, data `data/chat-scratch-400m`, and output `runs/pico-400m-chat`. `scratch_chat.py` loads the saved pretraining tokenizer and verifies its fingerprint. It generates responses from the selected weights. Scratch identity examples credit Montek Singh Kundan with creating the course model and distinguish the borrowed tokenizer from pretrained model weights.

The separate foundation experiment uses the pinned SmolLM2-1.7B-Instruct weights and SmolTalk conversations:

```sh
python chat_data.py --config configs/chat-1.7b.json --output data/chat-foundation --evaluation-suite evaluation/chat-evaluation.json
python chat_finetune.py --config configs/chat-1.7b.json --data-dir data/chat-foundation --output runs/chat-1.7b --device cuda --evaluation-suite evaluation/chat-evaluation.json
python chat_eval.py --config configs/chat-1.7b.json --adapter runs/chat-1.7b/adapter --device cuda --interactive
python chat_eval.py --config configs/chat-1.7b.json --adapter runs/chat-1.7b/adapter --device cuda --suite evaluation/chat-evaluation.json --output runs/chat-1.7b/generations.json
```

This preparation produced 10,000,106 nonpadding input tokens and 7,312,756 supervised assistant tokens. The declared 10M budget includes system and user text; it is not 10M training targets. Labels mask system text, user text, headers and padding, while retaining assistant content and its EOS. A conversation longer than the context is cut at the context length and nothing is added at the cut, so `<|im_end|>` is supervised only where an assistant turn really ends; data prepared before this change forced that label at the cut (including for scratch SFT) and should be prepared again. The tokenizer, source revision, example order, template, data hashes and actual token ledger are recorded. The frozen evaluation prompts are excluded from our SFT data; this does not establish exclusion from the foundation's historical pretraining. The runner captures untouched baseline and adapted generations. Identity scores under the course system measure instruction adherence; the additional neutral-system identity prompts check what the weights learned.

Check preparation and training progress after their logs appear:

```sh
tail -n 3 data/fineweb-edu/progress.jsonl
cat runs/pico-125m/runner.json
tail -n 3 runs/pico-125m/metrics.jsonl
cat runs/pico-125m/run-report.json
```

Sample an immutable completed checkpoint between training milestones with `checkpoint_probe.py`. Its fixed development prompts are distinct from `evaluation/chat-evaluation.json`; the frozen suite is read only to reject prompt overlap. Select a complete directory once, copy it to a machine that is not training, and probe that copy. Do not read a directory while it is being copied or rely on a live pointer that checkpoint retention can replace.

For pretraining, resolve `latest.json` once after the copy finishes:

```sh
mkdir -p runs/probes
pico_checkpoint="$(python -c 'import json; from pathlib import Path; root=Path("runs/pico-125m"); print(root / json.loads((root / "latest.json").read_text())["checkpoint"])')"
python checkpoint_probe.py --checkpoint "$pico_checkpoint" --tokenizer data/fineweb-edu/tokenizer --kind pretrain --device mps --dtype float32 --max-new-tokens 32 --metrics runs/pico-125m/metrics.jsonl --output runs/probes/pico-125m-pretrain.json
```

The 400M command uses the corresponding `runs/pico-400m` paths. Scratch SFT uses the same saved tokenizer and generates chat responses:

```sh
pico_chat_checkpoint="$(python -c 'import json; from pathlib import Path; root=Path("runs/pico-125m-chat"); print(root / json.loads((root / "latest.json").read_text())["checkpoint"])')"
python checkpoint_probe.py --checkpoint "$pico_chat_checkpoint" --tokenizer data/fineweb-edu/tokenizer --kind scratch_sft --device mps --dtype float32 --max-new-tokens 32 --metrics runs/pico-125m-chat/metrics.jsonl --output runs/probes/pico-125m-chat.json
```

For a LoRA checkpoint, use its saved tokenizer files and set `HF_HOME` to the existing cache containing the exact pinned foundation weights. This command loads cached files only and fails if the weights are absent:

```sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python checkpoint_probe.py --checkpoint runs/chat-1.7b/checkpoint-100 --tokenizer runs/chat-1.7b/checkpoint-100 --kind lora --config configs/chat-1.7b.json --device mps --max-new-tokens 32 --output runs/probes/chat-1.7b-checkpoint-100.json
```

Use `--device cpu --dtype float32` for scratch CPU probes or `--device cuda --dtype bfloat16` on a separate available CUDA GPU. LoRA chooses FP32 on CPU/MPS and BF16 on CUDA. `--frozen-suite PATH` overrides the source-bundled exclusion list; `--metrics PATH` optionally attaches the most recent finite losses at or before the selected checkpoint step. Each JSON report records checkpoint SHA, step, consumed tokens, exact generated IDs, unedited text, finish reason and repetition/length signals. These signals help spot training failures; they do not establish answer quality or final evaluation success.

A complete checkpoint retains model weights, optimizer, scheduler, random state, data cursor, token counts and resolved configuration. Pretraining and scratch SFT publish `complete.json` and update `latest.json` atomically; foundation SFT publishes a hashed `COMPLETE.json` after all resume files exist. An adapter alone is an inference artifact, not a full training checkpoint. Copy only complete checkpoints, verify their hashes, retain their data/configuration identities, and keep the total schedule fixed on resume.

CPU continuation and continuation in the same tested CUDA runtime matched uninterrupted training exactly. A CUDA checkpoint taken at update 2 continued on MPS through update 4 with maximum parameter difference approximately 1.88e-4 against the reference. That verifies bounded portability, not bit identity across devices or PyTorch versions.

Three setup failures are recorded: `EDQUOT` was a user quota failure despite free filesystem space, the macOS runtime required an APFS environment instead of the ExFAT location, and the Hub `RepoFile` import was corrected to `huggingface_hub.hf_api`. These are environment and preparation checks, not model-quality results. Falling loss, completed updates and successful resume do not establish a useful assistant. Review the full held-out responses, including failures, before making a quality claim.

## Hand the same artifact to the serving project

The base release contains `config.json`, `tokenizer.json`, `model.pt` and a checksum `manifest.json`. Chat adds `chat_template.json`, covered by the manifest. Its SFT report records the base-model checksum, so the relationship is inspectable. `resume.pt` is a separate training checkpoint containing optimizer and random-generator state. Only load artifacts you trust; checksums establish byte identity, not who supplied them.

Build the independent model wheel and copy it plus the selected release directory into the API project as described in the deployment course:

```sh
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir dist
```

The API, terminal and web clients all consume this one selected artifact. The 117,248-parameter mechanics weights are not a general-purpose assistant release. A learner's better-trained PicoLLM can replace them after passing the same artifact and endpoint checks.
