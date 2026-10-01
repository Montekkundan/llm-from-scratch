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

Outputs refuse to overwrite existing runs or reports. Use a new output path for a new experiment. On the reference CPU run, base validation NLL reached approximately 0.2753. The small chat experiment learned all eight training answers, including raw `red\n` followed by EOS for `echo: red`, but scored **0/2 on held-out test words**. This demonstrates model/data/training/serialization mechanics and memorization; it does not demonstrate general language, copying, instruction following or reasoning. Keep this failed generalization result visible.

The model source is `course_model.py`; the shared byte IDs are in `tokenizer.py`. `train.py` owns document windows, masked targets, AdamW, scheduling and base export. `evaluate.py` verifies and reloads artifacts. `chat.py` owns the exact role-lines-v1 template and loss labels. `sft.py` adapts the base weights and records parent checksums. `generate.py` samples those weights. `checks.py` and `experiments.py` exercise the actual modules, rather than separate toy architectures.

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

## Hand the same artifact to the serving project

The base release contains `config.json`, `tokenizer.json`, `model.pt` and a checksum `manifest.json`. Chat adds `chat_template.json`, covered by the manifest. Its SFT report records the base-model checksum, so the relationship is inspectable. `resume.pt` is a separate training checkpoint containing optimizer and random-generator state. Only load artifacts you trust; checksums establish byte identity, not who supplied them.

Build the independent model wheel and copy it plus the selected release directory into the API project as described in the deployment course:

```sh
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir dist
```

The API, terminal and web clients all consume this one selected artifact. The 117,248-parameter mechanics weights are not a general-purpose assistant release. A learner's better-trained PicoLLM can replace them after passing the same artifact and endpoint checks.
