# MMMem experiment guide

Run commands from the repository root. Set an external workspace for data, model snapshots, caches, training artifacts, and predictions.

```bash
export MMMEM_WORKSPACE="$HOME/mmmem-workspace"
mkdir -p "$MMMEM_WORKSPACE"
```

The [quick start](README.md#quick-start) uses CPU dependencies. For model-backed runs, install `requirements-models.txt` in a CUDA environment with BF16 support. The [GPU quick start](README.md#gpu-quick-start) runs the synthetic example through the model-backed entry point. The model requirements pin the Transformers revision used by the GPU workflow.

## Data and models

Use the upstream benchmark releases below. Keep the scorer's exported labels under `$MMMEM_WORKSPACE/private_eval/`, outside the input directory passed to the runner.

### LoCoMo

Use the [LoCoMo code and data](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376). The source views use the released captions. Category 5 retains the two answer choices shown to the reader.

```bash
python scripts/prepare_memorium_u04_locomo_cell.py \
  --locomo10 "$MMMEM_WORKSPACE"/upstream/locomo/data/locomo10.json \
  --output "$MMMEM_WORKSPACE"/data/locomo/cell --gold-output "$MMMEM_WORKSPACE"/private_eval/locomo.jsonl
```

### Mem-Gallery

Use the [dataset snapshot](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/tree/af912daba984e896e253016b7c7e334ef92c2a6f) and [benchmark code](https://github.com/YuanchenBei/Mem-Gallery/tree/a93959e1e978a6a7d77798ae92c2ffe41c538c62).

```bash
python scripts/prepare_memorium_u05_gallery_cell.py \
  --release "$MMMEM_WORKSPACE"/upstream/gallery-data/data \
  --prompts "$MMMEM_WORKSPACE"/upstream/gallery-code/benchmark/prompt \
  --output "$MMMEM_WORKSPACE"/data/gallery/cell --gold-output "$MMMEM_WORKSPACE"/private_eval/gallery.jsonl
```

### LooGLE v2

Download the four Parquet shards from the [LooGLE v2 dataset revision](https://huggingface.co/datasets/MuLabPKU/LooGLE-v2/tree/1b83bd4b08a3faef45168d7cc1ea8aca785f3d9c). The preparer splits each document into 8,000-character chunks and packs each chunk into paragraphs of at most 1,500 characters.

```bash
python -m pip install pyarrow
python scripts/prepare_memorium_u05_loogle_cell.py \
  --parquet "$MMMEM_WORKSPACE"/upstream/loogle/data/test-*.parquet \
  --output "$MMMEM_WORKSPACE"/data/loogle/cell --gold-output "$MMMEM_WORKSPACE"/private_eval/loogle.jsonl
```

## Generate tasks and train the reader

[Training splits](configs/training.json) specify which source units contribute to synthetic QA generation. The DEV and TEST units are held out by source. The optional few-shot examples for generation use question wording from GEN units.

Set `MODEL` to a local Qwen3.5-9B or MiniCPM-V 4.5 snapshot. Set `BACKEND` to `qwen35` or `minicpmv`. The following examples show a fresh run with one task recipe per benchmark.

```bash
for domain in locomo gallery loogle; do
  python scripts/run_memorium_joint_rsi.py \
    --domain "$domain" --dev-split configs/training.json \
    --model "$MODEL" --reader-backend "$BACKEND" \
    --recipe sequence --output "$MMMEM_WORKSPACE/runs/tasks/$domain"
done

python scripts/train_memorium_joint_reader.py \
  --dev-split configs/training.json --model "$MODEL" --reader-backend "$BACKEND" \
  --data locomo="$MMMEM_WORKSPACE"/runs/tasks/locomo/rsi.jsonl \
         gallery="$MMMEM_WORKSPACE"/runs/tasks/gallery/rsi.jsonl \
         loogle="$MMMEM_WORKSPACE"/runs/tasks/loogle/rsi.jsonl \
  --rank 16 --lr 1e-4 --accum 8 --output "$MMMEM_WORKSPACE"/runs/lora
```

For warm-up, generate a task file per benchmark and reader. The generator uses that TEST unit's own source records. The evaluation runner plays those synthetic questions before the official tasks, with fixed weights. It discards the generator's reference answers and evidence fields when building the warm-up requests.

```bash
DOMAIN=gallery
python scripts/run_memorium_joint_rsi.py \
  --domain "$DOMAIN" --dev-split configs/training.json \
  --model "$MODEL" --reader-backend "$BACKEND" \
  --warmup test --recipe sequence --output "$MMMEM_WORKSPACE/runs/warmup/$DOMAIN"
```

The generator writes `rsi.jsonl`, and training writes an adapter and a training report under the chosen output directory.

## Evaluate

Set paths to local model snapshots. The runner resolves model defaults under `$MMMEM_WORKSPACE/models/` (by default, `$HOME/mmmem-workspace/models/`). To override them, use `--text-encoder`, `--visual-encoder`, and `--image-model`. The visual encoder loads the upstream `scripts/qwen3_vl_embedding.py` from its model directory. Rendered pages use DejaVu Sans and Fontconfig. Use `VMC_CANVAS_FONT` to point to a local TrueType font when needed.

This example runs Mem-Gallery with `BACKEND`, `MODEL`, `LORA`, and `RERANKER` pointing to the chosen backend and local weights. Use the base Qwen3-Reranker-0.6B for the reference reranking setup.

```bash
python scripts/run_memorium_u05_cell.py \
  --cell "$MMMEM_WORKSPACE"/data/gallery/cell --split configs/gallery.json --part test \
  --reader-model "$MODEL" --reader-backend "$BACKEND" \
  --arm F_v2 --reader-adapter "$LORA" --reranker "$RERANKER" \
  --routes TV --token-budget 4000 --reader-memory text \
  --experience on --exp-topk 1 --exp-outside \
  --exp-promote-uses 2 --exp-promote-families 1 --abstain-edit \
  --exp-admission on --preserve-query-image-rank \
  --warmup-tasks "$MMMEM_WORKSPACE"/runs/warmup/gallery/rsi.jsonl --output "$MMMEM_WORKSPACE"/runs/gallery-test
```

| Benchmark | Source token budget | Benchmark-specific arguments | TEST questions |
| --- | ---: | --- | ---: |
| LoCoMo | 3,000 | `--routes TV` | 1,178 |
| Mem-Gallery | 4,000 | `--routes TV --exp-admission on --preserve-query-image-rank` | 1,116 |
| LooGLE v2 | 4,000 | `--routes T --prompt-mode loogle --v1-level off --v2-unit-budget 4` | 1,164 |

The retrieved experience occupies a separate reader slot outside the source evidence budget. Use `--routes TV` for text and visual retrieval. `--read-generated-views` additionally sends selected generated scenes to the reader within the evidence budget. Without that option, generated scenes serve the visual index and the reader receives the selected text and original images. The table describes the reference benchmark configurations. `rows.jsonl` contains the scored-task predictions and source IDs. `warmup_rows.jsonl` contains the earlier unscored tasks. `run_manifest.json` records the arguments and input hashes.

For Frozen MMMem use `F_v2_zero` and omit the adapter. For LoRA with no generated scene views use `F_lite`. For no warm-up omit `--warmup-tasks`. For no experience set `--experience off` and omit experience-specific options and warm-up. `A0` and `A0_lora` run the dense retrieval controls with `--routes T --v1-level off` and no experience. Use the same reader, prompt mode, split, and source budget in matched comparisons.

## Score

Use `scripts/score_memorium_u04_locomo_cell.py`, `scripts/score_memorium_u05_gallery_cell.py`, or `scripts/score_memorium_u05_loogle_cell.py`. Their `--help` lists the required paths. Use the corresponding benchmark evaluator for LoCoMo and Mem-Gallery. Install `nltk` and `bert-score` to load the LoCoMo evaluator. The Mem-Gallery evaluator also imports `openai` and `tqdm`. Scoring uses its local EM, F1, and BLEU-1 functions. Pass `benchmark/memengine/evaluate/evaluation.py` from the code snapshot above.

```bash
UNITS=$(python -c 'import json; print(",".join(json.load(open("configs/gallery.json"))["test"]))')
python scripts/score_memorium_u05_gallery_cell.py \
  --run-root "$MMMEM_WORKSPACE"/runs/gallery-test --gold "$MMMEM_WORKSPACE"/private_eval/gallery.jsonl \
  --evaluator "$MMMEM_WORKSPACE"/upstream/gallery-code/benchmark/memengine/evaluate/evaluation.py \
  --units "$UNITS" --output "$MMMEM_WORKSPACE"/runs/gallery-score
```
