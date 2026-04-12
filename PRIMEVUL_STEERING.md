# PrimeVul Runner

This project keeps the PrimeVul runner, Qwen2.5 steering backend, sparse head
calibration, and the new VulDeePecker-style `code_gadget` extractor inside
`Steer_VD`.

Current state:

- `primevul_eval.py` supports normal PrimeVul baseline evaluation.
- `--prior` is now fixed to `code_gadget`.
- steered `code_gadget` runs are intentionally blocked until the final
  multi-gadget-to-prior reduction rule is defined.
- `primevul_code_gadget_probe.py` remains available for inspecting extracted
  gadgets on individual PrimeVul rows.

## Dataset

The official PrimeVul release lives under:

- `Source/`

The runner expects:

- a code field such as `func_before` or `func`
- a binary vulnerability label such as `target`
- an optional id field such as `idx`

## Preview Dataset Mapping

```bash
python3 primevul_eval.py \
  --dataset-path /home/cs/x/xxr230000/Steer_VD/Source/primevul_test.jsonl \
  --code-field func_before,func,code \
  --label-field target,vul,label \
  --id-field idx,id,commit_id \
  --language c \
  --preview-only \
  --preview-count 3
```

## Baseline Run

```bash
python3 primevul_eval.py \
  --dataset-path /home/cs/x/xxr230000/Steer_VD/Source/primevul_test.jsonl \
  --variant baseline \
  --prior code_gadget \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --language c \
  --limit 200 \
  --max-new-tokens 128
```

## Code Gadget Probe

```bash
python3 primevul_code_gadget_probe.py \
  --dataset-path /home/cs/x/xxr230000/Steer_VD/Source/primevul_test.jsonl \
  --row-index 0
```

## Temporary Steering Guard

This currently fails by design:

```bash
python3 primevul_eval.py \
  --dataset-path /home/cs/x/xxr230000/Steer_VD/Source/primevul_test.jsonl \
  --variant steered \
  --steer \
  --prior code_gadget
```

Reason:

- the new extractor can return multiple gadgets for one snippet
- the final rule that turns those gadgets into one prompt-aligned steering prior
  has not been defined yet

