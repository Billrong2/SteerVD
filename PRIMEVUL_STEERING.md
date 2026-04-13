# PrimeVul Runner

This project keeps the PrimeVul baseline runner and the VulDeePecker-style
`code_gadget` extractor inside `Steer_VD`.

Current state:

- `primevul_eval.py` supports normal PrimeVul baseline evaluation only.
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

`primevul_eval.py` no longer exposes a steered run mode. The `code_gadget`
extractor remains available through `primevul_code_gadget_probe.py` while the
future inference-time steering path is developed separately.
