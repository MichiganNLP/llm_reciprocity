# Main reciprocity experiments

Only the main experiment runners are included. No robustness ablations, trace analyses, validation tools, plots, or earlier unrelated experiments are bundled.

| Entry point | Experiment |
|---|---|
| `regan.py` | Foundational Regan replication |
| `scenarios.py` | Eight-scenario generalization |
| `delegation.py` | Score-gap and price-premium tasks |
| `interaction.py` | Controlled dyadic and three-agent settings |
| `downstream.py` | PKU, HealthBench, and hiring choices |
| `check.py` | Offline syntax check and mock smoke test |

Shared implementations are in `_lib/`. There are six entry-point scripts, not six total Python files. Prompts and experimental logic are preserved; personal paths and cluster identifiers are removed. Fictional actor names, model IDs, and dataset names are retained for reproducibility.

## Usage

Use Python 3.10+ and run commands from this folder.

```bash
python -m pip install -r requirements.txt
python -B check.py
python regan.py --help
python scenarios.py --help
python delegation.py score-gap --help
python delegation.py price --help
python interaction.py dyad --help
python interaction.py committee --help
python downstream.py single-dyad --help
python downstream.py hiring --help
python downstream.py committee --help
```

```bash
python regan.py --backend huggingface --model Qwen/Qwen3-8B --hf-quantization 4bit --runs-per-cell 20 --output-dir results/regan/qwen3_8b
```

For PKU and HealthBench, `downstream.py single-dyad` accepts `--task pku` or `--task healthbench` and `--mode single` or `--mode dyadic`. Hiring uses `downstream.py hiring` with `--mode single` or `--mode dyadic`. The committee entry accepts all three tasks. Pass prepared JSONL manifests with `--input`; model downloads, data preparation, and API access require your own environment and permissions. Use `OPENAI_API_KEY` and, for gated models, `HF_TOKEN`; no credentials are included. Supply a model ID available to your account instead of relying on archived API aliases.

Default private storage paths are relocated beneath `data/`. Specify input, output, and cache paths explicitly for your environment. Runs write their raw records and summary files; no existing project results are bundled. No jobs are submitted automatically. Dependencies are not version-locked. This new folder does not modify either the original project code or the earlier code export.
