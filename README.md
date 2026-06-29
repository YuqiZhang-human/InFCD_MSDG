# InFCD_MSDG

InFCD_MSDG is an experimental framework for evaluating sparse placement
strategies for distributed LLM inference. It maps model segments onto GPU
nodes in a real-network topology, estimates deployment cost and served users,
and post-processes each placement result with TTFT/TPOT latency metrics.

The repository includes runnable experiment entry points, sparse optimizer
implementations, latency analysis, summary-table generation, a Streamlit
visualization dashboard, and a bundled example run for checking the workflow.

## Contents

- `src/experiment/run_sparse_experiment.py`: main sparse experiment entry point.
- `src/experiment/run_mix_chain_sparse.py`: mix-chain experiment entry point.
- `src/algorithm/single_chain_real_net/`: sparse optimizer implementations.
- `data/src/add_ttft_to_results_sparse.py`: TTFT/TPOT post-processing.
- `data/src/analysis/summarize_results_sparse.py`: result aggregation and summary table generation.
- `data/runs/optimal_solutions_compare_module/input/input_0000.csv`: bundled example input.
- `data/runs/optimal_solutions_compare_module/result/`: bundled example placement results.
- `data/runs/optimal_solutions_compare_module/results_ttft/`: bundled example latency-augmented results.
- `data/runs/optimal_solutions_compare_module/summary/`: bundled example summary tables.
- `visualization/app.py`: local dashboard for running experiments and plotting summaries.
- `scripts/run_sample.py`: smoke workflow that writes temporary outputs without overwriting bundled results.

## Strategies

The main entry point supports these strategy names:

- `multi_max_profit`: multi-all-node sparse deployment optimized for maximum profit.
- `lmp`: local module placement baseline.
- `multi_max_user`: multi-all-node sparse deployment optimized for maximum served users.
- `shortest_path`: shortest-path sparse deployment baseline.
- `brute`: exhaustive brute-force sparse deployment optimized for maximum profit.

The bundled example and the dashboard focus on `multi_max_profit`, `lmp`,
`multi_max_user`, and `shortest_path`. The mix-chain entry point supports the
same strategies with a `mix_` prefix, such as `mix_multi_max_profit` and
`mix_shortest_path`.

## Environment

Python 3.10 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Bundled Example Data

The default example run is:

```text
data/runs/optimal_solutions_compare_module
```

It contains one input CSV, reproduced placement results, TTFT/TPOT-augmented
results, and summary tables. This lets the dashboard show charts immediately
after the repository is cloned.

## Run a Smoke Workflow

From the repository root:

```bash
python scripts/run_sample.py
```

The smoke script runs the four dashboard strategies on the bundled input with
`--limit 1`, adds TTFT/TPOT metrics, and writes temporary outputs to ignored
directories:

- `data/runs/optimal_solutions_compare_module/smoke_result/`
- `data/runs/optimal_solutions_compare_module/smoke_results_ttft/`
- `data/runs/optimal_solutions_compare_module/smoke_summary/`

The tracked example outputs under `result/`, `results_ttft/`, and `summary/`
are not overwritten by the smoke script.

## Reproduce the Example Results

To regenerate the bundled result files from the example input, run:

```bash
python -m src.experiment.run_sparse_experiment \
  --run_dir data/runs/optimal_solutions_compare_module \
  --strategies multi_max_profit,lmp,multi_max_user,shortest_path \
  --processes 1
```

Then add TTFT/TPOT metrics:

```bash
python -m data.src.add_ttft_to_results_sparse \
  --input_dir data/runs/optimal_solutions_compare_module/input \
  --result_dir data/runs/optimal_solutions_compare_module/result \
  --output_dir data/runs/optimal_solutions_compare_module/results_ttft \
  --load_factors 1.0
```

Finally, regenerate summary tables:

```bash
python -m data.src.analysis.summarize_results_sparse \
  --run_dir data/runs/optimal_solutions_compare_module
```

The summary script detects strategy prefixes from result columns ending in
`_deployment_list`, so it works for both the main and mix-chain entries.

## Visualization Dashboard

Start the local dashboard from the repository root:

```bash
streamlit run visualization/app.py
```

The dashboard defaults to `data/runs/optimal_solutions_compare_module`. It can
run placement experiments, add TTFT/TPOT metrics, generate summary tables, plot
grouped bar charts, choose strategy colors, and validate reproduced results
against a reference experiment.

![Pipeline tab](pic/pipeline.png)

![Visualization tab](pic/visualization.png)

![Validation tab](pic/validation.png)

## Use a Custom Input

To run a larger experiment, place generated input CSV files under a run
directory such as:

```text
data/runs/my_run/input/
```

Then run:

```bash
python -m src.experiment.run_sparse_experiment \
  --run_dir data/runs/my_run \
  --strategies multi_max_profit,lmp,multi_max_user,shortest_path \
  --processes 1

python -m data.src.add_ttft_to_results_sparse \
  --input_dir data/runs/my_run/input \
  --result_dir data/runs/my_run/result \
  --output_dir data/runs/my_run/results_ttft \
  --load_factors 1.0

python -m data.src.analysis.summarize_results_sparse \
  --run_dir data/runs/my_run
```

For large inputs, increase `--processes` according to the available CPU and
memory budget. The exhaustive brute-force strategy is exponential in the number
of modules and candidate GPU nodes, so it should be used only on appropriately
small cases.
