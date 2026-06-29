#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Streamlit dashboard for running and visualizing InFCD_MSDG experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization import pipeline


DISPLAY_FROM_PREFIX = {value: key for key, value in pipeline.STRATEGY_LABELS.items()}
DEFAULT_COLORS = {
    "MSDG": "#2563eb",
    "LMP": "#f97316",
    "LMU": "#16a34a",
    "SPC": "#9333ea",
}


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def result_file_for_input(input_file: Path, result_dir: Path) -> Path:
    return result_dir / f"result_{input_file.stem}.csv"


def input_file_for_result(result_file: Path, input_dir: Path) -> Path:
    stem = result_file.stem
    input_stem = stem.removeprefix("result_")
    return input_dir / f"{input_stem}.csv"


def show_command_result(result: pipeline.CommandResult) -> None:
    st.code(" ".join(result.command), language="bash")
    if result.stdout.strip():
        st.text_area("stdout", result.stdout, height=160)
    if result.stderr.strip():
        st.text_area("stderr", result.stderr, height=160)
    if result.ok:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")


def show_command_results(results: List[pipeline.CommandResult]) -> None:
    if not results:
        st.info("No commands were run.")
        return

    failed = [result for result in results if not result.ok]
    if failed:
        st.error(f"{len(failed)} of {len(results)} commands failed.")
    else:
        st.success(f"All {len(results)} commands completed successfully.")

    for idx, result in enumerate(results, start=1):
        status = "OK" if result.ok else f"FAILED ({result.returncode})"
        command_name = Path(result.command[0]).name if result.command else "command"
        with st.expander(f"{idx}. {command_name} - {status}", expanded=not result.ok):
            show_command_result(result)


def list_summary_files(summary_dir: Path) -> List[Path]:
    if not summary_dir.exists():
        return []
    return sorted(
        [
            path
            for path in summary_dir.glob("*.csv")
            if path.name.startswith("analysis_view_") or path.name.startswith("summary_")
        ]
    )


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def display_strategy_name(prefix: str) -> str:
    return DISPLAY_FROM_PREFIX.get(prefix, prefix)


def normalize_analysis_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strategy" in out.columns:
        out["strategy_label"] = out["strategy"].map(display_strategy_name)
    return out


def normalize_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in df.columns if col not in {"strategy", "metric"}]
    melted = df.melt(
        id_vars=["strategy", "metric"],
        value_vars=value_cols,
        var_name="x_value",
        value_name="value",
    )
    melted["strategy_label"] = melted["strategy"].map(display_strategy_name)
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    return melted


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in {"row_count"}
    ]


def render_chart(summary_dir: Path) -> None:
    files = list_summary_files(summary_dir)
    if not files:
        st.info("No summary CSV files found. Generate summary tables first.")
        return

    selected_file = st.selectbox(
        "Summary file",
        files,
        format_func=lambda path: path.name,
    )
    raw_df = load_csv(str(selected_file))
    st.caption(f"Loaded `{selected_file.name}` with {len(raw_df)} rows.")

    is_analysis_view = selected_file.name.startswith("analysis_view_")
    if is_analysis_view:
        chart_df = normalize_analysis_view(raw_df)
        available_strategies = sorted(chart_df["strategy_label"].dropna().unique().tolist())
        selected_strategies = st.multiselect(
            "Strategies",
            available_strategies,
            default=available_strategies,
        )
        chart_df = chart_df[chart_df["strategy_label"].isin(selected_strategies)]
        x_default = selected_file.stem.replace("analysis_view_", "")
        x_options = list(chart_df.columns)
        x_index = x_options.index(x_default) if x_default in x_options else 0
        x_col = st.selectbox("X axis", x_options, index=x_index)
        y_options = numeric_columns(chart_df)
        default_y = "profit" if "profit" in y_options else (y_options[0] if y_options else None)
        if not default_y:
            st.warning("No numeric columns are available for the Y axis.")
            return
        y_col = st.selectbox("Y axis", y_options, index=y_options.index(default_y))
        plot_df = chart_df.dropna(subset=[x_col, y_col])
    else:
        chart_df = normalize_summary_table(raw_df)
        metric_options = sorted(chart_df["metric"].dropna().unique().tolist())
        metric = st.selectbox(
            "Metric",
            metric_options,
            index=metric_options.index("profit") if "profit" in metric_options else 0,
        )
        chart_df = chart_df[chart_df["metric"] == metric]
        available_strategies = sorted(chart_df["strategy_label"].dropna().unique().tolist())
        selected_strategies = st.multiselect(
            "Strategies",
            available_strategies,
            default=available_strategies,
        )
        plot_df = chart_df[chart_df["strategy_label"].isin(selected_strategies)]
        x_col = "x_value"
        y_col = "value"

    color_map: Dict[str, str] = {}
    cols = st.columns(max(1, len(selected_strategies)))
    for idx, strategy in enumerate(selected_strategies):
        with cols[idx % len(cols)]:
            color_map[strategy] = st.color_picker(
                strategy,
                DEFAULT_COLORS.get(strategy, "#64748b"),
            )

    if plot_df.empty:
        st.warning("No rows match the selected chart settings.")
        return

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        color="strategy_label",
        barmode="group",
        color_discrete_map=color_map,
        labels={"strategy_label": "strategy"},
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        legend_title_text="Strategy",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_df, use_container_width=True)
    st.download_button(
        "Download displayed data",
        plot_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="infcmdsg_chart_data.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="InFCD_MSDG", layout="wide")
    st.title("InFCD_MSDG")
    st.caption("Sparse placement experiments and latency summaries for distributed LLM inference.")

    st.sidebar.header("Paths")
    run_dir = Path(st.sidebar.text_input("Run directory", path_text(pipeline.DEFAULT_RUN_DIR)))
    input_dir = Path(st.sidebar.text_input("Input directory", path_text(run_dir / "input")))
    result_dir = Path(st.sidebar.text_input("Result directory", path_text(run_dir / "result")))
    ttft_dir = Path(st.sidebar.text_input("TTFT result directory", path_text(run_dir / "results_ttft")))
    summary_dir = Path(st.sidebar.text_input("Summary directory", path_text(run_dir / "summary")))
    validation_dir = Path(st.sidebar.text_input("Validation directory", path_text(run_dir / "validation")))

    st.sidebar.header("Execution")
    selected_labels = st.sidebar.multiselect(
        "Strategies",
        list(pipeline.STRATEGY_LABELS.keys()),
        default=["MSDG", "LMP", "LMU", "SPC"],
    )
    selected_strategy_ids = [pipeline.STRATEGY_LABELS[label] for label in selected_labels]
    processes = st.sidebar.number_input("Processes", min_value=1, max_value=128, value=1, step=1)
    limit_raw = st.sidebar.number_input("Optional row limit", min_value=0, value=0, step=1)
    limit = int(limit_raw) if int(limit_raw) > 0 else None
    load_factors = st.sidebar.text_input("TTFT load factors", "1.0")
    ttft_slo = st.sidebar.text_input("TTFT SLO", "")
    tpot_slo = st.sidebar.text_input("TPOT SLO", "")

    tabs = st.tabs(["Pipeline", "Charts", "Validation"])

    with tabs[0]:
        st.subheader("Pipeline")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run placement experiment", type="primary"):
                input_files = pipeline.list_csv_files(input_dir)
                if not selected_strategy_ids:
                    st.error("Select at least one strategy before running the experiment.")
                elif not input_files:
                    st.error(f"No CSV input files were found in `{input_dir}`.")
                else:
                    result_dir.mkdir(parents=True, exist_ok=True)
                    progress = st.progress(0.0, text="Starting placement experiment...")
                    status = st.empty()
                    results: List[pipeline.CommandResult] = []
                    for idx, input_file in enumerate(input_files, start=1):
                        status.info(f"Running placement {idx}/{len(input_files)}: {input_file.name}")
                        result = pipeline.run_placement_experiment(
                            run_dir=run_dir,
                            input_dir=input_dir,
                            output_dir=result_dir,
                            strategies=selected_strategy_ids,
                            processes=int(processes),
                            limit=limit,
                            input_file=str(input_file),
                        )
                        results.append(result)
                        progress.progress(
                            idx / len(input_files),
                            text=f"Placement {idx}/{len(input_files)} completed: {input_file.name}",
                        )
                        if not result.ok:
                            status.error(f"Stopped after failure on `{input_file.name}`.")
                            break
                    else:
                        status.success("Placement experiment completed.")
                    show_command_results(results)
                    expected_files = [result_file_for_input(path, result_dir) for path in input_files]
                    present_count = sum(path.exists() for path in expected_files)
                    st.caption(
                        f"Expected result files: {present_count}/{len(expected_files)} present."
                    )
        with col2:
            if st.button("Add TTFT/TPOT metrics"):
                result_files = pipeline.list_csv_files(result_dir)
                if not result_files:
                    st.error(f"No result CSV files were found in `{result_dir}`.")
                else:
                    ttft_dir.mkdir(parents=True, exist_ok=True)
                    progress = st.progress(0.0, text="Starting TTFT/TPOT metric generation...")
                    status = st.empty()
                    results: List[pipeline.CommandResult] = []
                    for idx, result_file in enumerate(result_files, start=1):
                        input_file = input_file_for_result(result_file, input_dir)
                        if not input_file.exists():
                            status.error(f"Missing input file for `{result_file.name}`: `{input_file}`.")
                            break
                        output_file = ttft_dir / result_file.name
                        status.info(f"Adding TTFT/TPOT {idx}/{len(result_files)}: {result_file.name}")
                        result = pipeline.add_ttft_metrics_for_file(
                            input_file=input_file,
                            result_file=result_file,
                            output_file=output_file,
                            load_factors=load_factors,
                            ttft_slo=ttft_slo,
                            tpot_slo=tpot_slo,
                        )
                        results.append(result)
                        progress.progress(
                            idx / len(result_files),
                            text=f"TTFT/TPOT {idx}/{len(result_files)} completed: {result_file.name}",
                        )
                        if not result.ok:
                            status.error(f"Stopped after failure on `{result_file.name}`.")
                            break
                    else:
                        status.success("TTFT/TPOT metrics completed.")
                    show_command_results(results)
        with col3:
            if st.button("Generate summary tables"):
                progress = st.progress(0.0, text="Preparing summary generation...")
                status = st.empty()
                status.info("Generating summary tables from TTFT results.")
                progress.progress(0.2, text="Running summary command...")
                result = pipeline.generate_summary(
                    run_dir=run_dir,
                    result_dir=ttft_dir,
                    output_dir=summary_dir,
                )
                progress.progress(
                    1.0,
                    text="Summary generation completed." if result.ok else "Summary generation failed.",
                )
                if result.ok:
                    status.success("Summary tables generated.")
                else:
                    status.error("Summary generation failed.")
                show_command_result(result)

        st.write("Current directories")
        st.json({
            "run_dir": str(run_dir),
            "input_dir": str(input_dir),
            "result_dir": str(result_dir),
            "ttft_result_dir": str(ttft_dir),
            "summary_dir": str(summary_dir),
        })

    with tabs[1]:
        st.subheader("Summary charts")
        render_chart(summary_dir)

    with tabs[2]:
        st.subheader("Reference validation")
        reference_result_dir = Path(st.text_input(
            "Reference MSDG/LMU/SPC result directory",
            path_text(pipeline.DEFAULT_REFERENCE_RESULT_DIR),
        ))
        reference_lmp_dir = Path(st.text_input(
            "Reference LMP result directory",
            path_text(pipeline.DEFAULT_REFERENCE_LMP_RESULT_DIR),
        ))
        col1, col2, col3 = st.columns(3)
        with col1:
            atol = st.number_input("Absolute tolerance", value=1e-6, format="%.10f")
        with col2:
            rtol = st.number_input("Relative tolerance", value=1e-9, format="%.12f")
        with col3:
            row_limit_raw = st.number_input("Optional validation row limit", min_value=0, value=0, step=1)
        row_limit = int(row_limit_raw) if int(row_limit_raw) > 0 else None
        only_new_files = st.checkbox(
            "Validate only reproduced files",
            value=False,
            help="Use this for smoke checks. Leave unchecked for full strict validation.",
        )

        if st.button("Validate against reference results", type="primary"):
            progress = st.progress(0.0, text="Preparing validation...")
            status = st.empty()
            status.info("Comparing reproduced result files with reference outputs.")
            progress.progress(0.2, text="Running validation command...")
            result = pipeline.validate_against_reference(
                new_result_dir=result_dir,
                reference_result_dir=reference_result_dir,
                reference_lmp_result_dir=reference_lmp_dir,
                output_dir=validation_dir,
                atol=float(atol),
                rtol=float(rtol),
                row_limit=row_limit,
                only_new_files=only_new_files,
            )
            progress.progress(
                1.0,
                text="Validation completed." if result.ok else "Validation failed.",
            )
            if result.ok:
                status.success("Validation completed.")
            else:
                status.error("Validation failed.")
            show_command_result(result)

        summary_path = validation_dir / "validation_summary.json"
        report_path = validation_dir / "validation_report.csv"
        if summary_path.exists():
            st.write("Latest validation summary")
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))
        if report_path.exists():
            report_df = load_csv(str(report_path))
            st.dataframe(report_df, use_container_width=True)


if __name__ == "__main__":
    main()
