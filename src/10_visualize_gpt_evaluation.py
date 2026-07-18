"""
GPT 분류 성능평가 결과 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_summary.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

OUTPUT_FIGURE_PATH = (
    FIGURE_DIR
    / "figure_06_overall_metrics.png"
)

OUTPUT_DATA_PATH = (
    FIGURE_DIR
    / "gpt_evaluation_plot_data.csv"
)

# =====================================================================
# 2. 그래프 라벨 설정
# =====================================================================
TASK_ORDER = [
    "sentiment",
    "target",
    "stance",
    "is_sarcasm_mockery",
]

TASK_DISPLAY_NAMES = {
    "sentiment": "sentiment",
    "target": "target",
    "stance": "stance",
    "is_sarcasm_mockery": "is_sarcasm_mockery",
}

# =====================================================================
# 3. 성능지표 설정
# =====================================================================
METRIC_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]

METRIC_DISPLAY_NAMES = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
]

METRIC_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#E15759",
    "#9C9C9C",
]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 4. 열 이름 설정
# =====================================================================
def normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    return df

# =====================================================================
# 5. 분석 이름 정리
# =====================================================================
def normalize_task_name(
    value: str,
) -> str:
    normalized_value = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    task_aliases = {
        "sentiment": "sentiment",
        "gpt_sentiment": "sentiment",
        "gold_sentiment": "sentiment",
        "정서": "sentiment",
        "정서_극성": "sentiment",

        "target": "target",
        "gpt_target": "target",
        "gold_target": "target",
        "대상": "target",
        "평가_대상": "target",

        "stance": "stance",
        "gpt_stance": "stance",
        "gold_stance": "stance",
        "태도": "stance",
        "대상에_대한_태도": "stance",

        "is_sarcasm_mockery": (
            "is_sarcasm_mockery"
        ),
        "gpt_is_sarcasm_mockery": (
            "is_sarcasm_mockery"
        ),
        "gold_is_sarcasm_mockery": (
            "is_sarcasm_mockery"
        ),
        "sarcasm": (
            "is_sarcasm_mockery"
        ),
        "sarcasm_mockery": (
            "is_sarcasm_mockery"
        ),
        "조롱": (
            "is_sarcasm_mockery"
        ),
        "조롱_냉소": (
            "is_sarcasm_mockery"
        ),
        "조롱_냉소_여부": (
            "is_sarcasm_mockery"
        ),
    }

    return task_aliases.get(
        normalized_value,
        normalized_value,
    )

# =====================================================================
# 6. 성능평가 결과 불러오기
# =====================================================================
def load_evaluation_summary() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "GPT 성능평가 요약 파일이 없습니다: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    df = normalize_column_names(
        df
    )

    required_columns = {
        "task",
        *METRIC_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "성능평가 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}\n"
            f"현재 열: {df.columns.tolist()}"
        )

    df["task"] = (
        df["task"]
        .apply(normalize_task_name)
    )

    return df

# =====================================================================
# 7. 전체 300건 평가 결과만 선택
# =====================================================================
def select_overall_results(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    overall_values = {
        "overall",
        "all",
        "total",
        "global",
        "전체",
        "통합",
        "통합데이터",
    }

    group_type_columns = [
        "group_type",
        "scope",
        "analysis_scope",
        "analysis_level",
        "evaluation_group",
        "subset",
    ]

    group_value_columns = [
        "group_value",
        "scope_value",
        "analysis_group",
        "group_name",
        "subset_value",
    ]

    selected_df = None

    for column in group_type_columns:
        if column not in df.columns:
            continue

        normalized_values = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        overall_mask = (
            normalized_values
            .isin(overall_values)
        )

        if overall_mask.any():
            selected_df = (
                df.loc[
                    overall_mask
                ]
                .copy()
            )
            break

    if selected_df is None:
        for column in group_value_columns:
            if column not in df.columns:
                continue

            normalized_values = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            overall_mask = (
                normalized_values
                .isin(overall_values)
            )

            if overall_mask.any():
                selected_df = (
                    df.loc[
                        overall_mask
                    ]
                    .copy()
                )
                break

    if selected_df is None:
        task_counts = (
            df["task"]
            .value_counts()
        )

        expected_tasks_present = all(
            task in task_counts.index
            for task in TASK_ORDER
        )

        one_row_per_task = (
            expected_tasks_present
            and all(
                task_counts[task] == 1
                for task in TASK_ORDER
            )
        )

        if one_row_per_task:
            selected_df = df.copy()

    if selected_df is None:
        diagnostic_columns = [
            column
            for column in (
                group_type_columns
                + group_value_columns
            )
            if column in df.columns
        ]

        diagnostic_text = {}

        for column in diagnostic_columns:
            diagnostic_text[column] = (
                df[column]
                .astype(str)
                .unique()
                .tolist()
            )

        raise ValueError(
            "전체 평가 결과를 자동으로 찾지 못했습니다.\n"
            f"과업별 행 수: "
            f"{df['task'].value_counts().to_dict()}\n"
            f"그룹 열 값: {diagnostic_text}"
        )

    return selected_df


# =====================================================================
# 8. 과업별 한 행 검사
# =====================================================================
def validate_task_rows(
    df: pd.DataFrame,
) -> None:
    missing_tasks = [
        task
        for task in TASK_ORDER
        if task not in set(
            df["task"]
        )
    ]

    if missing_tasks:
        raise ValueError(
            "전체 성능평가 결과에 없는 과업이 있습니다: "
            f"{missing_tasks}\n"
            f"현재 과업: "
            f"{sorted(df['task'].unique().tolist())}"
        )

    duplicate_tasks = (
        df["task"]
        .value_counts()
    )

    duplicate_tasks = (
        duplicate_tasks[
            duplicate_tasks > 1
        ]
    )

    if not duplicate_tasks.empty:
        raise ValueError(
            "전체 성능평가 결과에 동일 과업이 "
            "여러 행 존재합니다: "
            f"{duplicate_tasks.to_dict()}"
        )


# =====================================================================
# 9. 성능지표 숫자 변환
# =====================================================================
def convert_metric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    for column in METRIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_number_mask = (
            df[column]
            .isna()
        )

        if invalid_number_mask.any():
            invalid_tasks = df.loc[
                invalid_number_mask,
                "task",
            ].tolist()

            raise ValueError(
                f"{column}에 숫자가 아닌 값이 있습니다: "
                f"{invalid_tasks}"
            )

        if df[column].max() > 1:
            if df[column].max() <= 100:
                df[column] = (
                    df[column]
                    / 100
                )
            else:
                raise ValueError(
                    f"{column} 값이 정상 범위를 벗어났습니다."
                )

        invalid_range_mask = (
            df[column].lt(0)
            | df[column].gt(1)
        )

        if invalid_range_mask.any():
            invalid_values = df.loc[
                invalid_range_mask,
                [
                    "task",
                    column,
                ],
            ].to_dict(
                orient="records"
            )

            raise ValueError(
                f"{column} 값이 0~1 범위를 벗어났습니다: "
                f"{invalid_values}"
            )

    return df

# =====================================================================
# 10. 과업 순서 정리
# =====================================================================
def prepare_plot_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    plot_df = (
        df[
            df["task"].isin(
                TASK_ORDER
            )
        ]
        .copy()
    )

    validate_task_rows(
        plot_df
    )

    plot_df["task_order"] = (
        plot_df["task"]
        .map(
            {
                task: index
                for index, task in enumerate(
                    TASK_ORDER
                )
            }
        )
    )

    plot_df = (
        plot_df
        .sort_values(
            "task_order"
        )
        .reset_index(
            drop=True
        )
    )

    plot_df["task_display_name"] = (
        plot_df["task"]
        .map(
            TASK_DISPLAY_NAMES
        )
    )

    output_columns = [
        "task",
        "task_display_name",
    ]

    if "n" in plot_df.columns:
        output_columns.append(
            "n"
        )

    output_columns.extend(
        METRIC_COLUMNS
    )

    return plot_df[
        output_columns
    ].copy()


# =====================================================================
# 11. 전체 성능지표 묶음 막대그래프
# =====================================================================
def create_overall_metrics_figure(
    plot_df: pd.DataFrame,
) -> None:
    x = np.arange(
        len(plot_df)
    )

    cluster_width = 0.84

    bar_width = (
        cluster_width
        / len(METRIC_COLUMNS)
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for metric_index, (
        metric,
        display_name,
    ) in enumerate(
        zip(
            METRIC_COLUMNS,
            METRIC_DISPLAY_NAMES,
        )
    ):
        positions = (
            x
            - cluster_width / 2
            + bar_width / 2
            + metric_index * bar_width
        )

        values = (
            plot_df[metric]
            .astype(float)
            .values
        )

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=METRIC_COLORS[
                metric_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=display_name,
        )
        for bar, value in zip(
            bars,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="#222222",
            )

    ax.set_ylim(
        0,
        1.14,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df[
            "task_display_name"
        ],
        fontsize=10,
    )

    ax.set_ylabel(
        "Score",
        fontsize=10,
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.12,
        ),
        fontsize=8,
        columnspacing=1.4,
        handlelength=1.5,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# =====================================================================
# 12. 시각화 데이터 저장
# =====================================================================
def save_plot_data(
    plot_df: pd.DataFrame,
) -> None:
    plot_df.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

# =====================================================================
# 13. 실행
# =====================================================================
def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df = (
        load_evaluation_summary()
    )

    overall_df = (
        select_overall_results(
            evaluation_df
        )
    )

    overall_df = (
        convert_metric_columns(
            overall_df
        )
    )

    plot_df = (
        prepare_plot_data(
            overall_df
        )
    )

    create_overall_metrics_figure(
        plot_df
    )

    save_plot_data(
        plot_df
    )


if __name__ == "__main__":
    main()