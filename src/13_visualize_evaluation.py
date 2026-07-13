"""
GPT 성능평가 결과 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from labels import (
    SENTIMENT_VALUES,
    TARGET_VALUES,
    STANCE_VALUES,
    SARCASM_VALUES,
)


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
SUMMARY_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_summary.csv"
)

PER_CLASS_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_per_class.csv"
)

CONFUSION_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_confusion.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
    / "evaluation"
)


# =====================================================================
# 2. 분석 설정
# =====================================================================
TASK_CONFIG = {
    "sentiment": {
        "labels": SENTIMENT_VALUES,
        "class_filename": (
            "figure_08_sentiment_class_performance.png"
        ),
        "confusion_filename": (
            "figure_12_confusion_sentiment.png"
        ),
    },
    "target": {
        "labels": TARGET_VALUES,
        "class_filename": (
            "figure_09_target_class_performance.png"
        ),
        "confusion_filename": (
            "figure_13_confusion_target.png"
        ),
    },
    "stance": {
        "labels": STANCE_VALUES,
        "class_filename": (
            "figure_10_stance_class_performance.png"
        ),
        "confusion_filename": (
            "figure_14_confusion_stance.png"
        ),
    },
    "is_sarcasm_mockery": {
        "labels": SARCASM_VALUES,
        "class_filename": (
            "figure_11_is_sarcasm_mockery_class_performance.png"
        ),
        "confusion_filename": (
            "figure_15_confusion_is_sarcasm_mockery.png"
        ),
    },
}


TASK_ORDER = list(
    TASK_CONFIG.keys()
)


# =====================================================================
# 3. 그래프 설정
# =====================================================================
CASE_DISPLAY_NAMES = {
    "01": "룬 설정 오류",
    "02": "강타 재사용 대기시간 오류",
    "03": "경기 하이라이트·두 오류 포함",
}

PERFORMANCE_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#E15759",
    "#8C78A8",
]

GROUP_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#E15759",
]

CLASS_METRIC_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 4. 자료 불러오기
# =====================================================================
def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"평가 요약 파일이 없습니다: {SUMMARY_PATH}"
        )

    df = pd.read_csv(
        SUMMARY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "scope",
        "group",
        "task",
        "n",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"평가 요약 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "n",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["n"] = df["n"].astype(int)

    return df


def load_per_class() -> pd.DataFrame:
    if not PER_CLASS_PATH.exists():
        raise FileNotFoundError(
            f"클래스별 평가 파일이 없습니다: {PER_CLASS_PATH}"
        )

    df = pd.read_csv(
        PER_CLASS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "scope",
        "group",
        "task",
        "label",
        "support",
        "predicted_n",
        "precision",
        "recall",
        "f1",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"클래스별 평가 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "support",
        "predicted_n",
        "precision",
        "recall",
        "f1",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["support"] = df[
        "support"
    ].astype(int)

    df["predicted_n"] = df[
        "predicted_n"
    ].astype(int)

    return df


def load_confusion() -> pd.DataFrame:
    if not CONFUSION_PATH.exists():
        raise FileNotFoundError(
            f"혼동행렬 파일이 없습니다: {CONFUSION_PATH}"
        )

    df = pd.read_csv(
        CONFUSION_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "scope",
        "group",
        "task",
        "gold_label",
        "predicted_label",
        "count",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"혼동행렬 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    df["count"] = pd.to_numeric(
        df["count"],
        errors="raise",
    ).astype(int)

    return df


# =====================================================================
# 5. 공통 그래프 설정
# =====================================================================
def apply_axis_style(
    ax,
) -> None:
    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


def add_score_labels(
    ax,
    bars,
    values,
    offset: float = 0.015,
) -> None:
    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )


# =====================================================================
# 6. 전체 성능
# =====================================================================
def create_overall_performance_chart(
    summary_df: pd.DataFrame,
) -> None:
    overall_df = summary_df[
        summary_df["scope"].eq(
            "overall"
        )
        & summary_df["group"].eq(
            "all"
        )
    ].copy()

    overall_df["task"] = pd.Categorical(
        overall_df["task"],
        categories=TASK_ORDER,
        ordered=True,
    )

    overall_df = overall_df.sort_values(
        "task"
    )

    if len(overall_df) != len(TASK_ORDER):
        raise ValueError(
            "전체 성능 결과에 네 개 과업이 모두 없습니다."
        )

    metric_config = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro-Precision"),
        ("macro_recall", "Macro-Recall"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]

    x = np.arange(
        len(TASK_ORDER)
    )

    bar_width = (
        0.84
        / len(metric_config)
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for metric_index, (
        metric_column,
        metric_name,
    ) in enumerate(metric_config):
        positions = (
            x
            - 0.42
            + bar_width / 2
            + metric_index * bar_width
        )

        values = overall_df[
            metric_column
        ].to_numpy()

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=PERFORMANCE_COLORS[
                metric_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=metric_name,
        )

        add_score_labels(
            ax=ax,
            bars=bars,
            values=values,
        )

    ax.set_ylim(
        0,
        1.13,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        TASK_ORDER,
        fontsize=10,
    )

    ax.set_ylabel(
        "Score",
        fontsize=10,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.15,
        ),
        fontsize=9,
    )

    apply_axis_style(ax)

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "figure_05_overall_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 7. 사례별·댓글 유형별 Macro-F1
# =====================================================================
def get_scope_groups(
    summary_df: pd.DataFrame,
    scope: str,
) -> list[dict]:
    groups = []

    overall_df = summary_df[
        summary_df["scope"].eq(
            "overall"
        )
        & summary_df["group"].eq(
            "all"
        )
    ].copy()

    groups.append(
        {
            "name": (
                f"통합데이터 "
                f"(N={int(overall_df['n'].iloc[0]):,})"
            ),
            "data": overall_df,
        }
    )

    scope_df = summary_df[
        summary_df["scope"].eq(
            scope
        )
    ].copy()

    group_names = sorted(
        scope_df["group"].unique()
    )

    for group_name in group_names:
        group_df = scope_df[
            scope_df["group"].eq(
                group_name
            )
        ].copy()

        group_n = int(
            group_df["n"].iloc[0]
        )

        if scope == "case_id":
            normalized_id = (
                str(group_name)
                .strip()
                .zfill(2)
            )

            display_name = (
                CASE_DISPLAY_NAMES.get(
                    normalized_id,
                    f"사례 {normalized_id}",
                )
            )
        else:
            display_name = str(
                group_name
            )

        groups.append(
            {
                "name": (
                    f"{display_name} "
                    f"(N={group_n:,})"
                ),
                "data": group_df,
            }
        )

    return groups


def create_group_macro_f1_chart(
    summary_df: pd.DataFrame,
    scope: str,
    filename: str,
) -> None:
    groups = get_scope_groups(
        summary_df,
        scope,
    )

    if len(groups) > len(GROUP_COLORS):
        raise ValueError(
            "분석 집단보다 설정된 색상 수가 적습니다."
        )

    x = np.arange(
        len(TASK_ORDER)
    )

    bar_width = (
        0.84
        / len(groups)
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for group_index, group in enumerate(
        groups
    ):
        group_df = group[
            "data"
        ].copy()

        group_df["task"] = pd.Categorical(
            group_df["task"],
            categories=TASK_ORDER,
            ordered=True,
        )

        group_df = group_df.sort_values(
            "task"
        )

        positions = (
            x
            - 0.42
            + bar_width / 2
            + group_index * bar_width
        )

        values = group_df[
            "macro_f1"
        ].to_numpy()

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=GROUP_COLORS[
                group_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=group["name"],
        )

        add_score_labels(
            ax=ax,
            bars=bars,
            values=values,
        )

    ax.set_ylim(
        0,
        1.13,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        TASK_ORDER,
        fontsize=10,
    )

    ax.set_ylabel(
        "Macro-F1",
        fontsize=10,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.16,
        ),
        fontsize=9,
    )

    apply_axis_style(ax)

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 8. 클래스별 성능
# =====================================================================
def create_class_performance_chart(
    per_class_df: pd.DataFrame,
    task: str,
    labels: list[str],
    output_path,
) -> None:
    task_df = per_class_df[
        per_class_df["scope"].eq(
            "overall"
        )
        & per_class_df["group"].eq(
            "all"
        )
        & per_class_df["task"].eq(
            task
        )
    ].copy()

    task_df["label"] = pd.Categorical(
        task_df["label"],
        categories=labels,
        ordered=True,
    )

    task_df = task_df.sort_values(
        "label"
    )

    if len(task_df) != len(labels):
        raise ValueError(
            f"{task} 클래스별 결과에 "
            "필요한 라벨이 모두 없습니다."
        )

    x = np.arange(
        len(labels)
    )

    metric_config = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
    ]

    bar_width = (
        0.78
        / len(metric_config)
    )

    figure_width = max(
        9,
        len(labels) * 2.2,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            7,
        )
    )

    for metric_index, (
        metric_column,
        metric_name,
    ) in enumerate(metric_config):
        positions = (
            x
            - 0.39
            + bar_width / 2
            + metric_index * bar_width
        )

        values = task_df[
            metric_column
        ].to_numpy()

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=CLASS_METRIC_COLORS[
                metric_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=metric_name,
        )

        add_score_labels(
            ax=ax,
            bars=bars,
            values=values,
        )

    x_labels = [
        (
            f"{label}\n"
            f"(N={support:,})"
        )
        for label, support in zip(
            labels,
            task_df["support"],
        )
    ]

    ax.set_ylim(
        0,
        1.13,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        x_labels,
        fontsize=10,
    )

    ax.set_ylabel(
        "Score",
        fontsize=10,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.12,
        ),
        fontsize=9,
    )

    apply_axis_style(ax)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 9. 혼동행렬
# =====================================================================
def create_confusion_matrix_chart(
    confusion_df: pd.DataFrame,
    task: str,
    labels: list[str],
    output_path,
) -> None:
    task_df = confusion_df[
        confusion_df["scope"].eq(
            "overall"
        )
        & confusion_df["group"].eq(
            "all"
        )
        & confusion_df["task"].eq(
            task
        )
    ].copy()

    count_matrix = (
        task_df.pivot(
            index="gold_label",
            columns="predicted_label",
            values="count",
        )
        .reindex(
            index=labels,
            columns=labels,
            fill_value=0,
        )
        .fillna(0)
        .astype(int)
    )

    row_totals = count_matrix.sum(
        axis=1
    )

    row_percentages = (
        count_matrix
        .div(
            row_totals.replace(
                0,
                np.nan,
            ),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    matrix_values = (
        count_matrix.to_numpy()
    )

    figure_size = max(
        7,
        len(labels) * 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    image = ax.imshow(
        matrix_values,
        cmap="Blues",
        aspect="equal",
    )

    max_value = (
        matrix_values.max()
        if matrix_values.size > 0
        else 0
    )

    threshold = (
        max_value * 0.5
    )

    for row_index in range(
        len(labels)
    ):
        for column_index in range(
            len(labels)
        ):
            count = int(
                count_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            percentage = float(
                row_percentages.iloc[
                    row_index,
                    column_index,
                ]
            )

            text_color = (
                "white"
                if count > threshold
                else "#222222"
            )

            ax.text(
                column_index,
                row_index,
                (
                    f"{count:,}\n"
                    f"({percentage:.1f}%)"
                ),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=30,
        ha="right",
        fontsize=10,
    )

    ax.set_yticklabels(
        labels,
        fontsize=10,
    )

    ax.set_xlabel(
        "GPT 예측",
        fontsize=10,
    )

    ax.set_ylabel(
        "인간 기준 라벨",
        fontsize=10,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "건수",
        fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 10. 실행
# =====================================================================
def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = load_summary()
    per_class_df = load_per_class()
    confusion_df = load_confusion()

    create_overall_performance_chart(
        summary_df
    )

    create_group_macro_f1_chart(
        summary_df=summary_df,
        scope="case_id",
        filename="figure_06_case_macro_f1.png",
    )

    create_group_macro_f1_chart(
        summary_df=summary_df,
        scope="comment_type",
        filename="figure_07_comment_type_macro_f1.png",
    )

    for task, config in TASK_CONFIG.items():
        class_output_path = (
            FIGURE_DIR
            / config["class_filename"]
        )

        create_class_performance_chart(
            per_class_df=per_class_df,
            task=task,
            labels=config["labels"],
            output_path=class_output_path,
        )

        confusion_output_path = (
            FIGURE_DIR
            / config["confusion_filename"]
        )

        create_confusion_matrix_chart(
            confusion_df=confusion_df,
            task=task,
            labels=config["labels"],
            output_path=confusion_output_path,
        )

    print("=" * 60)
    print("GPT 성능평가 시각화 완료")
    print("=" * 60)
    print(f"저장 폴더: {FIGURE_DIR}")

    for path in sorted(
        FIGURE_DIR.glob("*.png")
    ):
        print(path)


if __name__ == "__main__":
    main()