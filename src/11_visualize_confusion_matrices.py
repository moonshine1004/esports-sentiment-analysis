"""
GPT 분류 혼동행렬 시각화
"""

from pathlib import Path

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
INPUT_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_confusion.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

# =====================================================================
# 2. 분석 설정
# =====================================================================
TASK_CONFIG = {
    "sentiment": {
        "labels": SENTIMENT_VALUES,
        "filename": (
            "figure_07_confusion_sentiment.png"
        ),
    },
    "target": {
        "labels": TARGET_VALUES,
        "filename": (
            "figure_08_confusion_target.png"
        ),
    },
    "stance": {
        "labels": STANCE_VALUES,
        "filename": (
            "figure_09_confusion_stance.png"
        ),
    },
    "is_sarcasm_mockery": {
        "labels": SARCASM_VALUES,
        "filename": (
            "figure_10_confusion_sarcasm.png"
        ),
    },
}

TASK_ORDER = [
    "sentiment",
    "target",
    "stance",
    "is_sarcasm_mockery",
]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 3. 열 라벨 설정
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
# 4. 분석명 정규화
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
# 5. 문자열 값 정규화
# =====================================================================
def normalize_group_value(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# =====================================================================
# 6. 혼동행렬 자료 불러오기
# =====================================================================
def load_confusion_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "혼동행렬 자료가 없습니다: "
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

    if (
        "gpt_label" not in df.columns
        and "predicted_label" in df.columns
    ):
        df = df.rename(
            columns={
                "predicted_label": "gpt_label",
            }
        )

    required_columns = {
        "task",
        "gold_label",
        "gpt_label",
        "count",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "혼동행렬 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}\n"
            f"현재 열: {df.columns.tolist()}"
        )

    df["task"] = (
        df["task"]
        .apply(normalize_task_name)
    )

    df["gold_label"] = (
        df["gold_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["gpt_label"] = (
        df["gpt_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["count"] = pd.to_numeric(
        df["count"],
        errors="coerce",
    )

    if df["count"].isna().any():
        invalid_rows = df.loc[
            df["count"].isna()
        ].head(10)

        raise ValueError(
            "count에 숫자가 아닌 값이 있습니다:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    if df["count"].lt(0).any():
        raise ValueError(
            "count에 음수가 있습니다."
        )

    df["count"] = (
        df["count"]
        .astype(int)
    )

    return df

# =====================================================================
# 7. 전체 평가 결과만 선택
# =====================================================================
def select_overall_confusion(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    overall_values = {
        "overall",
        "all",
        "total",
        "global",
        "overall_data",
        "all_data",
        "전체",
        "통합",
        "통합데이터",
        "전체자료",
        "전체_자료",
    }

    group_columns = [
        "group_type",
        "group_value",
        "group_name",
        "scope",
        "scope_value",
        "analysis_scope",
        "analysis_group",
        "analysis_level",
        "evaluation_scope",
        "evaluation_group",
        "subset",
        "subset_value",
    ]

    core_columns = [
        "task",
        "gold_label",
        "gpt_label",
    ]

    for column in group_columns:
        if column not in df.columns:
            continue

        normalized_values = (
            df[column]
            .apply(normalize_group_value)
        )

        overall_mask = (
            normalized_values
            .isin(overall_values)
        )

        if not overall_mask.any():
            continue

        candidate_df = (
            df.loc[
                overall_mask
            ]
            .copy()
        )

        if not candidate_df.duplicated(
            subset=core_columns
        ).any():
            return candidate_df

    duplicate_mask = df.duplicated(
        subset=core_columns
    )

    if not duplicate_mask.any():
        return df

    diagnostic_columns = [
        column
        for column in group_columns
        if column in df.columns
    ]

    diagnostic_values = {}

    for column in diagnostic_columns:
        diagnostic_values[column] = (
            df[column]
            .astype(str)
            .unique()
            .tolist()
        )

    raise ValueError(
        "전체 평가 결과를 자동으로 선택하지 못했습니다.\n"
        "gpt_evaluation_confusion.csv의 그룹 열을 확인하십시오.\n"
        f"그룹 열 값: {diagnostic_values}"
    )

# =====================================================================
# 8. 분석별 라벨 검사
# =====================================================================
def validate_confusion_labels(
    df: pd.DataFrame,
) -> None:
    available_tasks = set(
        df["task"]
    )

    missing_tasks = [
        task
        for task in TASK_ORDER
        if task not in available_tasks
    ]

    if missing_tasks:
        raise ValueError(
            "혼동행렬에 없는 과업이 있습니다: "
            f"{missing_tasks}"
        )

    for task, config in (
        TASK_CONFIG.items()
    ):
        labels = set(
            config["labels"]
        )

        task_df = df[
            df["task"].eq(task)
        ]

        invalid_gold_labels = sorted(
            set(task_df["gold_label"])
            - labels
        )

        invalid_gpt_labels = sorted(
            set(task_df["gpt_label"])
            - labels
        )

        if invalid_gold_labels:
            raise ValueError(
                f"{task}의 인간 라벨에 "
                "정의되지 않은 값이 있습니다: "
                f"{invalid_gold_labels}"
            )

        if invalid_gpt_labels:
            raise ValueError(
                f"{task}의 GPT 라벨에 "
                "정의되지 않은 값이 있습니다: "
                f"{invalid_gpt_labels}"
            )


# =====================================================================
# 9. 분석별 혼동행렬 생성
# =====================================================================

def build_confusion_matrix(
    df: pd.DataFrame,
    task: str,
    labels: list[str],
) -> np.ndarray:
    task_df = (
        df.loc[
            df["task"].eq(task),
            [
                "gold_label",
                "gpt_label",
                "count",
            ],
        ]
        .copy()
    )

    if task_df.empty:
        raise ValueError(
            f"{task} 혼동행렬 자료가 없습니다."
        )

    matrix_df = (
        task_df
        .pivot_table(
            index="gold_label",
            columns="gpt_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(
            index=labels,
            columns=labels,
            fill_value=0,
        )
    )

    matrix = (
        matrix_df
        .to_numpy(
            dtype=int
        )
    )

    if matrix.sum() == 0:
        raise ValueError(
            f"{task} 혼동행렬의 전체 건수가 0입니다."
        )

    return matrix

# =====================================================================
# 10. 혼동행렬 시각화
# =====================================================================
def create_confusion_matrix_figure(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    figure_size = max(
        6,
        len(labels) * 1.25,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    # 테스트 스크립트와 동일한 파란색 계열
    image = ax.imshow(
        matrix,
        cmap="Blues",
        interpolation="nearest",
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
        fontsize=9,
    )

    ax.set_yticklabels(
        labels,
        fontsize=9,
    )

    ax.set_xlabel(
        "GPT label",
        fontsize=10,
    )

    ax.set_ylabel(
        "Human label",
        fontsize=10,
    )

    threshold = (
        matrix.max() / 2
        if matrix.size > 0
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = int(
                matrix[
                    row_index,
                    column_index,
                ]
            )

            text_color = (
                "white"
                if value > threshold
                else "#222222"
            )

            ax.text(
                column_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# =====================================================================
# 11. 실행
# =====================================================================

def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_df = (
        load_confusion_data()
    )

    overall_df = (
        select_overall_confusion(
            confusion_df
        )
    )

    validate_confusion_labels(
        overall_df
    )

    matrix_totals = {}

    for task in TASK_ORDER:
        config = TASK_CONFIG[
            task
        ]

        labels = list(
            config["labels"]
        )

        matrix = build_confusion_matrix(
            df=overall_df,
            task=task,
            labels=labels,
        )

        output_path = (
            FIGURE_DIR
            / config["filename"]
        )

        create_confusion_matrix_figure(
            matrix=matrix,
            labels=labels,
            output_path=output_path,
        )

        matrix_totals[task] = int(
            matrix.sum()
        )

if __name__ == "__main__":
    main()