"""
부분 인간 코딩 자료를 이용한 임시 GPT 성능평가

현재 용도:
- coder1_coding.xlsx에서 코딩이 완료된 행만 추출
- 최대 100건까지 GPT 예측과 비교
- 과업별 성능지표 산출
- 클래스별 성능지표 산출
- 혼동행렬과 오분류 사례 저장
- 성능지표 및 혼동행렬 시각화

주의:
- 최종 논문용 평가가 아니라 중간 점검용 코드입니다.
- 05_finalize_gold_labels.py를 실행하지 않아도 됩니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import (
    HUMAN_DIR,
    RESULTS_DIR,
)
from labels import (
    SENTIMENT_VALUES,
    TARGET_VALUES,
    STANCE_VALUES,
    SARCASM_VALUES,
)
from text_utils import (
    normalize_label,
    normalize_boolean_text,
)


# =====================================================================
# 1. 평가 설정
# =====================================================================

# 현재 위에서부터 코딩한 100건만 사용
MAX_LABELED_ROWS = 111


# 인간 코딩 파일
CODER_PATH = (
    HUMAN_DIR
    / "coder1_coding.xlsx"
)

# GPT 전체 예측 결과
GPT_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.csv"
)

# 임시 평가 결과 폴더
OUTPUT_DIR = (
    RESULTS_DIR
    / "partial_evaluation_100"
)

# 임시 평가 그림 폴더
FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)


# =====================================================================
# 2. 과업 설정
# =====================================================================

TASK_CONFIG = {
    "sentiment": {
        "human_column": "sentiment",
        "gpt_column": "gpt_sentiment",
        "labels": SENTIMENT_VALUES,
        "confusion_filename": (
            "figure_02_confusion_sentiment.png"
        ),
    },
    "target": {
        "human_column": "target",
        "gpt_column": "gpt_target",
        "labels": TARGET_VALUES,
        "confusion_filename": (
            "figure_03_confusion_target.png"
        ),
    },
    "stance": {
        "human_column": "stance",
        "gpt_column": "gpt_stance",
        "labels": STANCE_VALUES,
        "confusion_filename": (
            "figure_04_confusion_stance.png"
        ),
    },
    "is_sarcasm_mockery": {
        "human_column": "is_sarcasm_mockery",
        "gpt_column": (
            "gpt_is_sarcasm_mockery"
        ),
        "labels": SARCASM_VALUES,
        "confusion_filename": (
            "figure_05_confusion_sarcasm.png"
        ),
    },
}


TASK_NAMES = list(
    TASK_CONFIG.keys()
)


# 그래프에 사용하는 차분한 색상
METRIC_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#E15759",
    "#9C9C9C",
]


# Windows 한글 표시
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 3. 인간 코딩 자료 불러오기
# =====================================================================

def load_partial_human_coding() -> pd.DataFrame:
    """
    coder1_coding.xlsx에서 네 개 라벨이 모두 입력된 행만
    원래 Excel 순서대로 최대 100건 추출합니다.
    """

    if not CODER_PATH.exists():
        raise FileNotFoundError(
            f"인간 코딩 파일이 없습니다: {CODER_PATH}"
        )

    df = pd.read_excel(
        CODER_PATH,
        sheet_name="Coding",
        dtype=str,
    ).fillna("")

    required_columns = {
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "parent_text",
        "analysis_text",
        "sentiment",
        "target",
        "stance",
        "is_sarcasm_mockery",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "인간 코딩 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------------
    # 라벨 표기 정규화
    # ---------------------------------------------------------------

    df["sentiment"] = (
        df["sentiment"]
        .apply(normalize_label)
    )

    df["target"] = (
        df["target"]
        .apply(normalize_label)
    )

    df["stance"] = (
        df["stance"]
        .apply(normalize_label)
    )

    df["is_sarcasm_mockery"] = (
        df["is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    human_label_columns = [
        config["human_column"]
        for config in TASK_CONFIG.values()
    ]

    # 네 개 과업이 모두 입력된 행만 평가 대상으로 선택
    complete_mask = (
        df[human_label_columns]
        .ne("")
        .all(axis=1)
    )

    completed_df = (
        df.loc[complete_mask]
        .copy()
    )

    if completed_df.empty:
        raise ValueError(
            "네 개 라벨이 모두 작성된 행이 없습니다. "
            "Excel 파일을 저장했는지 확인하십시오."
        )

    # 원래 Excel 순서에서 최대 100건만 사용
    completed_df = (
        completed_df
        .head(MAX_LABELED_ROWS)
        .reset_index(drop=True)
    )

    if completed_df[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "평가 표본에 중복 sample_id가 있습니다."
        )

    if completed_df[
        "analysis_unit_id"
    ].duplicated().any():
        raise ValueError(
            "평가 표본에 중복 analysis_unit_id가 있습니다."
        )

    validate_human_labels(
        completed_df
    )

    return completed_df


# =====================================================================
# 4. 인간 라벨 검사
# =====================================================================

def validate_human_labels(
    df: pd.DataFrame,
) -> None:
    """
    인간 코딩 라벨이 현재 labels.py의 정의와 일치하는지 검사합니다.
    """

    for task, config in TASK_CONFIG.items():
        column = config[
            "human_column"
        ]

        valid_labels = config[
            "labels"
        ]

        invalid_mask = ~df[
            column
        ].isin(valid_labels)

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    column,
                ].unique()
            )

            invalid_samples = df.loc[
                invalid_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{task}에 잘못된 인간 라벨이 있습니다. "
                f"값: {invalid_values}, "
                f"표본: {invalid_samples[:10]}"
            )


# =====================================================================
# 5. GPT 결과 불러오기
# =====================================================================

def load_gpt_predictions() -> pd.DataFrame:
    """
    전체 GPT 최종 예측 파일을 불러옵니다.
    """

    if not GPT_PATH.exists():
        raise FileNotFoundError(
            f"GPT 최종 예측 파일이 없습니다: {GPT_PATH}"
        )

    df = pd.read_csv(
        GPT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "analysis_unit_id",
        "gpt_sentiment",
        "gpt_target",
        "gpt_stance",
        "gpt_is_sarcasm_mockery",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "GPT 예측 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df[
        "analysis_unit_id"
    ].duplicated().any():
        raise ValueError(
            "GPT 결과에 중복 analysis_unit_id가 있습니다."
        )

    # 분류 이유가 없는 이전 결과에도 대응
    if "gpt_reason" not in df.columns:
        df["gpt_reason"] = ""

    # ---------------------------------------------------------------
    # GPT 라벨 표기 정규화
    # ---------------------------------------------------------------

    df["gpt_sentiment"] = (
        df["gpt_sentiment"]
        .apply(normalize_label)
    )

    df["gpt_target"] = (
        df["gpt_target"]
        .apply(normalize_label)
    )

    df["gpt_stance"] = (
        df["gpt_stance"]
        .apply(normalize_label)
    )

    df["gpt_is_sarcasm_mockery"] = (
        df["gpt_is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    validate_gpt_labels(df)

    return df


# =====================================================================
# 6. GPT 라벨 검사
# =====================================================================

def validate_gpt_labels(
    df: pd.DataFrame,
) -> None:
    """
    GPT 라벨이 현재 labels.py의 정의와 일치하는지 검사합니다.
    """

    for task, config in TASK_CONFIG.items():
        column = config[
            "gpt_column"
        ]

        valid_labels = config[
            "labels"
        ]

        invalid_mask = ~df[
            column
        ].isin(valid_labels)

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    column,
                ].unique()
            )

            raise ValueError(
                f"{task}에 잘못된 GPT 라벨이 있습니다: "
                f"{invalid_values}"
            )


# =====================================================================
# 7. 인간 코딩과 GPT 결과 결합
# =====================================================================

def merge_human_and_gpt(
    human_df: pd.DataFrame,
    gpt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    analysis_unit_id를 기준으로 인간 코딩과 GPT 결과를 결합합니다.
    """

    gpt_columns = [
        "analysis_unit_id",
        "gpt_sentiment",
        "gpt_target",
        "gpt_stance",
        "gpt_is_sarcasm_mockery",
        "gpt_reason",
    ]

    merged_df = human_df.merge(
        gpt_df[gpt_columns],
        on="analysis_unit_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_mask = (
        merged_df["_merge"]
        .ne("both")
    )

    if missing_mask.any():
        missing_ids = merged_df.loc[
            missing_mask,
            "analysis_unit_id",
        ].tolist()

        raise ValueError(
            "일부 인간 코딩 표본의 GPT 결과가 없습니다: "
            f"{missing_ids[:10]}"
        )

    merged_df = (
        merged_df
        .drop(columns=["_merge"])
    )

    return merged_df


# =====================================================================
# 8. 과업별 성능지표 계산
# =====================================================================

def calculate_task_metrics(
    merged_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    다음 결과를 생성합니다.

    1. 과업별 전체 성능
    2. 클래스별 성능
    3. 혼동행렬
    4. 오분류 사례
    """

    overall_rows = []
    per_class_rows = []
    confusion_rows = []
    error_frames = []

    for task, config in TASK_CONFIG.items():
        human_column = config[
            "human_column"
        ]

        gpt_column = config[
            "gpt_column"
        ]

        defined_labels = list(
            config["labels"]
        )

        y_true = merged_df[
            human_column
        ]

        y_pred = merged_df[
            gpt_column
        ]

        # 인간 코딩에 실제로 등장한 클래스
        observed_labels = [
            label
            for label in defined_labels
            if label in set(y_true)
        ]

        if not observed_labels:
            raise ValueError(
                f"{task}에서 관찰된 인간 라벨이 없습니다."
            )

        # -----------------------------------------------------------
        # 전체 성능지표
        # -----------------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=observed_labels,
            average="macro",
            zero_division=0,
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _,
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=defined_labels,
            average="weighted",
            zero_division=0,
        )

        overall_rows.append(
            {
                "task": task,
                "n": len(merged_df),
                "defined_class_count": (
                    len(defined_labels)
                ),
                "observed_gold_class_count": (
                    len(observed_labels)
                ),
                "accuracy": accuracy,
                "macro_precision": (
                    macro_precision
                ),
                "macro_recall": (
                    macro_recall
                ),
                "macro_f1": macro_f1,
                "weighted_precision": (
                    weighted_precision
                ),
                "weighted_recall": (
                    weighted_recall
                ),
                "weighted_f1": weighted_f1,
            }
        )

        # -----------------------------------------------------------
        # 클래스별 성능지표
        # -----------------------------------------------------------

        (
            class_precision,
            class_recall,
            class_f1,
            class_support,
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=defined_labels,
            average=None,
            zero_division=0,
        )

        for (
            label,
            precision,
            recall,
            f1,
            support,
        ) in zip(
            defined_labels,
            class_precision,
            class_recall,
            class_f1,
            class_support,
        ):
            predicted_count = int(
                y_pred.eq(label).sum()
            )

            per_class_rows.append(
                {
                    "task": task,
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "gold_support": int(
                        support
                    ),
                    "gpt_predicted_count": (
                        predicted_count
                    ),
                }
            )

        # -----------------------------------------------------------
        # 혼동행렬
        # -----------------------------------------------------------

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=defined_labels,
        )

        for gold_index, gold_label in enumerate(
            defined_labels
        ):
            for (
                prediction_index,
                prediction_label,
            ) in enumerate(
                defined_labels
            ):
                confusion_rows.append(
                    {
                        "task": task,
                        "gold_label": (
                            gold_label
                        ),
                        "gpt_label": (
                            prediction_label
                        ),
                        "count": int(
                            matrix[
                                gold_index,
                                prediction_index,
                            ]
                        ),
                    }
                )

        # -----------------------------------------------------------
        # 오분류 사례
        # -----------------------------------------------------------

        error_mask = (
            y_true
            .ne(y_pred)
        )

        task_errors = merged_df.loc[
            error_mask,
            [
                "sample_id",
                "analysis_unit_id",
                "case_id",
                "case_name",
                "comment_type",
                "parent_text",
                "analysis_text",
                human_column,
                gpt_column,
                "gpt_reason",
            ],
        ].copy()

        task_errors.insert(
            0,
            "task",
            task,
        )

        task_errors = task_errors.rename(
            columns={
                human_column: "human_label",
                gpt_column: "gpt_label",
            }
        )

        error_frames.append(
            task_errors
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    error_df = pd.concat(
        error_frames,
        ignore_index=True,
    )

    return (
        overall_df,
        per_class_df,
        confusion_df,
        error_df,
    )


# =====================================================================
# 9. 복합 정확도 계산
# =====================================================================

def calculate_exact_match_metrics(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    대상과 태도의 결합 정확도 및
    네 개 과업 전체의 완전 일치 정확도를 계산합니다.
    """

    result_df = merged_df.copy()

    # 대상과 태도가 동시에 맞은 경우
    result_df[
        "target_stance_correct"
    ] = (
        result_df["target"]
        .eq(result_df["gpt_target"])
        &
        result_df["stance"]
        .eq(result_df["gpt_stance"])
    )

    # 네 개 과업이 모두 맞은 경우
    result_df[
        "all_tasks_correct"
    ] = (
        result_df["sentiment"]
        .eq(result_df["gpt_sentiment"])
        &
        result_df["target"]
        .eq(result_df["gpt_target"])
        &
        result_df["stance"]
        .eq(result_df["gpt_stance"])
        &
        result_df[
            "is_sarcasm_mockery"
        ].eq(
            result_df[
                "gpt_is_sarcasm_mockery"
            ]
        )
    )

    total_count = len(
        result_df
    )

    target_stance_correct = int(
        result_df[
            "target_stance_correct"
        ].sum()
    )

    all_tasks_correct = int(
        result_df[
            "all_tasks_correct"
        ].sum()
    )

    exact_match_df = pd.DataFrame(
        [
            {
                "metric": (
                    "target_stance_exact_match"
                ),
                "correct_count": (
                    target_stance_correct
                ),
                "total_count": total_count,
                "accuracy": (
                    target_stance_correct
                    / total_count
                ),
            },
            {
                "metric": (
                    "all_tasks_exact_match"
                ),
                "correct_count": (
                    all_tasks_correct
                ),
                "total_count": total_count,
                "accuracy": (
                    all_tasks_correct
                    / total_count
                ),
            },
        ]
    )

    return (
        exact_match_df,
        result_df,
    )


# =====================================================================
# 10. 결과 저장
# =====================================================================

def save_results(
    overall_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    error_df: pd.DataFrame,
    exact_match_df: pd.DataFrame,
    evaluated_rows_df: pd.DataFrame,
) -> None:
    """
    임시 평가 결과를 CSV와 Excel로 저장합니다.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        OUTPUT_DIR
        / "partial_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_df.to_csv(
        OUTPUT_DIR
        / "partial_per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df.to_csv(
        OUTPUT_DIR
        / "partial_confusion_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    error_df.to_csv(
        OUTPUT_DIR
        / "partial_error_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    exact_match_df.to_csv(
        OUTPUT_DIR
        / "partial_exact_match_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    evaluated_rows_df.to_csv(
        OUTPUT_DIR
        / "partial_evaluated_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excel_path = (
        OUTPUT_DIR
        / "partial_evaluation.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        overall_df.to_excel(
            writer,
            sheet_name="overall",
            index=False,
        )

        per_class_df.to_excel(
            writer,
            sheet_name="per_class",
            index=False,
        )

        confusion_df.to_excel(
            writer,
            sheet_name="confusion",
            index=False,
        )

        exact_match_df.to_excel(
            writer,
            sheet_name="exact_match",
            index=False,
        )

        error_df.to_excel(
            writer,
            sheet_name="errors",
            index=False,
        )

        evaluated_rows_df.to_excel(
            writer,
            sheet_name="evaluated_rows",
            index=False,
        )


# =====================================================================
# 11. 전체 성능지표 시각화
# =====================================================================

def create_overall_metrics_figure(
    overall_df: pd.DataFrame,
) -> None:
    """
    과업별 Accuracy, Macro 지표와 Weighted-F1을
    묶음 세로 막대그래프로 표시합니다.
    """

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    metric_display_names = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
    ]

    x = np.arange(
        len(overall_df)
    )

    cluster_width = 0.84

    bar_width = (
        cluster_width
        / len(metric_columns)
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for metric_index, (
        metric,
        display_name,
    ) in enumerate(
        zip(
            metric_columns,
            metric_display_names,
        )
    ):
        positions = (
            x
            - cluster_width / 2
            + bar_width / 2
            + metric_index * bar_width
        )

        values = (
            overall_df[metric]
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
            )

    ax.set_ylim(
        0,
        1.14,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        overall_df["task"],
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
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "figure_01_overall_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 12. 혼동행렬 시각화
# =====================================================================

def create_confusion_matrix_figure(
    merged_df: pd.DataFrame,
    task: str,
    config: dict,
) -> None:
    """
    인간 라벨을 행, GPT 라벨을 열로 하는 혼동행렬을 생성합니다.
    """

    human_column = config[
        "human_column"
    ]

    gpt_column = config[
        "gpt_column"
    ]

    labels = list(
        config["labels"]
    )

    matrix = confusion_matrix(
        merged_df[human_column],
        merged_df[gpt_column],
        labels=labels,
    )

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

    image = ax.imshow(
        matrix,
        cmap="Blues",
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
        FIGURE_DIR
        / config[
            "confusion_filename"
        ],
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 13. 복합 정확도 시각화
# =====================================================================

def create_exact_match_figure(
    exact_match_df: pd.DataFrame,
) -> None:
    """
    대상-태도 동시 일치와 전체 과업 완전 일치를 표시합니다.
    """

    display_names = [
        "Target + stance",
        "All four tasks",
    ]

    values = (
        exact_match_df[
            "accuracy"
        ]
        .astype(float)
        .values
    )

    correct_counts = (
        exact_match_df[
            "correct_count"
        ]
        .astype(int)
        .values
    )

    total_counts = (
        exact_match_df[
            "total_count"
        ]
        .astype(int)
        .values
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        display_names,
        values,
        color=[
            "#4C78A8",
            "#9C9C9C",
        ],
        edgecolor="#444444",
        linewidth=0.5,
        width=0.55,
    )

    for (
        bar,
        value,
        correct_count,
        total_count,
    ) in zip(
        bars,
        values,
        correct_counts,
        total_counts,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.025,
            (
                f"{correct_count:,}/{total_count:,}\n"
                f"({value * 100:.1f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(
        0,
        1.15,
    )

    ax.set_ylabel(
        "Exact-match accuracy",
        fontsize=10,
    )

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

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "figure_06_exact_match.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 14. 모든 그림 생성
# =====================================================================

def create_figures(
    merged_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    exact_match_df: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_overall_metrics_figure(
        overall_df
    )

    for task, config in TASK_CONFIG.items():
        create_confusion_matrix_figure(
            merged_df=merged_df,
            task=task,
            config=config,
        )

    create_exact_match_figure(
        exact_match_df
    )


# =====================================================================
# 15. 콘솔 결과 출력
# =====================================================================

def print_summary(
    overall_df: pd.DataFrame,
    exact_match_df: pd.DataFrame,
) -> None:
    """
    핵심 지표를 명령 프롬프트에 출력합니다.
    """

    display_columns = [
        "task",
        "n",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    print()
    print("=" * 80)
    print("부분 인간 코딩 기반 GPT 임시 성능평가")
    print("=" * 80)

    print(
        overall_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("-" * 80)
    print("복합 정확도")
    print("-" * 80)

    print(
        exact_match_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print(f"결과 폴더: {OUTPUT_DIR}")
    print(f"그림 폴더: {FIGURE_DIR}")


# =====================================================================
# 16. 실행
# =====================================================================

def main() -> None:
    # 인간 코딩이 완료된 최대 100건 불러오기
    human_df = (
        load_partial_human_coding()
    )

    # 전체 GPT 예측 결과 불러오기
    gpt_df = (
        load_gpt_predictions()
    )

    # 인간 코딩과 GPT 결과 결합
    merged_df = merge_human_and_gpt(
        human_df=human_df,
        gpt_df=gpt_df,
    )

    # 과업별 평가
    (
        overall_df,
        per_class_df,
        confusion_df,
        error_df,
    ) = calculate_task_metrics(
        merged_df
    )

    # 대상-태도 및 전체 과업 완전 일치 평가
    (
        exact_match_df,
        evaluated_rows_df,
    ) = calculate_exact_match_metrics(
        merged_df
    )

    # CSV와 Excel 저장
    save_results(
        overall_df=overall_df,
        per_class_df=per_class_df,
        confusion_df=confusion_df,
        error_df=error_df,
        exact_match_df=exact_match_df,
        evaluated_rows_df=(
            evaluated_rows_df
        ),
    )

    # 그림 생성
    create_figures(
        merged_df=merged_df,
        overall_df=overall_df,
        exact_match_df=exact_match_df,
    )

    # 콘솔 요약
    print_summary(
        overall_df=overall_df,
        exact_match_df=exact_match_df,
    )


if __name__ == "__main__":
    main()