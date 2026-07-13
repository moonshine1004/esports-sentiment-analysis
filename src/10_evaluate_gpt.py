"""
분류 성능지표 산출
"""

import pandas as pd

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
# 1. 파일 경로 설정
# =====================================================================
GOLD_PATH = (
    HUMAN_DIR
    / "gold_labels.csv"
)

GPT_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.csv"
)

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

ERROR_PATH = (
    RESULTS_DIR
    / "gpt_evaluation_errors.csv"
)

EXCEL_PATH = (
    RESULTS_DIR
    / "gpt_evaluation.xlsx"
)


# =====================================================================
# 2. 평가 지표
# =====================================================================
TASK_CONFIG = {
    "sentiment": {
        "gold_column": "gold_sentiment",
        "gpt_column": "gpt_sentiment",
        "labels": SENTIMENT_VALUES,
    },
    "target": {
        "gold_column": "gold_target",
        "gpt_column": "gpt_target",
        "labels": TARGET_VALUES,
    },
    "stance": {
        "gold_column": "gold_stance",
        "gpt_column": "gpt_stance",
        "labels": STANCE_VALUES,
    },
    "is_sarcasm_mockery": {
        "gold_column": "gold_is_sarcasm_mockery",
        "gpt_column": "gpt_is_sarcasm_mockery",
        "labels": SARCASM_VALUES,
    },
}

# =====================================================================
# 3. 인간 코딩 데이터 불러오기
# =====================================================================
def load_gold_labels() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"인간 기준 라벨이 없습니다: {GOLD_PATH}"
        )

    df = pd.read_csv(
        GOLD_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "parent_text",
        "analysis_text",
        "gold_sentiment",
        "gold_target",
        "gold_stance",
        "gold_is_sarcasm_mockery",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"인간 기준 라벨에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "인간 기준 라벨에 중복 analysis_unit_id가 있습니다."
        )

    df["gold_sentiment"] = (
        df["gold_sentiment"]
        .apply(normalize_label)
    )

    df["gold_target"] = (
        df["gold_target"]
        .apply(normalize_label)
    )

    df["gold_stance"] = (
        df["gold_stance"]
        .apply(normalize_label)
    )

    df["gold_is_sarcasm_mockery"] = (
        df["gold_is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    validate_labels(
        df=df,
        label_source="gold",
    )

    return df

# =====================================================================
# 4. GPT 분류 불러오기
# =====================================================================
def load_gpt_predictions() -> pd.DataFrame:
    if not GPT_PATH.exists():
        raise FileNotFoundError(
            f"GPT 최종 예측이 없습니다: {GPT_PATH}"
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
        "model",
        "prompt_version",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"GPT 예측에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "GPT 예측에 중복 analysis_unit_id가 있습니다."
        )

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

    validate_labels(
        df=df,
        label_source="gpt",
    )

    return df


# =====================================================================
# 5. 라벨 검사
# =====================================================================
def validate_labels(
    df: pd.DataFrame,
    label_source: str,
) -> None:
    for task, config in TASK_CONFIG.items():
        column = config[
            f"{label_source}_column"
        ]

        valid_values = config["labels"]

        blank_mask = df[column].eq("")

        if blank_mask.any():
            ids = df.loc[
                blank_mask,
                "analysis_unit_id",
            ].head(10).tolist()

            raise ValueError(
                f"{column}에 빈 값이 있습니다: {ids}"
            )

        invalid_mask = ~df[column].isin(
            valid_values
        )

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    column,
                ].unique()
            )

            raise ValueError(
                f"{column}에 잘못된 라벨이 있습니다: "
                f"{invalid_values}"
            )


# =====================================================================
# 6. 인간 코딩 데이터과 GPT 데이터 결합
# =====================================================================
def merge_evaluation_data(
    gold_df: pd.DataFrame,
    gpt_df: pd.DataFrame,
) -> pd.DataFrame:
    gpt_columns = [
        "analysis_unit_id",
        "gpt_sentiment",
        "gpt_target",
        "gpt_stance",
        "gpt_is_sarcasm_mockery",
        "model",
        "prompt_version",
    ]

    if "prompt_hash" in gpt_df.columns:
        gpt_columns.append(
            "prompt_hash"
        )

    merged_df = gold_df.merge(
        gpt_df[gpt_columns],
        on="analysis_unit_id",
        how="left",
        validate="one_to_one",
    )

    missing_mask = (
        merged_df["gpt_sentiment"]
        .fillna("")
        .eq("")
    )

    if missing_mask.any():
        missing_ids = merged_df.loc[
            missing_mask,
            "analysis_unit_id",
        ].head(10).tolist()

        raise ValueError(
            "인간 표본에 대응하는 GPT 결과가 없습니다: "
            f"{missing_ids}"
        )

    if merged_df["model"].nunique() != 1:
        raise ValueError(
            "평가 자료에 서로 다른 모델 결과가 섞여 있습니다."
        )

    if merged_df["prompt_version"].nunique() != 1:
        raise ValueError(
            "평가 자료에 서로 다른 프롬프트 버전이 섞여 있습니다."
        )

    return merged_df


# =====================================================================
# 7. 평가 데이터 생성
# =====================================================================
def iter_evaluation_groups(
    df: pd.DataFrame,
):
    # 전체
    yield (
        "overall",
        "all",
        df,
    )

    # 사례별
    for case_id, group_df in df.groupby(
        "case_id",
        sort=True,
    ):
        yield (
            "case_id",
            str(case_id),
            group_df,
        )

    # 댓글 유형별
    for comment_type, group_df in df.groupby(
        "comment_type",
        sort=True,
    ):
        yield (
            "comment_type",
            str(comment_type),
            group_df,
        )


# =====================================================================
# 8. 성능 계산
# =====================================================================
def evaluate_task(
    df: pd.DataFrame,
    scope: str,
    group: str,
    task: str,
    config: dict,
) -> tuple[dict, list[dict], list[dict]]:
    gold_column = config["gold_column"]
    gpt_column = config["gpt_column"]
    all_labels = list(config["labels"])

    y_true = df[gold_column]
    y_pred = df[gpt_column]

    present_values = set(
        y_true.tolist()
        + y_pred.tolist()
    )

    present_labels = [
        label
        for label in all_labels
        if label in present_values
    ]

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
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    (
        _,
        _,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_labels,
        average="weighted",
        zero_division=0,
    )

    correct_n = int(
        y_true.eq(y_pred).sum()
    )

    summary_row = {
        "scope": scope,
        "group": group,
        "task": task,
        "n": len(df),
        "correct_n": correct_n,
        "error_n": len(df) - correct_n,
        "evaluated_class_count": len(
            present_labels
        ),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    (
        class_precision,
        class_recall,
        class_f1,
        class_support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=all_labels,
        average=None,
        zero_division=0,
    )

    per_class_rows = []

    for index, label in enumerate(
        all_labels
    ):
        predicted_n = int(
            y_pred.eq(label).sum()
        )

        per_class_rows.append(
            {
                "scope": scope,
                "group": group,
                "task": task,
                "label": label,
                "support": int(
                    class_support[index]
                ),
                "predicted_n": predicted_n,
                "precision": class_precision[index],
                "recall": class_recall[index],
                "f1": class_f1[index],
            }
        )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=all_labels,
    )

    confusion_rows = []

    for gold_index, gold_label in enumerate(
        all_labels
    ):
        for pred_index, predicted_label in enumerate(
            all_labels
        ):
            confusion_rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "task": task,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "count": int(
                        matrix[
                            gold_index,
                            pred_index,
                        ]
                    ),
                }
            )

    return (
        summary_row,
        per_class_rows,
        confusion_rows,
    )


# =====================================================================
# 9. 전체 평가
# =====================================================================
def create_evaluation_results(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_rows = []
    per_class_rows = []
    confusion_rows = []

    for scope, group, group_df in (
        iter_evaluation_groups(df)
    ):
        for task, config in (
            TASK_CONFIG.items()
        ):
            (
                summary_row,
                task_class_rows,
                task_confusion_rows,
            ) = evaluate_task(
                df=group_df,
                scope=scope,
                group=group,
                task=task,
                config=config,
            )

            summary_rows.append(
                summary_row
            )

            per_class_rows.extend(
                task_class_rows
            )

            confusion_rows.extend(
                task_confusion_rows
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(confusion_rows),
    )


# =====================================================================
# 10. 오류 사례 추출
# =====================================================================
def create_error_cases(
    df: pd.DataFrame,
) -> pd.DataFrame:
    error_rows = []

    for task, config in TASK_CONFIG.items():
        gold_column = config["gold_column"]
        gpt_column = config["gpt_column"]

        error_df = df[
            ~df[gold_column].eq(
                df[gpt_column]
            )
        ]

        for _, row in error_df.iterrows():
            error_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "analysis_unit_id": (
                        row["analysis_unit_id"]
                    ),
                    "case_id": row["case_id"],
                    "case_name": row["case_name"],
                    "comment_type": (
                        row["comment_type"]
                    ),
                    "parent_text": (
                        row["parent_text"]
                    ),
                    "analysis_text": (
                        row["analysis_text"]
                    ),
                    "task": task,
                    "gold_label": (
                        row[gold_column]
                    ),
                    "gpt_label": (
                        row[gpt_column]
                    ),
                    "error_type": (
                        f"{row[gold_column]}"
                        f" → "
                        f"{row[gpt_column]}"
                    ),
                    "model": row["model"],
                    "prompt_version": (
                        row["prompt_version"]
                    ),
                }
            )

    return pd.DataFrame(error_rows)


# =====================================================================
# 11. 결과 저장
# =====================================================================
def save_results(
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    per_class_df.to_csv(
        PER_CLASS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df.to_csv(
        CONFUSION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    errors_df.to_csv(
        ERROR_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        per_class_df.to_excel(
            writer,
            sheet_name="per_class",
            index=False,
        )

        confusion_df.to_excel(
            writer,
            sheet_name="confusion_matrix",
            index=False,
        )

        errors_df.to_excel(
            writer,
            sheet_name="error_cases",
            index=False,
        )


# =====================================================================
# 12. 실행
# =====================================================================
def main() -> None:
    gold_df = load_gold_labels()
    gpt_df = load_gpt_predictions()

    evaluation_df = merge_evaluation_data(
        gold_df,
        gpt_df,
    )

    (
        summary_df,
        per_class_df,
        confusion_df,
    ) = create_evaluation_results(
        evaluation_df
    )

    errors_df = create_error_cases(
        evaluation_df
    )

    save_results(
        summary_df,
        per_class_df,
        confusion_df,
        errors_df,
    )

    overall_df = summary_df[
        summary_df["scope"].eq(
            "overall"
        )
    ]

    print("=" * 60)
    print("GPT 분류 성능평가 완료")
    print("=" * 60)
    print(f"평가 표본: {len(evaluation_df)}개")
    print(
        f"모델: "
        f"{evaluation_df['model'].iloc[0]}"
    )
    print(
        f"프롬프트 버전: "
        f"{evaluation_df['prompt_version'].iloc[0]}"
    )

    print()
    print("전체 성능")
    print(
        overall_df[
            [
                "task",
                "n",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"전체 오류 행: {len(errors_df)}개")
    print(f"요약: {SUMMARY_PATH}")
    print(f"클래스별: {PER_CLASS_PATH}")
    print(f"혼동행렬: {CONFUSION_PATH}")
    print(f"오류 사례: {ERROR_PATH}")
    print(f"통합 Excel: {EXCEL_PATH}")


if __name__ == "__main__":
    main()