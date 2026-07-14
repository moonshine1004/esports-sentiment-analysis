"""
GPT 1회 분류 결과를 검사하고 최종 예측 파일을 생성
"""

import pandas as pd

from config import (
    INTERIM_DIR,
    RESULTS_DIR,
    REPEAT_COUNT,
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
RUNS_PATH = (
    RESULTS_DIR
    / "gpt_predictions_runs.csv"
)

TEXT_PATH = (
    INTERIM_DIR
    / "comments_preprocessed.csv"
)

OUTPUT_CSV_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.csv"
)

OUTPUT_XLSX_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.xlsx"
)


# =====================================================================
# 2. 분류 항목
# =====================================================================
TASKS = [
    "sentiment",
    "target",
    "stance",
    "is_sarcasm_mockery",
]

VALID_LABELS = {
    "sentiment": SENTIMENT_VALUES,
    "target": TARGET_VALUES,
    "stance": STANCE_VALUES,
    "is_sarcasm_mockery": SARCASM_VALUES,
}


# =====================================================================
# 3. GPT 결과 불러오기
# =====================================================================
def load_predictions() -> pd.DataFrame:
    if REPEAT_COUNT != 1:
        raise ValueError(
            "이 코드는 REPEAT_COUNT=1인 경우에만 실행할 수 있습니다."
        )

    if not RUNS_PATH.exists():
        raise FileNotFoundError(
            f"GPT 분류 결과가 없습니다: {RUNS_PATH}"
        )

    df = pd.read_csv(
        RUNS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "analysis_unit_id",
        "run_no",
        "sentiment",
        "target",
        "stance",
        "is_sarcasm_mockery",
        "model",
        "prompt_version",
        "prompt_hash",
        "input_hash",
        "response_id",
        "status",
        "error_message",
        "reason",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"GPT 결과에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    # 오류 후 재실행한 경우 마지막 결과 사용
    df = df.drop_duplicates(
        subset=[
            "analysis_unit_id",
            "run_no",
        ],
        keep="last",
    ).copy()

    error_df = df[
        ~df["status"].eq("success")
    ]

    if not error_df.empty:
        error_ids = error_df[
            "analysis_unit_id"
        ].head(10).tolist()

        raise ValueError(
            f"오류 결과가 {len(error_df)}건 있습니다. "
            f"예시: {error_ids}. "
            "08번 코드를 다시 실행하십시오."
        )

    if not df["run_no"].eq("1").all():
        invalid_runs = sorted(
            df.loc[
                ~df["run_no"].eq("1"),
                "run_no",
            ].unique()
        )

        raise ValueError(
            f"1회 분류가 아닌 반복 번호가 있습니다: "
            f"{invalid_runs}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "동일한 analysis_unit_id에 여러 결과가 있습니다."
        )

    # 라벨 표기 통일
    df["sentiment"] = df[
        "sentiment"
    ].apply(normalize_label)

    df["target"] = df[
        "target"
    ].apply(normalize_label)

    df["stance"] = df[
        "stance"
    ].apply(normalize_label)

    df["is_sarcasm_mockery"] = (
        df["is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    validate_labels(df)
    validate_experiment_settings(df)

    return df


# =====================================================================
# 4. 라벨 검사
# =====================================================================
def validate_labels(
    df: pd.DataFrame,
) -> None:
    for task in TASKS:
        blank_mask = df[task].eq("")

        if blank_mask.any():
            sample_ids = df.loc[
                blank_mask,
                "analysis_unit_id",
            ].head(10).tolist()

            raise ValueError(
                f"{task}에 빈 값이 있습니다: "
                f"{sample_ids}"
            )

        invalid_mask = ~df[task].isin(
            VALID_LABELS[task]
        )

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    task,
                ].unique()
            )

            raise ValueError(
                f"{task}에 잘못된 라벨이 있습니다: "
                f"{invalid_values}"
            )


# =====================================================================
# 5. 실험 설정 검사
# =====================================================================
def validate_experiment_settings(
    df: pd.DataFrame,
) -> None:
    if df["model"].nunique() != 1:
        models = sorted(
            df["model"].unique()
        )

        raise ValueError(
            f"서로 다른 모델 결과가 섞여 있습니다: {models}"
        )

    if df["prompt_version"].nunique() != 1:
        versions = sorted(
            df["prompt_version"].unique()
        )

        raise ValueError(
            f"서로 다른 프롬프트 버전이 섞여 있습니다: "
            f"{versions}"
        )

    if df["prompt_hash"].nunique() != 1:
        raise ValueError(
            "서로 다른 프롬프트 내용의 결과가 섞여 있습니다."
        )


# =====================================================================
# 6. 전처리 자료 불러오기
# =====================================================================
def load_texts() -> pd.DataFrame:
    if not TEXT_PATH.exists():
        raise FileNotFoundError(
            f"전처리 파일이 없습니다: {TEXT_PATH}"
        )

    df = pd.read_csv(
        TEXT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "analysis_unit_id",
        "case_id",
        "case_name",
        "source_name",
        "video_id",
        "comment_id",
        "comment_type",
        "parent_text",
        "analysis_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"전처리 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "전처리 파일에 중복 analysis_unit_id가 있습니다."
        )

    return df[
        [
            "analysis_unit_id",
            "case_id",
            "case_name",
            "source_name",
            "video_id",
            "comment_id",
            "comment_type",
            "parent_text",
            "analysis_text",
        ]
    ].copy()


# =====================================================================
# 7. 최종 결과 생성
# =====================================================================
def create_final_predictions(
    text_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    text_ids = set(
        text_df["analysis_unit_id"]
    )

    prediction_ids = set(
        prediction_df["analysis_unit_id"]
    )

    missing_ids = (
        text_ids
        - prediction_ids
    )

    extra_ids = (
        prediction_ids
        - text_ids
    )

    if missing_ids:
        raise ValueError(
            f"GPT 결과가 없는 분석 단위가 "
            f"{len(missing_ids)}개 있습니다. "
            f"예시: {sorted(missing_ids)[:10]}"
        )

    if extra_ids:
        raise ValueError(
            f"전처리 자료에 없는 GPT 결과가 "
            f"{len(extra_ids)}개 있습니다. "
            f"예시: {sorted(extra_ids)[:10]}"
        )

    prediction_columns = [
        "analysis_unit_id",
        "sentiment",
        "target",
        "stance",
        "is_sarcasm_mockery",
        "model",
        "prompt_version",
        "prompt_hash",
        "input_hash",
        "response_id",
        "started_at_utc",
        "completed_at_utc",
        "reason",
        "service_tier",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in prediction_df.columns
    ]

    prediction_df = prediction_df[
        prediction_columns
    ].copy()

    prediction_df = prediction_df.rename(
        columns={
            "sentiment": "gpt_sentiment",
            "target": "gpt_target",
            "stance": "gpt_stance",
            "is_sarcasm_mockery": (
                "gpt_is_sarcasm_mockery"
            ),
            "reason": "gpt_reason",
        }
    )

    output_df = text_df.merge(
        prediction_df,
        on="analysis_unit_id",
        how="inner",
        validate="one_to_one",
    )

    if len(output_df) != len(text_df):
        raise ValueError(
            "전처리 자료와 GPT 결과의 결합 행 수가 다릅니다."
        )

    return output_df


# =====================================================================
# 8. 결과 저장
# =====================================================================
def save_results(
    df: pd.DataFrame,
) -> None:
    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    df.to_excel(
        OUTPUT_XLSX_PATH,
        index=False,
    )


# =====================================================================
# 9. 실행
# =====================================================================
def main() -> None:
    prediction_df = load_predictions()
    text_df = load_texts()

    output_df = create_final_predictions(
        text_df,
        prediction_df,
    )

    save_results(output_df)

    print("=" * 60)
    print("GPT 최종 예측 파일 생성 완료")
    print("=" * 60)
    print(f"전체 분석 단위: {len(output_df)}개")
    print(
        f"모델: "
        f"{output_df['model'].iloc[0]}"
    )
    print(
        f"프롬프트 버전: "
        f"{output_df['prompt_version'].iloc[0]}"
    )

    for task in TASKS:
        column = f"gpt_{task}"

        print()
        print(f"{task} 분포")
        print(
            output_df[
                column
            ].value_counts(
                dropna=False
            ).to_string()
        )

    print()
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()