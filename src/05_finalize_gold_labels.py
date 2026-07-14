"""
인간 코더가 작성한 코딩 결과를 검사
최종 인간 기준 라벨 파일을 생성
"""

import pandas as pd

from config import (
    HUMAN_DIR,
    HUMAN_SAMPLE_MAX,
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
CODER_PATH = (
    HUMAN_DIR
    / "coder1_coding.xlsx"
)

# 최종 인간 기준 라벨
OUTPUT_CSV_PATH = (
    HUMAN_DIR
    / "gold_labels.csv"
)

OUTPUT_XLSX_PATH = (
    HUMAN_DIR
    / "gold_labels.xlsx"
)

# =====================================================================
# 2. 코딩 항목과 허용 라벨
# =====================================================================
TASK_CONFIG = {
    "sentiment": SENTIMENT_VALUES,
    "target": TARGET_VALUES,
    "stance": STANCE_VALUES,
    "is_sarcasm_mockery": SARCASM_VALUES,
}


# =====================================================================
# 3. 인간 코딩 파일 불러오기
# =====================================================================
def load_coding() -> pd.DataFrame:
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

    if len(df) != HUMAN_SAMPLE_MAX:
        raise ValueError(
            f"인간 코딩 표본 수가 "
            f"{HUMAN_SAMPLE_MAX}개가 아닙니다. "
            f"현재: {len(df)}개"
        )

    if df["sample_id"].duplicated().any():
        duplicate_ids = (
            df.loc[
                df["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "중복 sample_id가 있습니다: "
            f"{duplicate_ids[:10]}"
        )

    if df["analysis_unit_id"].duplicated().any():
        duplicate_ids = (
            df.loc[
                df[
                    "analysis_unit_id"
                ].duplicated(
                    keep=False
                ),
                "analysis_unit_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "중복 analysis_unit_id가 있습니다: "
            f"{duplicate_ids[:10]}"
        )

    return df


# =====================================================================
# 4. 라벨 표기 정규화
# =====================================================================
def normalize_coding(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

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

    return df


# =====================================================================
# 5. 코딩 완료 여부와 라벨 검사
# =====================================================================
def validate_labels(
    df: pd.DataFrame,
) -> None:
    for column, valid_values in (
        TASK_CONFIG.items()
    ):
        # 빈 셀 검사
        blank_mask = df[column].eq("")

        if blank_mask.any():
            sample_ids = df.loc[
                blank_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{column}에 빈 값이 있습니다: "
                f"{sample_ids[:10]}"
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

            invalid_sample_ids = df.loc[
                invalid_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{column}에 잘못된 라벨이 있습니다. "
                f"값: {invalid_values}, "
                f"표본: {invalid_sample_ids[:10]}"
            )


# =====================================================================
# 6. 최종 인간 기준 라벨 생성
# =====================================================================
def create_reference_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    인간 코딩 라벨에 gold_ 접두사를 붙여
    GPT 결과와 비교할 수 있는 형태로 만듭니다.
    """

    output_columns = [
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
    ]

    output_df = df[
        output_columns
    ].copy()

    output_df = output_df.rename(
        columns={
            "sentiment": (
                "gold_sentiment"
            ),
            "target": (
                "gold_target"
            ),
            "stance": (
                "gold_stance"
            ),
            "is_sarcasm_mockery": (
                "gold_is_sarcasm_mockery"
            ),
        }
    )

    return output_df


# =====================================================================
# 7. 결과 저장
# =====================================================================
def save_results(
    output_df: pd.DataFrame,
) -> None:
    """
    최종 인간 기준 라벨을 CSV와 Excel로 저장합니다.
    """

    output_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    output_df.to_excel(
        OUTPUT_XLSX_PATH,
        index=False,
    )


# =====================================================================
# 8. 실행
# =====================================================================
def main() -> None:
    # 인간 코딩 파일 불러오기
    coding_df = load_coding()

    # 라벨 표기 정리
    coding_df = normalize_coding(
        coding_df
    )

    # 누락과 잘못된 라벨 검사
    validate_labels(
        coding_df
    )

    # GPT 평가용 인간 기준 라벨 생성
    output_df = create_reference_labels(
        coding_df
    )

    # 결과 저장
    save_results(
        output_df
    )

    print("=" * 60)
    print("인간 기준 라벨 생성 완료")
    print("=" * 60)
    print(f"표본 수: {len(output_df)}개")
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()