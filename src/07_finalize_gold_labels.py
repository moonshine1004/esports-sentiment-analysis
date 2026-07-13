"""
합의 코딩 결과를 최종 인간 기준 라벨로 생성
"""

import pandas as pd

from config import (
    HUMAN_DIR,
    HUMAN_SAMPLE_MAX,
    CODER2_SAMPLE_SIZE,
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
CODER1_PATH = HUMAN_DIR / "coder1_coding.xlsx"
CODER2_PATH = HUMAN_DIR / "coder2_coding.xlsx"
CONSENSUS_PATH = HUMAN_DIR / "consensus_coding.xlsx"

OUTPUT_CSV_PATH = HUMAN_DIR / "gold_labels.csv"
OUTPUT_XLSX_PATH = HUMAN_DIR / "gold_labels.xlsx"


# =====================================================================
# 2. 코딩 항목
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
# 3. 라벨 정리
# =====================================================================
def normalize_task_labels(
    df: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    df = df.copy()

    for task in TASKS:
        column = f"{prefix}{task}"

        if task == "is_sarcasm_mockery":
            df[column] = df[column].apply(
                normalize_boolean_text
            )
        else:
            df[column] = df[column].apply(
                normalize_label
            )

    return df


def validate_labels(
    df: pd.DataFrame,
    prefix: str,
    file_name: str,
) -> None:
    for task in TASKS:
        column = f"{prefix}{task}"

        blank_mask = df[column].eq("")

        if blank_mask.any():
            sample_ids = df.loc[
                blank_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{file_name}의 {column}에 빈 값이 있습니다: "
                f"{sample_ids[:10]}"
            )

        invalid_mask = ~df[column].isin(
            VALID_LABELS[task]
        )

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    column,
                ].unique()
            )

            raise ValueError(
                f"{file_name}의 {column}에 "
                f"잘못된 라벨이 있습니다: {invalid_values}"
            )


# =====================================================================
# 4. 코더 1 결과
# =====================================================================
def load_coder1() -> pd.DataFrame:
    if not CODER1_PATH.exists():
        raise FileNotFoundError(
            f"코더 1 파일이 없습니다: {CODER1_PATH}"
        )

    df = pd.read_excel(
        CODER1_PATH,
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
        *TASKS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"코더 1 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != HUMAN_SAMPLE_MAX:
        raise ValueError(
            f"코더 1 표본 수가 {HUMAN_SAMPLE_MAX}개가 아닙니다. "
            f"현재: {len(df)}개"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "코더 1 파일에 중복 sample_id가 있습니다."
        )

    df = normalize_task_labels(df)

    validate_labels(
        df,
        prefix="",
        file_name="코더 1 파일",
    )

    rename_map = {
        task: f"coder1_{task}"
        for task in TASKS
    }

    return df.rename(
        columns=rename_map
    )


# =====================================================================
# 5. 코더 2 결과
# =====================================================================
def load_coder2() -> pd.DataFrame:
    if not CODER2_PATH.exists():
        raise FileNotFoundError(
            f"코더 2 파일이 없습니다: {CODER2_PATH}"
        )

    df = pd.read_excel(
        CODER2_PATH,
        sheet_name="Coding",
        dtype=str,
    ).fillna("")

    required_columns = {
        "sample_id",
        "analysis_unit_id",
        *TASKS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"코더 2 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != CODER2_SAMPLE_SIZE:
        raise ValueError(
            f"코더 2 표본 수가 {CODER2_SAMPLE_SIZE}개가 아닙니다. "
            f"현재: {len(df)}개"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "코더 2 파일에 중복 sample_id가 있습니다."
        )

    df = normalize_task_labels(df)

    validate_labels(
        df,
        prefix="",
        file_name="코더 2 파일",
    )

    rename_map = {
        "analysis_unit_id": "coder2_analysis_unit_id",
        **{
            task: f"coder2_{task}"
            for task in TASKS
        },
    }

    return df[
        [
            "sample_id",
            "analysis_unit_id",
            *TASKS,
        ]
    ].rename(
        columns=rename_map
    )


# =====================================================================
# 6. 합의 결과
# =====================================================================
def load_consensus() -> pd.DataFrame:
    if not CONSENSUS_PATH.exists():
        raise FileNotFoundError(
            f"합의 코딩 파일이 없습니다: {CONSENSUS_PATH}"
        )

    df = pd.read_excel(
        CONSENSUS_PATH,
        sheet_name="Consensus",
        dtype=str,
    ).fillna("")

    required_columns = {
        "sample_id",
        *[
            f"consensus_{task}"
            for task in TASKS
        ],
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"합의 코딩 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "합의 코딩 파일에 중복 sample_id가 있습니다."
        )

    df = normalize_task_labels(
        df,
        prefix="consensus_",
    )

    validate_labels(
        df,
        prefix="consensus_",
        file_name="합의 코딩 파일",
    )

    return df[
        [
            "sample_id",
            *[
                f"consensus_{task}"
                for task in TASKS
            ],
        ]
    ]


# =====================================================================
# 7. 최종 라벨 생성
# =====================================================================
def create_gold_labels(
    coder1_df: pd.DataFrame,
    coder2_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    coder1_ids = set(
        coder1_df["sample_id"]
    )

    coder2_ids = set(
        coder2_df["sample_id"]
    )

    if not coder2_ids.issubset(coder1_ids):
        raise ValueError(
            "코더 2 파일에 코더 1 파일에 없는 표본이 있습니다."
        )

    df = coder1_df.merge(
        coder2_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    overlap_mask = (
        df["coder2_analysis_unit_id"]
        .fillna("")
        .ne("")
    )

    analysis_id_mismatch = (
        overlap_mask
        & (
            df["analysis_unit_id"]
            != df["coder2_analysis_unit_id"]
        )
    )

    if analysis_id_mismatch.any():
        raise ValueError(
            "코더 1과 코더 2의 analysis_unit_id가 일치하지 않습니다."
        )

    df = df.merge(
        consensus_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    disagreement_mask = pd.Series(
        False,
        index=df.index,
    )

    for task in TASKS:
        coder1_column = f"coder1_{task}"
        coder2_column = f"coder2_{task}"

        task_disagreement = (
            overlap_mask
            & (
                df[coder1_column]
                != df[coder2_column]
            )
        )

        disagreement_mask |= task_disagreement

    expected_consensus_ids = set(
        df.loc[
            disagreement_mask,
            "sample_id",
        ]
    )

    actual_consensus_ids = set(
        consensus_df["sample_id"]
    )

    if expected_consensus_ids != actual_consensus_ids:
        raise ValueError(
            "합의 코딩 대상과 실제 합의 파일의 표본이 다릅니다."
        )

    for task in TASKS:
        coder1_column = f"coder1_{task}"
        coder2_column = f"coder2_{task}"
        consensus_column = f"consensus_{task}"
        gold_column = f"gold_{task}"

        task_disagreement = (
            overlap_mask
            & (
                df[coder1_column]
                != df[coder2_column]
            )
        )

        # 기본값은 코더 1 라벨
        df[gold_column] = df[
            coder1_column
        ]

        # 불일치 항목은 합의 라벨 사용
        df.loc[
            task_disagreement,
            gold_column,
        ] = df.loc[
            task_disagreement,
            consensus_column,
        ]

        invalid_mask = ~df[gold_column].isin(
            VALID_LABELS[task]
        )

        if invalid_mask.any():
            sample_ids = df.loc[
                invalid_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{gold_column}에 잘못된 값이 있습니다: "
                f"{sample_ids[:10]}"
            )

    return df


# =====================================================================
# 8. 저장
# =====================================================================
def save_results(
    df: pd.DataFrame,
) -> None:
    output_columns = [
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
    ]

    output_df = df[
        output_columns
    ].copy()

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
# 9. 실행
# =====================================================================
def main() -> None:
    coder1_df = load_coder1()
    coder2_df = load_coder2()
    consensus_df = load_consensus()

    gold_df = create_gold_labels(
        coder1_df,
        coder2_df,
        consensus_df,
    )

    save_results(
        gold_df
    )

    print("=" * 60)
    print("최종 인간 기준 라벨 생성 완료")
    print("=" * 60)
    print(f"전체 기준 라벨: {len(gold_df)}개")
    print(f"코더 2 공통 표본: {CODER2_SAMPLE_SIZE}개")
    print(f"합의 표본: {len(consensus_df)}개")
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()