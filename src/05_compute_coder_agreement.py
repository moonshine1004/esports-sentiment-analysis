"""
코더 1과 코더 2의 원시 일치율과 Cohen's kappa를 계산
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import (
    HUMAN_DIR,
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

AGREEMENT_PATH = HUMAN_DIR / "coder_agreement.csv"
DISAGREEMENT_PATH = HUMAN_DIR / "coding_disagreements.xlsx"


# =====================================================================
# 2. 코딩 항목
# =====================================================================
TASK_COLUMNS = [
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
# 3. 코더 파일 불러오기
# =====================================================================
def load_coder_file(
    path,
    coder_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{coder_name} 파일이 없습니다: {path}"
        )

    df = pd.read_excel(
        path,
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
        "coder_note",
        *TASK_COLUMNS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{coder_name} 파일에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            f"{coder_name} 파일에 중복 sample_id가 있습니다."
        )

    # 라벨 표기 통일
    df["sentiment"] = df["sentiment"].apply(
        normalize_label
    )
    df["target"] = df["target"].apply(
        normalize_label
    )
    df["stance"] = df["stance"].apply(
        normalize_label
    )
    df["is_sarcasm_mockery"] = (
        df["is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    # 빈 라벨과 잘못된 라벨 검사
    for task in TASK_COLUMNS:
        blank_mask = df[task].eq("")

        if blank_mask.any():
            sample_ids = df.loc[
                blank_mask,
                "sample_id",
            ].tolist()

            raise ValueError(
                f"{coder_name}의 {task}에 빈 값이 있습니다: "
                f"{sample_ids[:10]}"
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
                f"{coder_name}의 {task}에 "
                f"잘못된 라벨이 있습니다: {invalid_values}"
            )

    return df


# =====================================================================
# 4. 코더 결과 결합
# =====================================================================
def merge_coder_results(
    coder1_df: pd.DataFrame,
    coder2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    코더 2가 코딩한 150건만 코더 1 결과와 결합합니다.
    """
    coder1_ids = set(
        coder1_df["sample_id"]
    )

    coder2_ids = set(
        coder2_df["sample_id"]
    )

    if not coder2_ids.issubset(coder1_ids):
        raise ValueError(
            "코더 2 표본에 코더 1 파일에 없는 sample_id가 있습니다."
        )

    if len(coder2_df) != CODER2_SAMPLE_SIZE:
        raise ValueError(
            "코더 2 표본 수가 설정값과 다릅니다. "
            f"현재: {len(coder2_df)}"
        )

    coder1_columns = [
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "parent_text",
        "analysis_text",
        *TASK_COLUMNS,
        "coder_note",
    ]

    coder2_columns = [
        "sample_id",
        "analysis_unit_id",
        *TASK_COLUMNS,
        "coder_note",
    ]

    coder1_df = coder1_df[
        coder1_columns
    ].copy()

    coder2_df = coder2_df[
        coder2_columns
    ].copy()

    coder1_df = coder1_df.rename(
        columns={
            task: f"coder1_{task}"
            for task in TASK_COLUMNS
        }
        | {
            "coder_note": "coder1_note",
        }
    )

    coder2_df = coder2_df.rename(
        columns={
            "analysis_unit_id": (
                "coder2_analysis_unit_id"
            ),
            **{
                task: f"coder2_{task}"
                for task in TASK_COLUMNS
            },
            "coder_note": "coder2_note",
        }
    )

    merged_df = coder1_df.merge(
        coder2_df,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged_df) != CODER2_SAMPLE_SIZE:
        raise ValueError(
            "공통 코딩 표본 수가 150개가 아닙니다."
        )

    mismatch_mask = (
        merged_df["analysis_unit_id"]
        != merged_df["coder2_analysis_unit_id"]
    )

    if mismatch_mask.any():
        raise ValueError(
            "두 코더 파일의 analysis_unit_id가 일치하지 않습니다."
        )

    merged_df = merged_df.drop(
        columns=[
            "coder2_analysis_unit_id",
        ]
    )

    return merged_df


# =====================================================================
# 5. 일치도 계산
# =====================================================================
def calculate_agreement(
    df: pd.DataFrame,
    scope: str,
    group: str,
) -> list[dict]:
    results = []

    for task in TASK_COLUMNS:
        coder1_column = f"coder1_{task}"
        coder2_column = f"coder2_{task}"

        match_mask = (
            df[coder1_column]
            == df[coder2_column]
        )

        agreement_n = int(
            match_mask.sum()
        )

        raw_agreement = (
            agreement_n / len(df)
        )

        kappa = cohen_kappa_score(
            df[coder1_column],
            df[coder2_column],
        )

        if pd.isna(kappa):
            kappa = None

        results.append(
            {
                "scope": scope,
                "group": group,
                "task": task,
                "n": len(df),
                "agreement_n": agreement_n,
                "disagreement_n": (
                    len(df) - agreement_n
                ),
                "raw_agreement": raw_agreement,
                "cohen_kappa": kappa,
            }
        )

    return results


def create_agreement_results(
    df: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    # 전체 결과
    results.extend(
        calculate_agreement(
            df,
            scope="overall",
            group="all",
        )
    )

    # 사례별 결과
    for case_id, group_df in df.groupby(
        "case_id"
    ):
        results.extend(
            calculate_agreement(
                group_df,
                scope="case_id",
                group=str(case_id),
            )
        )

    # 댓글 유형별 결과
    for comment_type, group_df in df.groupby(
        "comment_type"
    ):
        results.extend(
            calculate_agreement(
                group_df,
                scope="comment_type",
                group=str(comment_type),
            )
        )

    return pd.DataFrame(results)


# =====================================================================
# 6. 불일치 항목 생성
# =====================================================================
def create_disagreements(
    df: pd.DataFrame,
) -> pd.DataFrame:
    match_columns = []

    for task in TASK_COLUMNS:
        match_column = f"{task}_match"

        df[match_column] = (
            df[f"coder1_{task}"]
            == df[f"coder2_{task}"]
        )

        match_columns.append(
            match_column
        )

    disagreement_mask = ~df[
        match_columns
    ].all(axis=1)

    disagreement_df = df[
        disagreement_mask
    ].copy()

    output_columns = [
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "parent_text",
        "analysis_text",
        "coder1_sentiment",
        "coder2_sentiment",
        "sentiment_match",
        "coder1_target",
        "coder2_target",
        "target_match",
        "coder1_stance",
        "coder2_stance",
        "stance_match",
        "coder1_is_sarcasm_mockery",
        "coder2_is_sarcasm_mockery",
        "is_sarcasm_mockery_match",
        "coder1_note",
        "coder2_note",
    ]

    return disagreement_df[
        output_columns
    ]


# =====================================================================
# 7. 실행
# =====================================================================
def main() -> None:
    coder1_df = load_coder_file(
        CODER1_PATH,
        "코더 1",
    )

    coder2_df = load_coder_file(
        CODER2_PATH,
        "코더 2",
    )

    merged_df = merge_coder_results(
        coder1_df,
        coder2_df,
    )

    agreement_df = create_agreement_results(
        merged_df
    )

    disagreement_df = create_disagreements(
        merged_df
    )

    agreement_df.to_csv(
        AGREEMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    disagreement_df.to_excel(
        DISAGREEMENT_PATH,
        index=False,
    )

    print("=" * 60)
    print("코더 간 일치도 계산 완료")
    print("=" * 60)
    print(f"전체 표본: {len(merged_df)}개")
    print(f"불일치 표본: {len(disagreement_df)}개")

    print()
    print("전체 일치도")
    print(
        agreement_df[
            agreement_df["scope"].eq("overall")
        ][
            [
                "task",
                "n",
                "raw_agreement",
                "cohen_kappa",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"일치도 결과: {AGREEMENT_PATH}")
    print(f"불일치 항목: {DISAGREEMENT_PATH}")


if __name__ == "__main__":
    main()