"""
전처리된 댓글에서 인간 코딩용 표본을 추출
사례와 댓글 유형을 기준으로 비례 표본을 추출
"""

import math

import pandas as pd

from config import (
    HUMAN_DIR,
    HUMAN_SAMPLE_MAX,
    CODER2_SAMPLE_SIZE,
    RANDOM_SEED,
    INTERIM_DIR,
)


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = (
    INTERIM_DIR
    / "comments_preprocessed.csv"
)

OUTPUT_CSV_PATH = (
    HUMAN_DIR
    / "human_sample_master.csv"
)

OUTPUT_XLSX_PATH = (
    HUMAN_DIR
    / "human_sample_master.xlsx"
)

# 표본 기준
GROUP_COLUMNS = [
    "case_id",
    "comment_type",
]

# =====================================================================
# 2. 데이터 불러오기
# =====================================================================
def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"전처리 파일이 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
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
        "top_level_comment_id",
        "parent_comment_id",
        "raw_text",
        "analysis_text",
        "parent_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "중복 analysis_unit_id가 있습니다."
        )

    return df


# =====================================================================
# 3. 층별 표본 수 계산
# =====================================================================
def calculate_sample_counts(
    df: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    group_sizes = (
        df.groupby(
            GROUP_COLUMNS
        )
        .size()
        .reset_index(
            name="group_size"
        )
        .sort_values(
            GROUP_COLUMNS
        )
        .reset_index(drop=True)
    )

    group_sizes["sample_n"] = 0

    group_count = len(group_sizes)

    if sample_size >= group_count:
        group_sizes["sample_n"] = 1

    else:
        largest_groups = (
            group_sizes
            .sort_values(
                "group_size",
                ascending=False,
            )
            .head(sample_size)
            .index
        )

        group_sizes.loc[
            largest_groups,
            "sample_n",
        ] = 1

    remaining = (
        sample_size
        - int(
            group_sizes["sample_n"].sum()
        )
    )

    if remaining <= 0:
        return group_sizes

    group_sizes["capacity"] = (
        group_sizes["group_size"]
        - group_sizes["sample_n"]
    )

    total_capacity = int(
        group_sizes["capacity"].sum()
    )

    group_sizes["quota"] = (
        group_sizes["capacity"]
        / total_capacity
        * remaining
    )

    group_sizes["additional_n"] = (
        group_sizes["quota"]
        .apply(math.floor)
        .astype(int)
    )

    group_sizes["sample_n"] += (
        group_sizes["additional_n"]
    )

    leftover = (
        sample_size
        - int(
            group_sizes["sample_n"].sum()
        )
    )

    group_sizes["remainder"] = (
        group_sizes["quota"]
        - group_sizes["additional_n"]
    )

    if leftover > 0:
        available = group_sizes[
            group_sizes["sample_n"]
            < group_sizes["group_size"]
        ].sort_values(
            [
                "remainder",
                "group_size",
            ],
            ascending=False,
        )

        for index in available.index[
            :leftover
        ]:
            group_sizes.loc[
                index,
                "sample_n",
            ] += 1

    return group_sizes


# =====================================================================
# 4. 표본 추출
# =====================================================================
def draw_stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    seed_offset: int,
) -> pd.DataFrame:
    """
    사례와 댓글 유형에 따라 층화 표본 추출
    """
    sample_counts = calculate_sample_counts(
        df,
        sample_size,
    )

    sampled_groups = []

    for index, group in sample_counts.iterrows():
        group_df = df.copy()

        for column in GROUP_COLUMNS:
            group_df = group_df[
                group_df[column].eq(
                    group[column]
                )
            ]

        sample_n = int(group["sample_n"])

        if sample_n == 0:
            continue

        group_sample = group_df.sample(
            n=sample_n,
            random_state=(
                RANDOM_SEED
                + seed_offset
                + index
            ),
        )

        sampled_groups.append(
            group_sample
        )

    return pd.concat(
        sampled_groups,
        ignore_index=True,
    )


def sample_comments(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    코더 1용 500건과 코더 2용 공통 표본 150건을 선정
    """
    coder1_sample_size = min(
        HUMAN_SAMPLE_MAX,
        len(df),
    )

    sampled_df = draw_stratified_sample(
        df,
        coder1_sample_size,
        seed_offset=0,
    )

    sampled_df = sampled_df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    coder2_sample_size = min(
        CODER2_SAMPLE_SIZE,
        len(sampled_df),
    )

    coder2_df = draw_stratified_sample(
        sampled_df,
        coder2_sample_size,
        seed_offset=1000,
    )

    coder2_ids = set(
        coder2_df["analysis_unit_id"]
    )

    sampled_df["coder2_overlap"] = (
        sampled_df["analysis_unit_id"]
        .isin(coder2_ids)
    )

    sampled_df.insert(
        0,
        "sample_id",
        [
            f"S{index:04d}"
            for index in range(
                1,
                len(sampled_df) + 1,
            )
        ],
    )

    return sampled_df

# =====================================================================
# 5. 저장
# =====================================================================
def save_results(
    df: pd.DataFrame,
) -> None:
    output_columns = [
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "source_name",
        "video_id",
        "comment_id",
        "comment_type",
        "top_level_comment_id",
        "parent_comment_id",
        "raw_text",
        "analysis_text",
        "parent_text",
        "coder2_overlap",
    ]

    df = df[
        output_columns
    ]

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
# 6. 실행
# =====================================================================
def main() -> None:
    df = load_data()

    print("=" * 60)
    print("인간 코딩 표본 추출 시작")
    print("=" * 60)
    print(f"전체 분석 단위: {len(df)}개")

    sampled_df = sample_comments(
        df
    )

    save_results(
        sampled_df
    )

    sample_ratio = (
        len(sampled_df)
        / len(df)
        * 100
    )

    print()
    print("=" * 60)
    print("인간 코딩 표본 추출 완료")
    print("=" * 60)
    print(f"추출 표본: {len(sampled_df)}개")

    coder2_count = int(
        sampled_df["coder2_overlap"].sum()
    )
    print(f"코더 2 공통 표본: {coder2_count}개")
    
    print(f"전체 자료 대비 비율: {sample_ratio:.2f}%")

    print()
    print(
        sampled_df.groupby(
            GROUP_COLUMNS
        ).size()
    )

    print()
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()