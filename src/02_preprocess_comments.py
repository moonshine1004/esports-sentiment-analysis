"""
수집한 댓글을 전처리
"""

import pandas as pd

from config import RAW_DIR, INTERIM_DIR
from text_utils import clean_comment_text


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = RAW_DIR / "youtube_comments_raw.csv"

OUTPUT_CSV_PATH = INTERIM_DIR / "comments_preprocessed.csv"
OUTPUT_XLSX_PATH = INTERIM_DIR / "comments_preprocessed.xlsx"


# =====================================================================
# 2. 원본 검사
# =====================================================================
def load_raw_comments() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"원자료 파일이 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "raw_row_id",
        "case_id",
        "case_name",
        "source_name",
        "video_id",
        "thread_id",
        "comment_id",
        "comment_type",
        "top_level_comment_id",
        "parent_comment_id",
        "text",
        "like_count",
        "published_at",
        "updated_at",
        "collected_at_utc",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "원자료에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    valid_comment_types = {
        "top_level",
        "reply",
    }

    invalid_types = set(
        df["comment_type"].unique()
    ) - valid_comment_types

    if invalid_types:
        raise ValueError(
            "잘못된 comment_type이 있습니다: "
            f"{sorted(invalid_types)}"
        )

    return df


# =====================================================================
# 3. 댓글 전처리
# =====================================================================
def preprocess_comments(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # 중복 제거
    df = df.drop_duplicates(
        subset=[
            "video_id",
            "comment_id",
        ],
        keep="last",
    ).copy()

    # 원문 열 이름 변경
    df = df.rename(
        columns={
            "text": "raw_text",
        }
    )

    # URL과 불필요한 공백만 제거
    df["analysis_text"] = (
        df["raw_text"]
        .apply(clean_comment_text)
    )

    # 전처리 후 빈 댓글 제거
    blank_mask = (
        df["analysis_text"]
        .str.strip()
        .eq("")
    )

    blank_count = int(blank_mask.sum())

    df = df[
        ~blank_mask
    ].copy()

    # 댓글 텍스트를 영상 ID와 댓글 ID 기준으로 저장
    top_level_df = df[
        df["comment_type"].eq("top_level")
    ]

    parent_text_map = {
        (
            row["video_id"],
            row["comment_id"],
        ): row["analysis_text"]
        for _, row in top_level_df.iterrows()
    }

    # 대댓글에 부모 댓글 텍스트 연결
    df["parent_text"] = [
        parent_text_map.get(
            (
                row["video_id"],
                row["parent_comment_id"],
            ),
            "",
        )
        if row["comment_type"] == "reply"
        else ""
        for _, row in df.iterrows()
    ]

    # 부모 댓글을 찾지 못한 대댓글 수
    missing_parent_mask = (
        df["comment_type"].eq("reply")
        & df["parent_text"].eq("")
    )

    missing_parent_count = int(
        missing_parent_mask.sum()
    )

    # 분석 단위 ID 생성
    df = df.reset_index(drop=True)

    df.insert(
        0,
        "analysis_unit_id",
        [
            f"AU{index:07d}"
            for index in range(
                1,
                len(df) + 1,
            )
        ],
    )

    print(f"빈 댓글 제거: {blank_count}개")
    print(
        "부모 댓글을 찾지 못한 대댓글: "
        f"{missing_parent_count}개"
    )

    return df


# =====================================================================
# 4. 저장
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
# 5. 실행
# =====================================================================
def main() -> None:
    df = load_raw_comments()

    print("=" * 60)
    print("댓글 전처리 시작")
    print("=" * 60)
    print(f"원자료 수: {len(df)}개")

    processed_df = preprocess_comments(
        df
    )

    save_results(
        processed_df
    )

    print()
    print("=" * 60)
    print("댓글 전처리 완료")
    print("=" * 60)
    print(f"최종 분석 단위: {len(processed_df)}개")

    print()
    print(
        processed_df.groupby(
            [
                "case_id",
                "comment_type",
            ]
        ).size()
    )

    print()
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()