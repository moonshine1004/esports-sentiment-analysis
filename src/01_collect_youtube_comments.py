"""
YouTube Data API v3를 이용하여 분석 대상 영상의 댓글 수집

수집 절차:
1. VIDEO_SOURCES에 입력한 영상별로 수집을 시작
2. commentThreads.list를 이용해 최상위 댓글을 수집
3. 최상위 댓글에 대댓글이 있으면 comments.list를 추가 호출
4. nextPageToken이 없을 때까지 모든 페이지를 반복
5. 수집 결과를 CSV와 Excel 파일로 저장
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    RAW_DIR,
    YOUTUBE_API_KEY,
    require_youtube_api_key,
)


# =====================================================================
# 1. 분석 대상 영상 직접 입력
# =====================================================================
VIDEO_SOURCES = [
    {
        "enabled": True,
        "case_id": "01",
        "case_name": "룬 설정 오류",
        "source_name": "마법의 거대 나르! 앙코르 파괴자 | 한화생명 vs. T1 게임2 하이라이트 | 07.13 | 2022 LCK 서머 스플릿",
        "video_id": "QzBn2S3HuiY",
    },
    {
        "enabled": True,
        "case_id": "02",
        "case_name": "강타 재사용 대기시간 오류",
        "source_name": "폴짝폴짝 | 한화생명 vs. T1 게임3 하이라이트 | 07.13 | 2022 LCK 서머 스플릿",
        "video_id": "HicGP5fLZC8",
    },
    {
        "enabled": True,
        "case_id": "03",
        "case_name": "룬 설정 오류 & 강타 재사용 대기시간 오류",
        "source_name": "한화생명 vs. T1 | 매치41 하이라이트 | 07.13 | 2022 LCK 서머 스플릿",
        "video_id": "30PbzcJYimM",
    },
]

# =====================================================================
# 2. 출력 파일 경로
# =====================================================================

# 댓글과 대댓글을 모두 포함한 원자료
RAW_CSV_PATH = RAW_DIR / "youtube_comments_raw.csv"
RAW_XLSX_PATH = RAW_DIR / "youtube_comments_raw.xlsx"

# API 요청 중 발생한 오류 기록
ERROR_PATH = RAW_DIR / "collection_errors.csv"

# 수집 날짜와 자료 수 기록
METADATA_PATH = RAW_DIR / "collection_metadata.json"


# =====================================================================
# 3. API 요청 설정
# =====================================================================
# YouTube API가 한 요청에서 반환하는 최대 댓글 수
PAGE_SIZE = 100

# =====================================================================
# 4. 영상 입력값 검사
# =====================================================================
def get_active_sources() -> list[dict[str, Any]]:
    active_sources = [
        source
        for source in VIDEO_SOURCES
        if source.get("enabled") is True
    ]

    if not active_sources:
        raise ValueError(
            "VIDEO_SOURCES에 enabled=True인 영상이 없습니다."
        )

    required_keys = {
        "source_name",
        "case_id",
        "case_name",
        "video_id",
    }

    for index, source in enumerate(
        active_sources,
        start=1,
    ):
        missing_keys = required_keys - set(source.keys())

        if missing_keys:
            raise ValueError(
                f"VIDEO_SOURCES의 {index}번째 영상에 "
                f"필수 항목이 없습니다: {sorted(missing_keys)}"
            )

        for key in required_keys:
            value = str(source.get(key, "")).strip()

            if not value:
                raise ValueError(
                    f"VIDEO_SOURCES의 {index}번째 영상에서 "
                    f"{key} 값이 비어 있습니다."
                )

    video_ids = [
        source["video_id"]
        for source in active_sources
    ]

    if len(video_ids) != len(set(video_ids)):
        raise ValueError(
            "VIDEO_SOURCES에 중복 video_id가 있습니다."
        )

    return active_sources

# =====================================================================
# 5. YouTube API 요청 재시도
# =====================================================================
# 최대 재시도 횟수
MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}

def execute_with_retry(request: Any) -> dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return request.execute()

        except HttpError as error:
            status_code = getattr(
                error.resp,
                "status",
                None,
            )

            should_retry = (
                status_code in RETRYABLE_STATUS_CODES
                and attempt < MAX_RETRIES
            )

            if not should_retry:
                raise

            # 2초, 4초, 8초, 16초 순으로 대기합니다.
            wait_seconds = 2 ** attempt

            print(
                f"일시적 API 오류: HTTP {status_code}. "
                f"{wait_seconds}초 후 재시도합니다. "
                f"({attempt}/{MAX_RETRIES})"
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "YouTube API 요청 재시도에 실패했습니다."
    )

# =====================================================================
# 6. YouTube 댓글 리소스를 데이터 행으로 변환
# =====================================================================
def parse_comment(
    comment_resource: dict[str, Any],
    *,
    source: dict[str, Any],
    thread_id: str,
    comment_type: str,
    top_level_comment_id: str,
    parent_comment_id: str,
    collected_at_utc: str,
) -> dict[str, Any]:
    """
    YouTube API에서 받은 댓글 데이터를 pandas 데이터프레임으로 변환
    """
    snippet = comment_resource.get(
        "snippet",
        {},
    )

    return {
        # -------------------------------------------------------------
        # 영상과 사례 구분 정보
        # -------------------------------------------------------------
        "source_name": source["source_name"],
        "case_id": source["case_id"],
        "case_name": source["case_name"],
        "video_id": source["video_id"],

        # -------------------------------------------------------------
        # 댓글 구조 정보
        # -------------------------------------------------------------
        "thread_id": thread_id,
        "comment_id": comment_resource.get(
            "id",
            "",
        ),
        "comment_type": comment_type,
        "top_level_comment_id": top_level_comment_id,
        "parent_comment_id": parent_comment_id,

        # -------------------------------------------------------------
        # 댓글 내용과 공개 메타데이터
        # -------------------------------------------------------------
        "text": snippet.get(
            "textDisplay",
            "",
        ),
        "like_count": snippet.get(
            "likeCount",
            0,
        ),
        "published_at": snippet.get(
            "publishedAt",
            "",
        ),
        "updated_at": snippet.get(
            "updatedAt",
            "",
        ),

        # 자료 수집 시각
        "collected_at_utc": collected_at_utc,
    }

# =====================================================================
# 7. 대댓글 수집
# =====================================================================
def fetch_all_replies(
    youtube: Any,
    *,
    source: dict[str, Any],
    thread_id: str,
    top_level_comment_id: str,
    collected_at_utc: str,
) -> list[dict[str, Any]]:
    """
    comments.list의 parentId에 댓글 ID를 넣고, nextPageToken이 없을 때까지 페이지를 반복
    """
    reply_rows: list[dict[str, Any]] = []

    page_token: str | None = None

    while True:
        request_parameters = {
            "part": "snippet",
            "parentId": top_level_comment_id,
            "maxResults": PAGE_SIZE,
            "textFormat": "plainText",
        }

        # 두 번째 페이지부터 페이지 토큰을 추가
        if page_token:
            request_parameters["pageToken"] = page_token

        request = youtube.comments().list(
            **request_parameters
        )

        response = execute_with_retry(request)

        for reply_resource in response.get(
            "items",
            [],
        ):
            reply_rows.append(
                parse_comment(
                    reply_resource,
                    source=source,
                    thread_id=thread_id,
                    comment_type="reply",
                    top_level_comment_id=top_level_comment_id,
                    parent_comment_id=top_level_comment_id,
                    collected_at_utc=collected_at_utc,
                )
            )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return reply_rows

# =====================================================================
# 8. 댓글 수집
# =====================================================================
def fetch_video_comments(
    youtube: Any,
    *,
    source: dict[str, Any],
    collected_at_utc: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    comment_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    page_token: str | None = None

    while True:
        request_parameters = {
            "part": "snippet",
            "videoId": source["video_id"],
            "maxResults": PAGE_SIZE,
            "textFormat": "plainText",
            "order": "time",
        }

        if page_token:
            request_parameters["pageToken"] = page_token

        try:
            request = youtube.commentThreads().list(
                **request_parameters
            )

            response = execute_with_retry(request)

        except HttpError as error:
            error_rows.append(
                {
                    "case_id": source["case_id"],
                    "video_id": source["video_id"],
                    "stage": "commentThreads.list",
                    "parent_comment_id": "",
                    "http_status": getattr(
                        error.resp,
                        "status",
                        "",
                    ),
                    "error_message": str(error),
                }
            )

            break

        for thread_resource in response.get(
            "items",
            [],
        ):
            thread_id = thread_resource.get(
                "id",
                "",
            )

            thread_snippet = thread_resource.get(
                "snippet",
                {},
            )

            top_level_resource = thread_snippet.get(
                "topLevelComment",
                {},
            )

            top_level_comment_id = top_level_resource.get(
                "id",
                "",
            )

            reply_count = int(
                thread_snippet.get(
                    "totalReplyCount",
                    0,
                )
            )

            comment_rows.append(
                parse_comment(
                    top_level_resource,
                    source=source,
                    thread_id=thread_id,
                    comment_type="top_level",
                    top_level_comment_id=top_level_comment_id,
                    parent_comment_id="",
                    collected_at_utc=collected_at_utc,
                )
            )

            if reply_count > 0:
                try:
                    reply_rows = fetch_all_replies(
                        youtube,
                        source=source,
                        thread_id=thread_id,
                        top_level_comment_id=top_level_comment_id,
                        collected_at_utc=collected_at_utc,
                    )

                    comment_rows.extend(
                        reply_rows
                    )

                except HttpError as error:
                    error_rows.append(
                        {
                            "case_id": source["case_id"],
                            "video_id": source["video_id"],
                            "stage": "comments.list",
                            "parent_comment_id": top_level_comment_id,
                            "http_status": getattr(
                                error.resp,
                                "status",
                                "",
                            ),
                            "error_message": str(error),
                        }
                    )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return comment_rows, error_rows

# =====================================================================
# 9. 중간 저장
# =====================================================================

def save_current_results(
    comment_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    """
    현재까지 수집한 결과를 파일로 저장합니다.

    수집 도중 프로그램이 중단되더라도
    앞선 영상의 결과를 보존하기 위한 함수입니다.
    """
    if comment_rows:
        comments_df = pd.DataFrame(
            comment_rows
        )

        # 동일 영상에서 동일 comment_id가 중복된 경우만 제거합니다.
        comments_df = comments_df.drop_duplicates(
            subset=[
                "video_id",
                "comment_id",
            ],
            keep="last",
        ).reset_index(drop=True)

        # 원자료 행을 구분하는 순차 ID를 추가합니다.
        comments_df.insert(
            0,
            "raw_row_id",
            [
                f"R{index:07d}"
                for index in range(
                    1,
                    len(comments_df) + 1,
                )
            ],
        )

        comments_df.to_csv(
            RAW_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        comments_df.to_excel(
            RAW_XLSX_PATH,
            index=False,
        )

    error_columns = [
        "case_id",
        "video_id",
        "stage",
        "parent_comment_id",
        "http_status",
        "error_message",
    ]

    error_df = pd.DataFrame(
        error_rows,
        columns=error_columns,
    )

    error_df.to_csv(
        ERROR_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# =====================================================================
# 10. 댓글 수집 실행
# =====================================================================

def main() -> None:
    """
    VIDEO_SOURCES에 등록된 영상의 댓글과 대댓글을 수집합니다.
    """
    # .env에 YouTube API 키가 있는지 검사합니다.
    require_youtube_api_key()

    # 활성화된 영상 정보와 입력값을 검사합니다.
    active_sources = get_active_sources()

    print("=" * 70)
    print("YouTube 댓글·대댓글 수집 시작")
    print("=" * 70)
    print(f"수집 대상 영상 수: {len(active_sources)}")

    # YouTube Data API 클라이언트를 생성합니다.
    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        cache_discovery=False,
    )

    # 이번 수집에서 공통으로 사용할 UTC 기준 시각입니다.
    collected_at_utc = datetime.now(
        timezone.utc
    ).isoformat()

    all_comment_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []

    for source in active_sources:
        print()
        print(
            f"[수집 시작] {source['case_id']} | "
            f"{source['source_name']}"
        )

        comment_rows, error_rows = fetch_video_comments(
            youtube,
            source=source,
            collected_at_utc=collected_at_utc,
        )

        all_comment_rows.extend(
            comment_rows
        )

        all_error_rows.extend(
            error_rows
        )

        top_level_count = sum(
            row["comment_type"] == "top_level"
            for row in comment_rows
        )

        reply_count = sum(
            row["comment_type"] == "reply"
            for row in comment_rows
        )

        print(
            f"[수집 완료] 최상위 댓글 {top_level_count}개, "
            f"대댓글 {reply_count}개"
        )

        # 영상 하나가 끝날 때마다 현재 결과를 저장
        save_current_results(
            all_comment_rows,
            all_error_rows,
        )

    if not all_comment_rows:
        raise RuntimeError(
            "수집된 댓글이 없습니다. "
            "영상 ID, API 키, 댓글 공개 여부를 확인하십시오."
        )

    final_df = pd.DataFrame(
        all_comment_rows
    ).drop_duplicates(
        subset=[
            "video_id",
            "comment_id",
        ],
        keep="last",
    )

    # 수집 결과 메타데이터를 저장합니다.
    metadata = {
        "collected_at_utc": collected_at_utc,
        "source_count": len(active_sources),
        "total_analysis_units": int(
            len(final_df)
        ),
        "top_level_comment_count": int(
            (
                final_df["comment_type"]
                == "top_level"
            ).sum()
        ),
        "reply_count": int(
            (
                final_df["comment_type"]
                == "reply"
            ).sum()
        ),
        "collection_error_count": int(
            len(all_error_rows)
        ),
        "api_methods": [
            "commentThreads.list",
            "comments.list(parentId)",
        ],
        "page_size": PAGE_SIZE,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("YouTube 댓글·대댓글 수집 완료")
    print("=" * 70)

    print(
        final_df.groupby(
            [
                "case_id",
                "comment_type",
            ]
        ).size()
    )

    print()
    print(f"전체 분석 단위: {len(final_df)}")
    print(f"원자료 CSV: {RAW_CSV_PATH}")
    print(f"원자료 Excel: {RAW_XLSX_PATH}")
    print(f"오류 기록: {ERROR_PATH}")
    print(f"수집 메타데이터: {METADATA_PATH}")


if __name__ == "__main__":
    main()