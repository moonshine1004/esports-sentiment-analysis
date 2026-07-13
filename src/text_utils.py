"""
댓글과 대댓글 텍스트 처리에 사용하는 공통 함수
URL, HTML 표기, 불필요한 공백만 정리

다음 요소는 정서와 표현 방식의 단서가 될 수 있으므로 삭제하지 않음
- ㅋㅋㅋ, ㅎㅎㅎ
- ㅠㅠ, ㅜㅜ
- 느낌표와 물음표
- 반복 문자
- 이모지
- 비속어
- 밈적 표현
"""

from typing import Any
import html
import re

import pandas as pd


# =====================================================================
# 1. 정규표현식 패턴
# =====================================================================
# http://, https:// 또는 www.로 시작하는 URL을 찾음
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

# 줄바꿈, 탭, 연속 공백을 찾음
WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

# =====================================================================
# 2. 결측값 확인
# =====================================================================
def is_missing_value(value: Any) -> bool:
    """
    값이 None 또는 pandas의 결측값인지 검사
    """
    if value is None:
        return True

    try:
        return bool(pd.isna(value))

    except (TypeError, ValueError):
        return False


# =====================================================================
# 3. 댓글 텍스트 정제
# =====================================================================
def clean_comment_text(value: Any) -> str:
    if is_missing_value(value):
        return ""

    text = str(value)

    # &amp;를 &, &quot;를 "처럼 실제 문자로 복원
    text = html.unescape(text)
    # URL을 공백으로 대체
    text = URL_PATTERN.sub(" ", text)
    # 줄바꿈과 여러 개의 공백을 하나의 공백으로 대체
    text = WHITESPACE_PATTERN.sub(" ", text)
    # 문자열 앞뒤의 공백을 제거
    return text.strip()


# =====================================================================
# 4. 라벨 문자열 정규화
# =====================================================================
def normalize_label(value: Any) -> str:
    if is_missing_value(value):
        return ""

    return str(value).strip().lower()

# 문자열 표기 통일
def normalize_boolean_text(value: Any) -> str:
    normalized = normalize_label(value)

    true_values = {
        "true",
        "1",
        "yes",
        "y",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
    }

    if normalized in true_values:
        return "true"

    if normalized in false_values:
        return "false"

    return normalized


# =====================================================================
# 5. 최상위 댓글과 대댓글 문맥 구성
# =====================================================================

def build_analysis_context(
    comment_type: str,
    parent_text: Any,
    analysis_text: Any,
) -> str:
    normalized_type = normalize_label(comment_type)

    cleaned_parent = clean_comment_text(parent_text)
    cleaned_analysis = clean_comment_text(analysis_text)

    if normalized_type == "reply":
        return (
            "[댓글 유형]\n"
            "대댓글(reply)\n\n"
            "[부모 댓글 문맥]\n"
            f"{cleaned_parent}\n\n"
            "[실제 분석 대상 대댓글]\n"
            f"{cleaned_analysis}"
        )

    return (
        "[댓글 유형]\n"
        "최상위 댓글(top_level)\n\n"
        "[부모 댓글 문맥]\n"
        "없음\n\n"
        "[실제 분석 대상 댓글]\n"
        f"{cleaned_analysis}"
    )