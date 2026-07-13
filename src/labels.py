"""
인간 코더와 GPT가 공통으로 사용하는 분류 라벨을 정의

모든 허용 라벨을 이 파일에서 한 번만 정의
"""

from enum import Enum
from typing import Type


# =====================================================================
# 1. 정서 극성 라벨
# =====================================================================
class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# =====================================================================
# 2. 대상 라벨
# =====================================================================
class TargetLabel(str, Enum):
    REFEREE = "referee"
    LEAGUE = "league"
    PLAYER = "player"
    GAME_SYSTEM = "game_system"
    NONE = "none"


# =====================================================================
# 3. 태도 라벨
# =====================================================================
class StanceLabel(str, Enum):
    BLAME = "blame"
    SUPPORT = "support"
    NEUTRAL_FACT = "neutral_fact"
    OTHER = "other"

# =====================================================================
# 4. 표현 방식 라벨
# =====================================================================
SARCASM_VALUES = [
    "true",
    "false",
]

# =====================================================================
# 5. Enum을 문자열 목록으로 변환
# =====================================================================

def enum_values(enum_class: Type[Enum]) -> list[str]:
    """
    Enum 클래스의 실제 문자열 값만 목록으로 반환
    """
    return [str(item.value) for item in enum_class]

SENTIMENT_VALUES = enum_values(SentimentLabel)
TARGET_VALUES = enum_values(TargetLabel)
STANCE_VALUES = enum_values(StanceLabel)


# =====================================================================
# 6. 허용 라벨
# =====================================================================
TASK_LABELS = {
    "sentiment": SENTIMENT_VALUES,
    "target": TARGET_VALUES,
    "stance": STANCE_VALUES,
    "is_sarcasm_mockery": SARCASM_VALUES,
}

def is_valid_label(task: str, value: str) -> bool:
    """
    주어진 값이 특정 과업에서 허용되는 라벨인지 검사
    """
    if task not in TASK_LABELS:
        raise KeyError(
            f"정의되지 않은 분류 과업입니다: {task}"
        )

    normalized_value = str(value).strip().lower()

    return normalized_value in TASK_LABELS[task]