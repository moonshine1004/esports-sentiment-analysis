"""
프로젝트 전체에서 공통으로 사용하는 설정을 정의
1. 프로젝트의 폴더 경로
2. .env 파일의 환경변수
3. GPT 반복 횟수와 인간 코딩 표본 설wjd
4. API 키 설정 여부 검사
"""

from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]


# =====================================================================
# 1. 프로젝트 기본 경로 설정
# =====================================================================
DATA_DIR = BASE_DIR / "data"

# 프로젝트 내부의 주요 폴더 경로 정의
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
HUMAN_DIR = DATA_DIR / "human_coding"
RESULTS_DIR = DATA_DIR / "results"

DOCS_DIR = BASE_DIR / "docs"
PROMPTS_DIR = BASE_DIR / "prompts"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# 필요한 폴더가 없으면 자동으로 생성
for directory in [
    RAW_DIR,
    INTERIM_DIR,
    HUMAN_DIR,
    RESULTS_DIR,
    DOCS_DIR,
    PROMPTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. .env 파일 불러오기
# =====================================================================
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)


# =====================================================================
# 3. 환경변수 읽기
# =====================================================================

def read_string_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)

    return str(value).strip()


def read_int_env(name: str, default: int) -> int:
    raw_value = read_string_env(name, str(default))

    try:
        return int(raw_value)

    except ValueError as error:
        raise ValueError(
            f"{name} 값은 정수여야 합니다. 현재 값: {raw_value}"
        ) from error


def read_float_env(name: str, default: float) -> float:
    raw_value = read_string_env(name, str(default))

    try:
        return float(raw_value)

    except ValueError as error:
        raise ValueError(
            f"{name} 값은 숫자여야 합니다. 현재 값: {raw_value}"
        ) from error


# =====================================================================
# 4. API 관련 설정
# =====================================================================
YOUTUBE_API_KEY = read_string_env("YOUTUBE_API_KEY")
OPENAI_API_KEY = read_string_env("OPENAI_API_KEY")

OPENAI_MODEL = read_string_env("OPENAI_MODEL")

# =====================================================================
# 5. 실험 관련 설정
# =====================================================================

# 동일 댓글을 GPT로 반복 분류할 횟수
REPEAT_COUNT = read_int_env(
    "REPEAT_COUNT",
    default=5,
)

# 인간 코딩 표본 추출 등 무작위 처리의 재현성을 위한 값
RANDOM_SEED = read_int_env(
    "RANDOM_SEED",
    default=42,
)

# 전체 자료 중 인간 코딩 표본으로 사용할 비율
HUMAN_SAMPLE_RATIO = read_float_env(
    "HUMAN_SAMPLE_RATIO",
    default=0.20,
)

# 인간 코딩 표본의 최대 개수
HUMAN_SAMPLE_MAX = read_int_env(
    "HUMAN_SAMPLE_MAX",
    default=500,
)

# 코더 2의 표본 개수
CODER2_SAMPLE_SIZE = read_int_env(
    "CODER2_SAMPLE_SIZE",
    default=150,
)

# 연속 API 호출 사이에 기다릴 기본 시간
API_DELAY_SECONDS = read_float_env(
    "API_DELAY_SECONDS",
    default=0.5,
)

# 프롬프트 변경 이력을 구분하기 위한 버전
PROMPT_VERSION = read_string_env(
    "PROMPT_VERSION",
    default="v1.0",
)

# =====================================================================
# 6. 설정값 검증
# =====================================================================
def validate_common_settings() -> None:
    if REPEAT_COUNT < 1:
        raise ValueError(
            "REPEAT_COUNT는 1 이상이어야 합니다."
        )

    if not 0 < HUMAN_SAMPLE_RATIO <= 1:
        raise ValueError(
            "HUMAN_SAMPLE_RATIO는 0보다 크고 1 이하여야 합니다."
        )

    if HUMAN_SAMPLE_MAX < 1:
        raise ValueError(
            "HUMAN_SAMPLE_MAX는 1 이상이어야 합니다."
        )
    
    if CODER2_SAMPLE_SIZE < 1:
        raise ValueError(
            "CODER2_SAMPLE_SIZE는 1 이상이어야 합니다."
        )

    if CODER2_SAMPLE_SIZE > HUMAN_SAMPLE_MAX:
        raise ValueError(
            "CODER2_SAMPLE_SIZE는 HUMAN_SAMPLE_MAX보다 클 수 없습니다."
        )

    if API_DELAY_SECONDS < 0:
        raise ValueError(
            "API_DELAY_SECONDS는 0 이상이어야 합니다."
        )


def require_youtube_api_key() -> None:
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY가 설정되지 않았습니다. "
            "프로젝트 최상위 폴더의 .env 파일을 확인하십시오."
        )

def require_openai_configuration() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "프로젝트 최상위 폴더의 .env 파일을 확인하십시오."
        )

    if not OPENAI_MODEL:
        raise RuntimeError(
            "OPENAI_MODEL이 설정되지 않았습니다. "
            ".env 파일에 실제 사용할 모델 ID를 입력하십시오."
        )

validate_common_settings()