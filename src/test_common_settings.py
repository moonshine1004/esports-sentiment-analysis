"""
코드의 정상 작동 여부를 확인하는 테스트 파일
"""

from config import (
    BASE_DIR,
    RAW_DIR,
    INTERIM_DIR,
    HUMAN_DIR,
    RESULTS_DIR,
    REPEAT_COUNT,
    RANDOM_SEED,
    HUMAN_SAMPLE_RATIO,
    HUMAN_SAMPLE_MAX,
    API_DELAY_SECONDS,
    PROMPT_VERSION,
    YOUTUBE_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

from labels import (
    SENTIMENT_VALUES,
    TARGET_VALUES,
    STANCE_VALUES,
    SARCASM_VALUES,
    is_valid_label,
)

from models import GPTClassificationResult

from text_utils import (
    clean_comment_text,
    build_analysis_context,
)


def main() -> None:
    """
    공통 설정 파일을 차례대로 검사합니다.
    """

    print("=" * 60)
    print("1. 프로젝트 경로 확인")
    print("=" * 60)

    print("프로젝트 폴더:", BASE_DIR)
    print("원자료 폴더:", RAW_DIR)
    print("중간자료 폴더:", INTERIM_DIR)
    print("인간 코딩 폴더:", HUMAN_DIR)
    print("결과 폴더:", RESULTS_DIR)

    print()
    print("=" * 60)
    print("2. 실험 설정 확인")
    print("=" * 60)

    print("반복 횟수:", REPEAT_COUNT)
    print("랜덤 시드:", RANDOM_SEED)
    print("인간 코딩 비율:", HUMAN_SAMPLE_RATIO)
    print("인간 코딩 최대 표본:", HUMAN_SAMPLE_MAX)
    print("API 대기시간:", API_DELAY_SECONDS)
    print("프롬프트 버전:", PROMPT_VERSION)

    print()
    print("=" * 60)
    print("3. API 설정 여부 확인")
    print("=" * 60)

    # 실제 API 키는 출력하지 않습니다.
    print("YouTube API 키 설정:", bool(YOUTUBE_API_KEY))
    print("OpenAI API 키 설정:", bool(OPENAI_API_KEY))
    print("OpenAI 모델 ID:", OPENAI_MODEL or "미설정")

    print()
    print("=" * 60)
    print("4. 분류 라벨 확인")
    print("=" * 60)

    print("정서 극성:", SENTIMENT_VALUES)
    print("대상:", TARGET_VALUES)
    print("태도:", STANCE_VALUES)
    print("조롱·냉소 여부:", SARCASM_VALUES)

    print(
        "negative가 sentiment의 유효 라벨인가:",
        is_valid_label("sentiment", "negative"),
    )

    print(
        "blame이 sentiment의 유효 라벨인가:",
        is_valid_label("sentiment", "blame"),
    )

    print()
    print("=" * 60)
    print("5. 댓글 전처리 확인")
    print("=" * 60)

    sample_text = (
        "  운영진 뭐함?! ㅋㅋㅋ\n"
        "https://example.com/test  "
    )

    cleaned_text = clean_comment_text(sample_text)

    print("정제 전:", repr(sample_text))
    print("정제 후:", repr(cleaned_text))

    print()
    print("=" * 60)
    print("6. 대댓글 문맥 구성 확인")
    print("=" * 60)

    context = build_analysis_context(
        comment_type="reply",
        parent_text="운영진이 오류를 알고도 진행한 게 문제임",
        analysis_text="그러니까 역대급 운영이지 ㅋㅋ",
    )

    print(context)

    print()
    print("=" * 60)
    print("7. GPT 구조화 결과 검증")
    print("=" * 60)

    # 실제 API 호출 결과가 아니라 테스트용 가상 결과입니다.
    sample_result = GPTClassificationResult(
        sentiment="negative",
        target="league",
        stance="blame",
        is_sarcasm_mockery=True,
        model_confidence=0.85,
        rationale="리그 운영을 조롱하며 책임을 묻고 있다.",
    )

    print(sample_result.model_dump(mode="json"))

    print()
    print("=" * 60)
    print("공통 설정 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()