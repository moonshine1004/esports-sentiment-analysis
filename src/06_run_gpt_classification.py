"""
전처리된 댓글을 GPT로 분류
"""

from datetime import datetime, timezone
import hashlib
import time

import openai
import pandas as pd
from openai import OpenAI

from config import (
    INTERIM_DIR,
    PROMPTS_DIR,
    RESULTS_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    REPEAT_COUNT,
    PROMPT_VERSION,
    API_DELAY_SECONDS,
    OPENAI_SERVICE_TIER,
    require_openai_configuration,
)
from models import GPTClassificationResult
from text_utils import build_analysis_context


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = (
    INTERIM_DIR
    / "comments_preprocessed.csv"
)

PROMPT_PATH = (
    PROMPTS_DIR
    / "classification_prompt_v4.txt"
)

OUTPUT_CSV_PATH = (
    RESULTS_DIR
    / "gpt_predictions_runs.csv"
)

OUTPUT_XLSX_PATH = (
    RESULTS_DIR
    / "gpt_predictions_runs.xlsx"
)

# 분류 설정(0이면 전체 자료 분류)
TEST_LIMIT = 0

# =====================================================================
# 2. 데이터와 프롬프트 설정
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
        "comment_type",
        "analysis_text",
        "parent_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "중복 analysis_unit_id가 있습니다."
        )

    if TEST_LIMIT > 0:
        df = df.head(TEST_LIMIT).copy()

    return df


def load_prompt() -> tuple[str, str]:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"프롬프트 파일이 없습니다: {PROMPT_PATH}"
        )

    prompt = PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt:
        raise ValueError(
            "분류 프롬프트가 비어 있습니다."
        )

    prompt_hash = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    return prompt, prompt_hash


def make_input_text(
    row: pd.Series,
) -> str:
    if row["comment_type"] == "reply":
        return (
            f"부모 댓글: {row['parent_text']}\n"
            f"분석 대댓글: {row['analysis_text']}"
        )

    return (
        f"분석 댓글: {row['analysis_text']}"
    )

# =====================================================================
# 3. 덮어쓰기 방지
# =====================================================================
def load_existing_results(
    prompt_hash: str,
) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    if not OUTPUT_CSV_PATH.exists():
        return pd.DataFrame(), set()

    result_df = pd.read_csv(
        OUTPUT_CSV_PATH,
        dtype={
            "analysis_unit_id": str,
            "run_no": int,
        },
        keep_default_na=False,
    )

    if result_df.empty:
        return result_df, set()

    model_values = set(
        result_df["model"].unique()
    )

    prompt_hash_values = set(
        result_df["prompt_hash"].unique()
    )

    prompt_version_values = set(
        result_df["prompt_version"].unique()
    )

    if model_values != {OPENAI_MODEL}:
        raise ValueError(
            "기존 결과의 모델이 현재 설정과 다릅니다. "
            "기존 결과 파일을 보존한 뒤 새 파일로 실행하십시오."
        )

    if prompt_hash_values != {prompt_hash}:
        raise ValueError(
            "기존 결과의 프롬프트가 현재 프롬프트와 다릅니다."
        )

    if prompt_version_values != {PROMPT_VERSION}:
        raise ValueError(
            "기존 결과의 프롬프트 버전이 다릅니다."
        )

    success_df = result_df[
        result_df["status"].eq("success")
    ]

    completed = {
        (
            row["analysis_unit_id"],
            int(row["run_no"]),
        )
        for _, row in success_df.iterrows()
    }

    return result_df, completed


# =====================================================================
# 4. 결과 저장
# =====================================================================
def append_result(
    result_row: dict,
) -> None:
    result_df = pd.DataFrame(
        [result_row]
    )

    file_exists = OUTPUT_CSV_PATH.exists()

    result_df.to_csv(
        OUTPUT_CSV_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


def finalize_results() -> pd.DataFrame:
    result_df = pd.read_csv(
        OUTPUT_CSV_PATH,
        dtype={
            "analysis_unit_id": str,
            "run_no": int,
        },
        keep_default_na=False,
    )

    result_df = result_df.drop_duplicates(
        subset=[
            "analysis_unit_id",
            "run_no",
        ],
        keep="last",
    )

    result_df = result_df.sort_values(
        [
            "analysis_unit_id",
            "run_no",
        ]
    ).reset_index(drop=True)

    result_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    result_df.to_excel(
        OUTPUT_XLSX_PATH,
        index=False,
    )

    return result_df


# =====================================================================
# 5. GPT 분류
# =====================================================================
def classify_comment(
    client: OpenAI,
    prompt: str,
    prompt_hash: str,
    row: pd.Series,
    run_no: int,
) -> dict:
    input_text = make_input_text(row)

    input_hash = hashlib.sha256(
        input_text.encode("utf-8")
    ).hexdigest()

    started_at = datetime.now(
        timezone.utc
    )

    try:
        response = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            text_format=GPTClassificationResult,
            reasoning={
                "effort": "none",
            },
            max_output_tokens=180,
            service_tier=OPENAI_SERVICE_TIER,
            store=False,
        )

        result = response.output_parsed

        if result is None:
            raise ValueError(
                "구조화된 분류 결과가 없습니다."
            )

        completed_at = datetime.now(
            timezone.utc
        )

        return {
            "analysis_unit_id": row["analysis_unit_id"],
            "case_id": row["case_id"],
            "case_name": row["case_name"],
            "comment_type": row["comment_type"],
            "run_no": run_no,
            "sentiment": result.sentiment.value,
            "target": (
                result.target_attitude.target.value
            ),
            "stance": (
                result.target_attitude.stance.value
            ),
            "is_sarcasm_mockery": (
                result.is_sarcasm_mockery
            ),
            "reason": result.reason,
            "service_tier": OPENAI_SERVICE_TIER,
            "model": OPENAI_MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "response_id": response.id,
            "openai_sdk_version": openai.__version__,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "status": "success",
            "error_message": "",
        }

    except Exception as error:
        completed_at = datetime.now(
            timezone.utc
        )

        return {
            "analysis_unit_id": row["analysis_unit_id"],
            "case_id": row["case_id"],
            "case_name": row["case_name"],
            "comment_type": row["comment_type"],
            "run_no": run_no,
            "sentiment": "",
            "target": "",
            "stance": "",
            "is_sarcasm_mockery": "",
            "reason": "",
            "service_tier": OPENAI_SERVICE_TIER,
            "model": OPENAI_MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "response_id": "",
            "openai_sdk_version": openai.__version__,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "status": "error",
            "error_message": str(error),
        }


# =====================================================================
# 6. 실행
# =====================================================================

def main() -> None:
    require_openai_configuration()

    df = load_data()
    prompt, prompt_hash = load_prompt()

    _, completed = load_existing_results(
        prompt_hash
    )

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=5,
        timeout=900.0,
    )

    total_jobs = len(df) * REPEAT_COUNT
    completed_count = len(completed)
    new_count = 0

    print("=" * 60)
    print("GPT 분류 시작")
    print("=" * 60)
    print(f"분석 단위: {len(df)}개")
    print(f"반복 횟수: {REPEAT_COUNT}회")
    print(f"전체 요청 수: {total_jobs}회")
    print(f"기존 완료 수: {completed_count}회")
    print(f"모델: {OPENAI_MODEL}")

    for _, row in df.iterrows():
        for run_no in range(
            1,
            REPEAT_COUNT + 1,
        ):
            key = (
                row["analysis_unit_id"],
                run_no,
            )

            if key in completed:
                continue

            result_row = classify_comment(
                client=client,
                prompt=prompt,
                prompt_hash=prompt_hash,
                row=row,
                run_no=run_no,
            )

            append_result(
                result_row
            )

            new_count += 1

            if (
                new_count % 50 == 0
                or completed_count + new_count == total_jobs
            ):
                print(
                    f"진행: "
                    f"{completed_count + new_count}"
                    f"/{total_jobs}"
                )

            if API_DELAY_SECONDS > 0:
                time.sleep(
                    API_DELAY_SECONDS
                )

    result_df = finalize_results()

    success_count = int(
        result_df["status"]
        .eq("success")
        .sum()
    )

    error_count = int(
        result_df["status"]
        .eq("error")
        .sum()
    )

    print()
    print("=" * 60)
    print("GPT 분류 완료")
    print("=" * 60)
    print(f"성공: {success_count}건")
    print(f"오류: {error_count}건")
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Excel: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()