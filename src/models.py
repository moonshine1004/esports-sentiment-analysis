"""
GPT 구조화 출력 스키마
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from labels import (
    SentimentLabel,
    TargetLabel,
    StanceLabel,
)


class TargetAttitudeResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    target: TargetLabel
    stance: StanceLabel


class GPTClassificationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    sentiment: SentimentLabel

    target_attitude: TargetAttitudeResult

    is_sarcasm_mockery: bool

    reason: str = Field(
        min_length=1,
        max_length=160,
    )


CLASSIFICATION_SCHEMA = (
    GPTClassificationResult.model_json_schema()
)