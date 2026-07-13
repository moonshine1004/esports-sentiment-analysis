"""
GPT 구조화 출력의 데이터 형식 정의
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

from pydantic import BaseModel, ConfigDict

from labels import (
    SentimentLabel,
    TargetLabel,
    StanceLabel,
)

class GPTClassificationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    sentiment: SentimentLabel
    target: TargetLabel
    stance: StanceLabel
    is_sarcasm_mockery: bool


CLASSIFICATION_SCHEMA = (
    GPTClassificationResult.model_json_schema()
)