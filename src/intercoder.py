"""세 코더 공통 표본의 무결성 검증과 명목척도 신뢰도 계산."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from labels import TASK_LABELS
from text_utils import normalize_boolean_text, normalize_label

ID_COLUMNS = ["sample_id", "analysis_unit_id"]
METADATA_COLUMNS = [
    "analysis_unit_id",
    "case_id",
    "case_name",
    "comment_type",
    "parent_text",
    "analysis_text",
]
TASK_COLUMNS = list(TASK_LABELS)
CODING_COLUMNS = ["sample_id"] + METADATA_COLUMNS + TASK_COLUMNS


def sha256_file(path: Path) -> str:
    """파일을 수정하지 않고 SHA-256 해시를 계산한다."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intercoder_manifest(path: Path) -> dict:
    """공통 표본 생성 스크립트가 기록한 재현성 매니페스트를 검사한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"공통 표본 매니페스트가 없습니다: {path}. "
            "먼저 04_create_coder_workbooks.py를 실행하십시오."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "coder1_sha256", "sample_size", "sample_ids"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"공통 표본 매니페스트에 필요한 항목이 없습니다: {sorted(missing)}")
    sample_ids = [str(value).strip() for value in manifest["sample_ids"]]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("공통 표본 매니페스트의 sample_ids가 비었거나 중복되었습니다.")
    if int(manifest["sample_size"]) != len(sample_ids):
        raise ValueError("공통 표본 매니페스트의 sample_size와 sample_ids 수가 다릅니다.")
    manifest["sample_ids"] = sample_ids
    manifest["sample_size"] = int(manifest["sample_size"])
    return manifest


def verify_coder1_hash(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            "coder1_coding.xlsx가 공통 표본 생성 이후 변경되었습니다. "
            f"기록 해시={expected_sha256}, 현재 해시={actual}"
        )


def load_coder(
    path: Path,
    coder_name: str,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """코더 파일의 식별자·메타데이터·라벨을 정규화하고 검사한다."""
    if not path.exists():
        raise FileNotFoundError(f"{coder_name} 파일이 없습니다: {path}")
    df = pd.read_excel(path, sheet_name="Coding", dtype=str).fillna("")
    missing = set(CODING_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{coder_name} 파일에 필요한 열이 없습니다: {sorted(missing)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{coder_name}은(는) {expected_rows}건이어야 합니다: 현재 {len(df)}건")

    df = df.copy()
    for column in CODING_COLUMNS:
        df[column] = df[column].apply(lambda value: str(value).strip())
    for column in ID_COLUMNS:
        if df[column].eq("").any():
            raise ValueError(f"{coder_name}의 {column}에 빈 값이 있습니다.")
        if df[column].duplicated().any():
            duplicate_ids = df.loc[df[column].duplicated(keep=False), column].tolist()[:10]
            raise ValueError(f"{coder_name}의 {column}에 중복 값이 있습니다: {duplicate_ids}")
    for column in ["case_id", "case_name", "comment_type", "analysis_text"]:
        if df[column].eq("").any():
            bad = df.loc[df[column].eq(""), "sample_id"].tolist()[:10]
            raise ValueError(f"{coder_name}의 {column}이 비어 있습니다: {bad}")
    invalid_comment_types = ~df["comment_type"].isin(["top_level", "reply"])
    if invalid_comment_types.any():
        bad = df.loc[invalid_comment_types, "sample_id"].tolist()[:10]
        raise ValueError(f"{coder_name}의 comment_type이 잘못되었습니다: {bad}")

    for task in TASK_COLUMNS:
        normalizer = normalize_boolean_text if task == "is_sarcasm_mockery" else normalize_label
        df[task] = df[task].apply(normalizer)
        invalid = ~df[task].isin(TASK_LABELS[task])
        if invalid.any():
            bad = df.loc[invalid, "sample_id"].tolist()[:10]
            values = sorted(df.loc[invalid, task].unique().tolist())
            raise ValueError(
                f"{coder_name}의 {task}에 빈 값 또는 잘못된 값이 있습니다: "
                f"표본={bad}, 값={values}"
            )
    return df


def align_overlap(
    coders: dict[str, pd.DataFrame],
    expected_sample_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """세 코더를 매니페스트 순서로 정렬하고 텍스트·사례 연결까지 검사한다."""
    if "coder1" not in coders:
        raise ValueError("coders에는 기준 자료인 coder1이 필요합니다.")
    if len(coders) < 2:
        raise ValueError("신뢰도 계산에는 둘 이상의 코더가 필요합니다.")

    coder1 = coders["coder1"]
    coder1_ids = set(coder1["sample_id"])
    if expected_sample_ids is None:
        other_sets = [set(df["sample_id"]) for name, df in coders.items() if name != "coder1"]
        if not other_sets or any(item_set != other_sets[0] for item_set in other_sets[1:]):
            raise ValueError("코더 2와 코더 3의 sample_id 집합이 서로 다릅니다.")
        expected_ids = sorted(other_sets[0])
    else:
        expected_ids = [str(value).strip() for value in expected_sample_ids]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected_sample_ids에 중복 값이 있습니다.")

    expected_set = set(expected_ids)
    if not expected_set:
        raise ValueError("공통 코딩 표본이 비어 있습니다.")
    missing_from_coder1 = expected_set - coder1_ids
    if missing_from_coder1:
        raise ValueError(f"공통 표본이 coder1에 없습니다: {sorted(missing_from_coder1)[:10]}")
    for name, df in coders.items():
        if name == "coder1":
            continue
        current = set(df["sample_id"])
        if current != expected_set:
            missing = sorted(expected_set - current)[:10]
            extra = sorted(current - expected_set)[:10]
            raise ValueError(f"{name}의 공통 표본이 매니페스트와 다릅니다: 누락={missing}, 추가={extra}")

    base = coder1.set_index("sample_id").loc[expected_ids]
    overlap = base[METADATA_COLUMNS].reset_index().copy()
    for name, df in coders.items():
        aligned = df.set_index("sample_id").loc[expected_ids]
        for column in METADATA_COLUMNS:
            expected_values = base[column].astype(str).to_numpy()
            actual_values = aligned[column].astype(str).to_numpy()
            if not np.array_equal(expected_values, actual_values):
                mismatch = np.flatnonzero(expected_values != actual_values)[0]
                sample_id = expected_ids[int(mismatch)]
                raise ValueError(f"{name}의 {column}이 coder1과 다릅니다: {sample_id}")
        for task in TASK_COLUMNS:
            overlap[f"{name}_{task}"] = aligned[task].to_numpy()
    return overlap


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """두 명목척도 코더의 Cohen's kappa. 분산이 없으면 NaN을 반환한다."""
    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("Cohen's kappa 입력 길이는 같고 1개 이상이어야 합니다.")
    observed = float(np.mean(left_values == right_values))
    labels = sorted(set(left_values.tolist()) | set(right_values.tolist()))
    expected = sum(
        float(np.mean(left_values == label)) * float(np.mean(right_values == label))
        for label in labels
    )
    return float("nan") if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: pd.DataFrame, labels: list[str]) -> float:
    if ratings.empty or ratings.shape[1] < 2:
        raise ValueError("Fleiss' kappa에는 항목 1개 이상과 코더 2명 이상이 필요합니다.")
    counts = np.column_stack([(ratings == label).sum(axis=1) for label in labels])
    n_raters = ratings.shape[1]
    if not np.all(counts.sum(axis=1) == n_raters):
        raise ValueError("허용 라벨에 포함되지 않은 코딩 값이 있습니다.")
    observed = float(
        (((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))).mean()
    )
    proportions = counts.sum(axis=0) / counts.sum()
    expected = float((proportions**2).sum())
    return float("nan") if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def krippendorff_alpha_nominal(ratings: pd.DataFrame) -> float:
    if ratings.empty or ratings.shape[1] < 2:
        raise ValueError("Krippendorff's alpha에는 항목 1개 이상과 코더 2명 이상이 필요합니다.")
    values = ratings.to_numpy(dtype=object)
    pairs = [(left, right) for row in values for left, right in combinations(row, 2)]
    observed_disagreement = sum(left != right for left, right in pairs) / len(pairs)
    _, counts = np.unique(values.ravel(), return_counts=True)
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    expected_disagreement = 1 - float(np.sum(counts * (counts - 1))) / (total * (total - 1))
    return (
        float("nan")
        if np.isclose(expected_disagreement, 0.0)
        else 1 - observed_disagreement / expected_disagreement
    )


def add_item_agreement_columns(overlap: pd.DataFrame, coder_names: Sequence[str]) -> pd.DataFrame:
    output = overlap.copy()
    agreement_columns = []
    for task in TASK_COLUMNS:
        coder_columns = [f"{name}_{task}" for name in coder_names]
        agreement_column = f"agreement_{task}"
        output[agreement_column] = output[coder_columns].nunique(axis=1).eq(1)
        agreement_columns.append(agreement_column)
    output["requires_adjudication"] = ~output[agreement_columns].all(axis=1)
    output["disagreement_tasks"] = output.apply(
        lambda row: ",".join(
            task for task in TASK_COLUMNS if not bool(row[f"agreement_{task}"])
        ),
        axis=1,
    )
    return output


def calculate_reliability(
    overlap: pd.DataFrame,
    coder_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    for task in TASK_COLUMNS:
        ratings = overlap[[f"{name}_{task}" for name in coder_names]]
        unanimous = ratings.nunique(axis=1).eq(1)
        fleiss = fleiss_kappa(ratings, TASK_LABELS[task])
        alpha = krippendorff_alpha_nominal(ratings)
        notes = []
        if np.isnan(fleiss):
            notes.append("Fleiss kappa undefined: no marginal label variation")
        if np.isnan(alpha):
            notes.append("Krippendorff alpha undefined: no marginal label variation")
        agreement_rate = float(unanimous.mean())
        summary_rows.append(
            {
                "task": task,
                "n_items": len(ratings),
                "n_coders": len(coder_names),
                "unanimous_agreement": agreement_rate,
                "percent_unanimous_agreement": agreement_rate * 100,
                "fleiss_kappa": fleiss,
                "krippendorff_alpha_nominal": alpha,
                "n_disagreements": int((~unanimous).sum()),
                "metric_note": "; ".join(notes),
            }
        )
        for left, right in combinations(coder_names, 2):
            y_left = overlap[f"{left}_{task}"]
            y_right = overlap[f"{right}_{task}"]
            pair_agreement = float((y_left == y_right).mean())
            kappa = cohen_kappa(y_left, y_right)
            pairwise_rows.append(
                {
                    "task": task,
                    "coder_pair": f"{left}-{right}",
                    "n_items": len(overlap),
                    "n_agreements": int((y_left == y_right).sum()),
                    "n_disagreements": int((y_left != y_right).sum()),
                    "agreement_rate": pair_agreement,
                    "percent_agreement": pair_agreement * 100,
                    "cohen_kappa": kappa,
                    "metric_note": (
                        "Cohen kappa undefined: no marginal label variation"
                        if np.isnan(kappa)
                        else ""
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(pairwise_rows)
