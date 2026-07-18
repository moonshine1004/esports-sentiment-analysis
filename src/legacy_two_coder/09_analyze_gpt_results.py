"""
GPT 분류 결과의 분포와 사례 간 차이를 분석
"""

import math

import pandas as pd
from scipy.stats import chi2_contingency

from config import RESULTS_DIR
from labels import (
    SENTIMENT_VALUES,
    TARGET_VALUES,
    STANCE_VALUES,
    SARCASM_VALUES,
)
from text_utils import (
    normalize_label,
    normalize_boolean_text,
)


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.csv"
)

DISTRIBUTION_PATH = (
    RESULTS_DIR
    / "gpt_label_distribution.csv"
)

CROSSTAB_PATH = (
    RESULTS_DIR
    / "gpt_case_crosstab.csv"
)

CHI_SQUARE_PATH = (
    RESULTS_DIR
    / "gpt_case_chi_square.csv"
)

RESIDUAL_PATH = (
    RESULTS_DIR
    / "gpt_case_residuals.csv"
)

EXCEL_PATH = (
    RESULTS_DIR
    / "gpt_descriptive_analysis.xlsx"
)


# =====================================================================
# 2. 분석 과업
# =====================================================================
TASK_CONFIG = {
    "sentiment": {
        "column": "gpt_sentiment",
        "labels": SENTIMENT_VALUES,
    },
    "target": {
        "column": "gpt_target",
        "labels": TARGET_VALUES,
    },
    "stance": {
        "column": "gpt_stance",
        "labels": STANCE_VALUES,
    },
    "is_sarcasm_mockery": {
        "column": "gpt_is_sarcasm_mockery",
        "labels": SARCASM_VALUES,
    },
}


# =====================================================================
# 3. 자료 불러오기
# =====================================================================
def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"GPT 최종 예측 파일이 없습니다: {INPUT_PATH}"
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
        "gpt_sentiment",
        "gpt_target",
        "gpt_stance",
        "gpt_is_sarcasm_mockery",
        "model",
        "prompt_version",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    if df["analysis_unit_id"].duplicated().any():
        raise ValueError(
            "중복 analysis_unit_id가 있습니다."
        )

    df["gpt_sentiment"] = (
        df["gpt_sentiment"]
        .apply(normalize_label)
    )

    df["gpt_target"] = (
        df["gpt_target"]
        .apply(normalize_label)
    )

    df["gpt_stance"] = (
        df["gpt_stance"]
        .apply(normalize_label)
    )

    df["gpt_is_sarcasm_mockery"] = (
        df["gpt_is_sarcasm_mockery"]
        .apply(normalize_boolean_text)
    )

    validate_data(df)

    return df


# =====================================================================
# 4. 자료 검사
# =====================================================================
def validate_data(
    df: pd.DataFrame,
) -> None:
    for task, config in TASK_CONFIG.items():
        column = config["column"]
        valid_labels = config["labels"]

        blank_mask = df[column].eq("")

        if blank_mask.any():
            ids = df.loc[
                blank_mask,
                "analysis_unit_id",
            ].head(10).tolist()

            raise ValueError(
                f"{column}에 빈 값이 있습니다: {ids}"
            )

        invalid_mask = ~df[column].isin(
            valid_labels
        )

        if invalid_mask.any():
            invalid_values = sorted(
                df.loc[
                    invalid_mask,
                    column,
                ].unique()
            )

            raise ValueError(
                f"{column}에 잘못된 라벨이 있습니다: "
                f"{invalid_values}"
            )

    if df["model"].nunique() != 1:
        raise ValueError(
            "서로 다른 모델의 결과가 섞여 있습니다."
        )

    if df["prompt_version"].nunique() != 1:
        raise ValueError(
            "서로 다른 프롬프트 버전의 결과가 섞여 있습니다."
        )


# =====================================================================
# 5. 빈도와 비율
# =====================================================================
def calculate_distribution(
    df: pd.DataFrame,
    scope: str,
    group: str,
    task: str,
    column: str,
    labels: list[str],
) -> list[dict]:
    counts = (
        df[column]
        .value_counts()
        .reindex(
            labels,
            fill_value=0,
        )
    )

    total_n = len(df)
    rows = []

    for label in labels:
        count = int(
            counts[label]
        )

        proportion = (
            count / total_n
            if total_n > 0
            else 0.0
        )

        rows.append(
            {
                "scope": scope,
                "group": group,
                "task": task,
                "label": label,
                "count": count,
                "proportion": proportion,
                "total_n": total_n,
            }
        )

    return rows


def create_distribution_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for task, config in TASK_CONFIG.items():
        rows.extend(
            calculate_distribution(
                df=df,
                scope="overall",
                group="all",
                task=task,
                column=config["column"],
                labels=config["labels"],
            )
        )

    for case_id, group_df in df.groupby(
        "case_id",
        sort=True,
    ):
        case_name = group_df[
            "case_name"
        ].iloc[0]

        group_name = (
            f"{case_id}: {case_name}"
        )

        for task, config in TASK_CONFIG.items():
            rows.extend(
                calculate_distribution(
                    df=group_df,
                    scope="case_id",
                    group=group_name,
                    task=task,
                    column=config["column"],
                    labels=config["labels"],
                )
            )

    for comment_type, group_df in df.groupby(
        "comment_type",
        sort=True,
    ):
        for task, config in TASK_CONFIG.items():
            rows.extend(
                calculate_distribution(
                    df=group_df,
                    scope="comment_type",
                    group=str(comment_type),
                    task=task,
                    column=config["column"],
                    labels=config["labels"],
                )
            )

    return pd.DataFrame(rows)


# =====================================================================
# 6. 사례별 교차표
# =====================================================================
def create_case_crosstabs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    case_names = (
        df[
            [
                "case_id",
                "case_name",
            ]
        ]
        .drop_duplicates()
        .set_index("case_id")[
            "case_name"
        ]
        .to_dict()
    )

    for task, config in TASK_CONFIG.items():
        column = config["column"]
        labels = config["labels"]

        count_table = pd.crosstab(
            df["case_id"],
            df[column],
        ).reindex(
            columns=labels,
            fill_value=0,
        )

        row_totals = count_table.sum(
            axis=1
        )

        for case_id in count_table.index:
            for label in labels:
                count = int(
                    count_table.loc[
                        case_id,
                        label,
                    ]
                )

                total_n = int(
                    row_totals.loc[
                        case_id
                    ]
                )

                proportion = (
                    count / total_n
                    if total_n > 0
                    else 0.0
                )

                rows.append(
                    {
                        "task": task,
                        "case_id": case_id,
                        "case_name": case_names[
                            case_id
                        ],
                        "label": label,
                        "count": count,
                        "row_proportion": proportion,
                        "case_total_n": total_n,
                    }
                )

    return pd.DataFrame(rows)


# =====================================================================
# 7. Cramér's V
# =====================================================================
def calculate_cramers_v(
    chi_square: float,
    total_n: int,
    row_count: int,
    column_count: int,
) -> float:
    denominator = (
        total_n
        * min(
            row_count - 1,
            column_count - 1,
        )
    )

    if denominator <= 0:
        return 0.0

    return math.sqrt(
        chi_square / denominator
    )


# =====================================================================
# 8. 조정 표준화 잔차
# =====================================================================
def calculate_adjusted_residuals(
    observed: pd.DataFrame,
    expected,
    task: str,
) -> list[dict]:
    observed_values = observed.to_numpy()

    row_totals = observed_values.sum(
        axis=1
    )

    column_totals = observed_values.sum(
        axis=0
    )

    total_n = observed_values.sum()
    rows = []

    for row_index, case_id in enumerate(
        observed.index
    ):
        for column_index, label in enumerate(
            observed.columns
        ):
            observed_count = observed_values[
                row_index,
                column_index,
            ]

            expected_count = expected[
                row_index,
                column_index,
            ]

            row_proportion = (
                row_totals[row_index]
                / total_n
            )

            column_proportion = (
                column_totals[column_index]
                / total_n
            )

            denominator = math.sqrt(
                expected_count
                * (1 - row_proportion)
                * (1 - column_proportion)
            )

            if denominator == 0:
                residual = 0.0
            else:
                residual = (
                    observed_count
                    - expected_count
                ) / denominator

            rows.append(
                {
                    "task": task,
                    "case_id": case_id,
                    "label": label,
                    "observed_count": int(
                        observed_count
                    ),
                    "expected_count": (
                        expected_count
                    ),
                    "adjusted_standardized_residual": (
                        residual
                    ),
                    "absolute_residual": abs(
                        residual
                    ),
                    "residual_over_1_96": (
                        abs(residual) >= 1.96
                    ),
                }
            )

    return rows


# =====================================================================
# 9. 사례 간 카이제곱 검정
# =====================================================================
def create_chi_square_results(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_rows = []
    residual_rows = []

    for task, config in TASK_CONFIG.items():
        column = config["column"]
        labels = config["labels"]

        observed = pd.crosstab(
            df["case_id"],
            df[column],
        ).reindex(
            columns=labels,
            fill_value=0,
        )

        # 전체 빈도가 0인 라벨은 검정에서 제외
        active_columns = (
            observed.sum(axis=0)
            .gt(0)
        )

        test_table = observed.loc[
            :,
            active_columns,
        ]

        if (
            test_table.shape[0] < 2
            or test_table.shape[1] < 2
        ):
            summary_rows.append(
                {
                    "task": task,
                    "n": int(
                        test_table.to_numpy().sum()
                    ),
                    "case_count": (
                        test_table.shape[0]
                    ),
                    "class_count": (
                        test_table.shape[1]
                    ),
                    "chi_square": "",
                    "degrees_of_freedom": "",
                    "p_value": "",
                    "cramers_v": "",
                    "min_expected_count": "",
                    "expected_under_5_cells": "",
                    "expected_under_5_ratio": "",
                }
            )

            continue

        (
            chi_square,
            p_value,
            degrees_of_freedom,
            expected,
        ) = chi2_contingency(
            test_table,
            correction=False,
        )

        total_n = int(
            test_table.to_numpy().sum()
        )

        cramers_v = calculate_cramers_v(
            chi_square=chi_square,
            total_n=total_n,
            row_count=test_table.shape[0],
            column_count=test_table.shape[1],
        )

        expected_under_5 = int(
            (expected < 5).sum()
        )

        expected_cell_count = int(
            expected.size
        )

        expected_under_5_ratio = (
            expected_under_5
            / expected_cell_count
        )

        summary_rows.append(
            {
                "task": task,
                "n": total_n,
                "case_count": (
                    test_table.shape[0]
                ),
                "class_count": (
                    test_table.shape[1]
                ),
                "chi_square": chi_square,
                "degrees_of_freedom": (
                    degrees_of_freedom
                ),
                "p_value": p_value,
                "cramers_v": cramers_v,
                "min_expected_count": (
                    float(expected.min())
                ),
                "expected_under_5_cells": (
                    expected_under_5
                ),
                "expected_under_5_ratio": (
                    expected_under_5_ratio
                ),
            }
        )

        residual_rows.extend(
            calculate_adjusted_residuals(
                observed=test_table,
                expected=expected,
                task=task,
            )
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = add_bh_adjusted_p_values(
        summary_df
    )

    residual_df = pd.DataFrame(
        residual_rows
    )

    return (
        summary_df,
        residual_df,
    )


# =====================================================================
# 10. 다중검정 보정
# =====================================================================
def add_bh_adjusted_p_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Benjamini-Hochberg 방식으로
    다중검정 보정 p값을 계산합니다.
    """
    df = df.copy()

    # p값을 숫자형으로 변환
    p_values = pd.to_numeric(
        df["p_value"],
        errors="coerce",
    )

    df["p_value"] = p_values

    # 실수형 열로 생성
    df["p_value_bh"] = float("nan")

    valid_p_values = (
        p_values
        .dropna()
        .sort_values()
    )

    if not valid_p_values.empty:
        test_count = len(
            valid_p_values
        )

        adjusted_values = []

        for rank, p_value in enumerate(
            valid_p_values,
            start=1,
        ):
            adjusted_value = min(
                p_value
                * test_count
                / rank,
                1.0,
            )

            adjusted_values.append(
                adjusted_value
            )

        # BH 보정값의 단조성 보정
        for index in range(
            len(adjusted_values) - 2,
            -1,
            -1,
        ):
            adjusted_values[index] = min(
                adjusted_values[index],
                adjusted_values[index + 1],
            )

        adjusted_series = pd.Series(
            adjusted_values,
            index=valid_p_values.index,
            dtype="float64",
        )

        df.loc[
            adjusted_series.index,
            "p_value_bh",
        ] = adjusted_series

    # 유의성 여부
    df["significant_0_05"] = (
        df["p_value"]
        .lt(0.05)
        .astype("boolean")
    )

    df.loc[
        df["p_value"].isna(),
        "significant_0_05",
    ] = pd.NA

    df["significant_bh_0_05"] = (
        df["p_value_bh"]
        .lt(0.05)
        .astype("boolean")
    )

    df.loc[
        df["p_value_bh"].isna(),
        "significant_bh_0_05",
    ] = pd.NA

    return df


# =====================================================================
# 11. 저장
# =====================================================================
def save_results(
    distribution_df: pd.DataFrame,
    crosstab_df: pd.DataFrame,
    chi_square_df: pd.DataFrame,
    residual_df: pd.DataFrame,
) -> None:
    distribution_df.to_csv(
        DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    crosstab_df.to_csv(
        CROSSTAB_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    chi_square_df.to_csv(
        CHI_SQUARE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    residual_df.to_csv(
        RESIDUAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
    ) as writer:
        distribution_df.to_excel(
            writer,
            sheet_name="distribution",
            index=False,
        )

        crosstab_df.to_excel(
            writer,
            sheet_name="case_crosstab",
            index=False,
        )

        chi_square_df.to_excel(
            writer,
            sheet_name="chi_square",
            index=False,
        )

        residual_df.to_excel(
            writer,
            sheet_name="residuals",
            index=False,
        )


# =====================================================================
# 12. 실행
# =====================================================================
def main() -> None:
    df = load_data()

    distribution_df = (
        create_distribution_table(df)
    )

    crosstab_df = (
        create_case_crosstabs(df)
    )

    (
        chi_square_df,
        residual_df,
    ) = create_chi_square_results(df)

    save_results(
        distribution_df=distribution_df,
        crosstab_df=crosstab_df,
        chi_square_df=chi_square_df,
        residual_df=residual_df,
    )

    print("=" * 60)
    print("전체 GPT 분류 결과 분석 완료")
    print("=" * 60)
    print(f"분석 단위: {len(df)}개")
    print(f"모델: {df['model'].iloc[0]}")
    print(
        f"프롬프트 버전: "
        f"{df['prompt_version'].iloc[0]}"
    )

    print()
    print("사례 간 카이제곱 검정")
    print(
        chi_square_df[
            [
                "task",
                "chi_square",
                "degrees_of_freedom",
                "p_value",
                "p_value_bh",
                "cramers_v",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"분포: {DISTRIBUTION_PATH}")
    print(f"교차표: {CROSSTAB_PATH}")
    print(f"카이제곱: {CHI_SQUARE_PATH}")
    print(f"잔차: {RESIDUAL_PATH}")
    print(f"통합 Excel: {EXCEL_PATH}")


if __name__ == "__main__":
    main()