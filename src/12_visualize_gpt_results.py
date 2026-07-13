"""
GPT 분류 결과 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 1. 파일 경로
# =====================================================================
INPUT_PATH = (
    RESULTS_DIR
    / "gpt_predictions_final.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)


# =====================================================================
# 2. 그래프 설정
# =====================================================================
TASK_CONFIG = {
    "sentiment": {
        "column": "gpt_sentiment",
        "labels": SENTIMENT_VALUES,
        "filename": "figure_01_sentiment.png",
    },
    "target": {
        "column": "gpt_target",
        "labels": TARGET_VALUES,
        "filename": "figure_02_target.png",
    },
    "stance": {
        "column": "gpt_stance",
        "labels": STANCE_VALUES,
        "filename": "figure_03_stance.png",
    },
    "is_sarcasm_mockery": {
        "column": "gpt_is_sarcasm_mockery",
        "labels": SARCASM_VALUES,
        "filename": "figure_04_is_sarcasm_mockery.png",
    },
}

# 사례별 라벨명
CASE_DISPLAY_NAMES = {
    "01": "룬 설정 오류",
    "02": "강타 재사용 대기시간 오류",
    "03": "전체 경기 하이라이트",
}

# 그래프 색상 설정
BAR_COLORS = [
    "#4C78A8",  # 통합데이터
    "#F2A65A",  # 사례 01
    "#59A14F",  # 사례 02
    "#E15759",  # 사례 03
]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


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
        "gpt_sentiment",
        "gpt_target",
        "gpt_stance",
        "gpt_is_sarcasm_mockery",
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

    df["case_id"] = (
        df["case_id"]
        .astype(str)
        .str.strip()
        .str.zfill(2)
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

    validate_labels(df)

    return df


# =====================================================================
# 4. 라벨 검사
# =====================================================================
def validate_labels(
    df: pd.DataFrame,
) -> None:
    valid_labels = {
        "gpt_sentiment": SENTIMENT_VALUES,
        "gpt_target": TARGET_VALUES,
        "gpt_stance": STANCE_VALUES,
        "gpt_is_sarcasm_mockery": SARCASM_VALUES,
    }

    for column, labels in valid_labels.items():
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
            labels
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


# =====================================================================
# 5. 통합데이터와 사례별 자료 구성
# =====================================================================
def get_case_groups(
    df: pd.DataFrame,
) -> list[dict]:
    groups = [
        {
            "name": (
                f"통합데이터 "
                f"(N={len(df):,})"
            ),
            "data": df,
        }
    ]

    case_info_df = (
        df[
            [
                "case_id",
                "case_name",
            ]
        ]
        .drop_duplicates(
            subset=["case_id"]
        )
        .sort_values("case_id")
    )

    for _, row in case_info_df.iterrows():
        case_id = row["case_id"]

        case_df = df[
            df["case_id"].eq(case_id)
        ].copy()

        display_name = (
            CASE_DISPLAY_NAMES.get(
                case_id,
                row["case_name"],
            )
        )

        groups.append(
            {
                "name": (
                    f"{display_name} "
                    f"(N={len(case_df):,})"
                ),
                "data": case_df,
            }
        )

    return groups


# =====================================================================
# 6. 막대 끝 표기
# =====================================================================
def add_bar_labels(
    ax,
    bars,
    counts: pd.Series,
    percentages: pd.Series,
    offset: float,
) -> None:
    for bar, count, percentage in zip(
        bars,
        counts.values,
        percentages.values,
    ):
        label_text = (
            f"{int(count):,}"
            f"({percentage:.0f}%)"
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + offset,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )


# =====================================================================
# 7. 그래프 생성
# =====================================================================
def create_bar_chart(
    df: pd.DataFrame,
    column: str,
    labels: list[str],
    output_path,
) -> None:
    groups = get_case_groups(df)

    group_count = len(groups)

    if group_count > len(BAR_COLORS):
        raise ValueError(
            "사례 수보다 BAR_COLORS의 색상 수가 적습니다."
        )

    x = np.arange(
        len(labels)
    )

    cluster_width = 0.84

    bar_width = (
        cluster_width
        / group_count
    )

    figure_width = max(
        11,
        len(labels) * 2.6,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            7,
        )
    )

    group_results = []
    max_count = 0

    for group_index, group in enumerate(
        groups
    ):
        group_df = group["data"]

        counts = (
            group_df[column]
            .value_counts()
            .reindex(
                labels,
                fill_value=0,
            )
        )

        percentages = (
            counts
            / len(group_df)
            * 100
        )

        max_count = max(
            max_count,
            int(counts.max()),
        )

        group_results.append(
            {
                "name": group["name"],
                "counts": counts,
                "percentages": percentages,
                "index": group_index,
            }
        )

    label_offset = max(
        max_count * 0.012,
        1,
    )

    for result in group_results:
        group_index = result["index"]

        positions = (
            x
            - cluster_width / 2
            + bar_width / 2
            + group_index * bar_width
        )

        bars = ax.bar(
            positions,
            result["counts"].values,
            width=bar_width,
            color=BAR_COLORS[
                group_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=result["name"],
        )

        add_bar_labels(
            ax=ax,
            bars=bars,
            counts=result["counts"],
            percentages=(
                result["percentages"]
            ),
            offset=label_offset,
        )

    upper_limit = max(
        10,
        max_count * 1.22,
    )

    ax.set_ylim(
        0,
        upper_limit,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.set_ylabel(
        "건수",
        fontsize=10,
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.17,
        ),
        fontsize=9,
        columnspacing=2.0,
        handlelength=1.8,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 8. 실행
# =====================================================================
def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    for config in TASK_CONFIG.values():
        output_path = (
            FIGURE_DIR
            / config["filename"]
        )

        create_bar_chart(
            df=df,
            column=config["column"],
            labels=config["labels"],
            output_path=output_path,
        )

    print("=" * 60)
    print("GPT 결과 시각화 완료")
    print("=" * 60)
    print(f"전체 분석 단위: {len(df):,}개")

    for config in TASK_CONFIG.values():
        print(
            FIGURE_DIR
            / config["filename"]
        )


if __name__ == "__main__":
    main()