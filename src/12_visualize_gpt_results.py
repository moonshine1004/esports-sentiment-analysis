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
# 1. 파일 경로 설정
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
# 2. 일반 그래프 설정
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


TARGET_STANCE_PATH = (
    FIGURE_DIR
    / "figure_05_target_by_stance.png"
)


# =====================================================================
# 3. 그래프명 
# =====================================================================
CASE_DISPLAY_NAMES = {
    "01": "룬 설정 오류",
    "02": "강타 재사용 대기시간 오류",
    "03": "경기 전체",
}

CASE_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#E15759",
]


STANCE_COLORS = [
    "#4C78A8",
    "#F2A65A",
    "#59A14F",
    "#9C9C9C",
]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 4. 자료 불러오기
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
# 5. 라벨 검사
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
# 6. 통합데이터와 사례별 자료
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

    if len(groups) > len(CASE_COLORS):
        raise ValueError(
            "사례 수보다 CASE_COLORS의 색상 수가 적습니다."
        )

    return groups

# =====================================================================
# 7. 막대 위 숫자 표시
# =====================================================================
def add_bar_labels(
    ax,
    bars,
    counts,
    percentages,
    offset: float,
    fontsize: float = 8,
) -> None:
    for bar, count, percentage in zip(
        bars,
        counts,
        percentages,
    ):
        count = int(count)

        if count == 0:
            continue

        label_text = (
            f"{count:,}"
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
            fontsize=fontsize,
            color="#222222",
        )


# =====================================================================
# 8. 일반 분포 그래프
# =====================================================================
def create_grouped_bar_chart(
    df: pd.DataFrame,
    column: str,
    labels: list[str],
    output_path,
) -> None:
    groups = get_case_groups(df)

    group_count = len(groups)

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
        len(labels) * 2.7,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            7,
        )
    )

    result_rows = []
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

        result_rows.append(
            {
                "group_index": group_index,
                "name": group["name"],
                "counts": counts,
                "percentages": percentages,
            }
        )

    label_offset = max(
        max_count * 0.012,
        1,
    )

    for result in result_rows:
        group_index = result[
            "group_index"
        ]

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
            color=CASE_COLORS[
                group_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=result["name"],
        )

        add_bar_labels(
            ax=ax,
            bars=bars,
            counts=result[
                "counts"
            ].values,
            percentages=result[
                "percentages"
            ].values,
            offset=label_offset,
            fontsize=8,
        )

    upper_limit = max(
        10,
        max_count * 1.25,
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

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 9. 통합데이터 대상별 태도 그래프
# =====================================================================
def create_target_by_stance_chart(
    df: pd.DataFrame,
    output_path,
) -> None:
    target_labels = list(
        TARGET_VALUES
    )

    stance_labels = list(
        STANCE_VALUES
    )

    target_count = len(
        target_labels
    )

    stance_count = len(
        stance_labels
    )

    if stance_count > len(STANCE_COLORS):
        raise ValueError(
            "태도 수보다 STANCE_COLORS의 색상 수가 적습니다."
        )

    x = np.arange(
        target_count
    )

    cluster_width = 0.84

    bar_width = (
        cluster_width
        / stance_count
    )

    figure_width = max(
        12,
        target_count * 2.8,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            7,
        )
    )

    target_totals = (
        df["gpt_target"]
        .value_counts()
        .reindex(
            target_labels,
            fill_value=0,
        )
    )

    result_rows = []
    max_count = 0

    for stance_index, stance in enumerate(
        stance_labels
    ):
        stance_df = df[
            df["gpt_stance"].eq(
                stance
            )
        ]

        counts = (
            stance_df[
                "gpt_target"
            ]
            .value_counts()
            .reindex(
                target_labels,
                fill_value=0,
            )
        )

        percentages = []

        for target in target_labels:
            target_total = int(
                target_totals[target]
            )

            count = int(
                counts[target]
            )

            if target_total > 0:
                percentage = (
                    count
                    / target_total
                    * 100
                )
            else:
                percentage = 0.0

            percentages.append(
                percentage
            )

        max_count = max(
            max_count,
            int(counts.max()),
        )

        result_rows.append(
            {
                "stance_index": stance_index,
                "stance": stance,
                "counts": counts.values,
                "percentages": percentages,
            }
        )

    label_offset = max(
        max_count * 0.012,
        1,
    )

    for result in result_rows:
        stance_index = result[
            "stance_index"
        ]

        positions = (
            x
            - cluster_width / 2
            + bar_width / 2
            + stance_index * bar_width
        )

        bars = ax.bar(
            positions,
            result["counts"],
            width=bar_width,
            color=STANCE_COLORS[
                stance_index
            ],
            edgecolor="#444444",
            linewidth=0.5,
            label=result["stance"],
        )

        add_bar_labels(
            ax=ax,
            bars=bars,
            counts=result["counts"],
            percentages=result[
                "percentages"
            ],
            offset=label_offset,
            fontsize=8,
        )

    upper_limit = max(
        10,
        max_count * 1.27,
    )

    ax.set_ylim(
        0,
        upper_limit,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        target_labels,
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
        ncol=stance_count,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.12,
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

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 10. 실행
# =====================================================================
def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    # 기존 네 개 분포 그래프
    for config in TASK_CONFIG.values():
        output_path = (
            FIGURE_DIR
            / config["filename"]
        )

        create_grouped_bar_chart(
            df=df,
            column=config["column"],
            labels=config["labels"],
            output_path=output_path,
        )

    # 통합데이터 대상별 태도 그래프
    create_target_by_stance_chart(
        df=df,
        output_path=TARGET_STANCE_PATH,
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

    print(
        TARGET_STANCE_PATH
    )


if __name__ == "__main__":
    main()