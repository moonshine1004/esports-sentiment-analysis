"""댓글 작성 시점 분포를 사건일·수집일과 함께 표와 그림으로 산출."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

from config import EVENT_DATE, FIGURES_DIR, INTERIM_DIR, TABLES_DIR

INPUT_PATH = INTERIM_DIR / "comments_preprocessed.csv"
DETAIL_CSV = TABLES_DIR / "comment_time_detail.csv"
SUMMARY_CSV = TABLES_DIR / "comment_time_summary.csv"
PERIOD_CSV = TABLES_DIR / "comment_time_distribution_period.csv"
YEAR_CSV = TABLES_DIR / "comment_time_distribution_year.csv"
YEAR_MONTH_CSV = TABLES_DIR / "comment_time_distribution_month.csv"
CASE_PERIOD_CSV = TABLES_DIR / "comment_time_distribution_case_period.csv"
COMMENT_TYPE_PERIOD_CSV = TABLES_DIR / "comment_time_distribution_comment_type_period.csv"
FIGURE_PATH = FIGURES_DIR / "comment_time_distribution.png"

PERIOD_LABELS = ["0-7일", "8-30일", "31-90일", "91-365일", "366일 이상"]
PERIOD_BINS = [-1, 7, 30, 90, 365, float("inf")]


def configure_korean_font() -> str | None:
    """Windows·macOS·Linux에서 사용 가능한 한글 글꼴을 순서대로 선택."""
    preferred = [
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in installed), None)
    if selected:
        plt.rcParams["font.family"] = selected
    else:
        warnings.warn(
            "한글 글꼴을 찾지 못했습니다. 표 CSV는 정상이며 그림의 한글 글리프를 확인하십시오.",
            stacklevel=2,
        )
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def prepare_time_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "published_at",
        "collected_at_utc",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"작성 시점 분석에 필요한 열이 없습니다: {sorted(missing)}")
    if df["analysis_unit_id"].duplicated().any():
        duplicate_ids = df.loc[
            df["analysis_unit_id"].duplicated(keep=False), "analysis_unit_id"
        ].tolist()[:10]
        raise ValueError(f"중복 analysis_unit_id가 있습니다: {duplicate_ids}")

    output = df.copy()
    output["published_at_utc"] = pd.to_datetime(
        output["published_at"], utc=True, errors="coerce"
    )
    output["collected_at_parsed_utc"] = pd.to_datetime(
        output["collected_at_utc"], utc=True, errors="coerce"
    )
    for column in ["published_at_utc", "collected_at_parsed_utc"]:
        if output[column].isna().any():
            bad = output.loc[output[column].isna(), "analysis_unit_id"].tolist()[:10]
            raise ValueError(f"{column}을 날짜로 해석할 수 없습니다: {bad}")
    if (output["published_at_utc"] > output["collected_at_parsed_utc"]).any():
        bad = output.loc[
            output["published_at_utc"] > output["collected_at_parsed_utc"],
            "analysis_unit_id",
        ].tolist()[:10]
        raise ValueError(f"수집 시각보다 늦게 작성된 댓글이 있습니다: {bad}")

    event_timestamp = pd.Timestamp(EVENT_DATE, tz="UTC")
    output["days_since_event"] = (
        output["published_at_utc"].dt.normalize() - event_timestamp
    ).dt.days
    if output["days_since_event"].lt(0).any():
        bad = output.loc[output["days_since_event"].lt(0), "analysis_unit_id"].tolist()[:10]
        raise ValueError(f"사건 발생일보다 이른 댓글이 있습니다. EVENT_DATE를 확인하십시오: {bad}")

    output["time_period"] = pd.cut(
        output["days_since_event"],
        bins=PERIOD_BINS,
        labels=PERIOD_LABELS,
        ordered=True,
    )
    output["published_date"] = output["published_at_utc"].dt.date.astype(str)
    output["published_year"] = output["published_at_utc"].dt.year.astype(int)
    output["year_month"] = (
        output["published_at_utc"].dt.tz_localize(None).dt.to_period("M").astype(str)
    )
    return output


def summarize(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = df.groupby(group_columns, observed=False).size().reset_index(name="n")
    denominator_columns = group_columns[:-1]
    if denominator_columns:
        denominator = result.groupby(denominator_columns, observed=False)["n"].transform("sum")
    else:
        denominator = pd.Series(result["n"].sum(), index=result.index)
    result["percent"] = result["n"].div(denominator).mul(100).fillna(0.0)
    return result


def summarize_years(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("published_year").size()
    first_year = min(EVENT_DATE.year, int(counts.index.min()))
    last_year = int(df["collected_at_parsed_utc"].dt.year.max())
    years = pd.Index(range(first_year, last_year + 1), name="published_year")
    result = counts.reindex(years, fill_value=0).rename("n").reset_index()
    result["percent"] = result["n"] / result["n"].sum() * 100
    return result


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_comments": len(df),
                "event_date": EVENT_DATE.isoformat(),
                "earliest_published_at_utc": df["published_at_utc"].min(),
                "latest_published_at_utc": df["published_at_utc"].max(),
                "collection_started_at_utc": df["collected_at_parsed_utc"].min(),
                "collection_ended_at_utc": df["collected_at_parsed_utc"].max(),
                "median_days_since_event": float(df["days_since_event"].median()),
                "mean_days_since_event": float(df["days_since_event"].mean()),
                "percent_within_7_days": float(df["days_since_event"].le(7).mean() * 100),
                "percent_within_30_days": float(df["days_since_event"].le(30).mean() * 100),
                "percent_within_365_days": float(df["days_since_event"].le(365).mean() * 100),
            }
        ]
    )


def create_figure(period: pd.DataFrame, years: pd.DataFrame) -> None:
    configure_korean_font()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})

    period_bars = axes[0].bar(
        period["time_period"].astype(str),
        period["n"],
        color="#4472C4",
    )
    for bar, percent in zip(period_bars, period["percent"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_xlabel(f"사건 발생일({EVENT_DATE.isoformat()}) 이후 작성 시점")
    axes[0].set_ylabel("댓글 수")
    axes[0].set_title("사건 발생일 기준 댓글 작성 시점 분포")
    axes[0].tick_params(axis="x", rotation=20)

    year_bars = axes[1].bar(
        years["published_year"].astype(str),
        years["n"],
        color="#70AD47",
    )
    for bar, percent in zip(year_bars, years["percent"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xlabel("댓글 작성 연도")
    axes[1].set_ylabel("댓글 수")
    axes[1].set_title("연도별 댓글 작성 분포")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.12)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze_comment_time_distribution(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = prepare_time_data(df)
    period = summarize(data, ["time_period"])
    years = summarize_years(data)
    outputs = {
        "detail": data,
        "summary": create_summary(data),
        "period": period,
        "year": years,
        "month": summarize(data, ["year_month"]),
        "case_period": summarize(data, ["case_id", "case_name", "time_period"]),
        "comment_type_period": summarize(data, ["comment_type", "time_period"]),
    }
    return outputs


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"전처리 파일이 없습니다: {INPUT_PATH}")
    outputs = analyze_comment_time_distribution(
        pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    detail_columns = [
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "published_at_utc",
        "collected_at_parsed_utc",
        "days_since_event",
        "time_period",
        "published_date",
        "published_year",
        "year_month",
    ]
    outputs["detail"][detail_columns].to_csv(
        DETAIL_CSV, index=False, encoding="utf-8-sig"
    )
    outputs["summary"].to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    outputs["period"].to_csv(PERIOD_CSV, index=False, encoding="utf-8-sig")
    outputs["year"].to_csv(YEAR_CSV, index=False, encoding="utf-8-sig")
    outputs["month"].to_csv(YEAR_MONTH_CSV, index=False, encoding="utf-8-sig")
    outputs["case_period"].to_csv(CASE_PERIOD_CSV, index=False, encoding="utf-8-sig")
    outputs["comment_type_period"].to_csv(
        COMMENT_TYPE_PERIOD_CSV, index=False, encoding="utf-8-sig"
    )
    create_figure(outputs["period"], outputs["year"])
    print(outputs["summary"].to_string(index=False))
    print("\n" + outputs["period"].to_string(index=False))
    print(f"\n요약 표: {SUMMARY_CSV}")
    print(f"구간별 표: {PERIOD_CSV}")
    print(f"연도별 표: {YEAR_CSV}")
    print(f"그림: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
