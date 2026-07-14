"""
코더 1과 코더 2의 Excel 파일을 생성
"""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from config import (
    HUMAN_DIR,
)
from labels import (
    SENTIMENT_VALUES,
    TARGET_VALUES,
    STANCE_VALUES,
    SARCASM_VALUES,
)


# =====================================================================
# 1. 파일 경로 설정
# =====================================================================
INPUT_PATH = HUMAN_DIR / "human_sample_master.csv"

CODER1_PATH = HUMAN_DIR / "coder1_coding.xlsx"

# =====================================================================
# 2. 출력 열
# =====================================================================
OUTPUT_COLUMNS = [
    "sample_id",
    "analysis_unit_id",
    "case_id",
    "case_name",
    "comment_type",
    "parent_text",
    "analysis_text",
    "sentiment",
    "target",
    "stance",
    "is_sarcasm_mockery",
    "coder_note",
]


# =====================================================================
# 3. 표본 불러오기
# =====================================================================
def load_sample() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"표본 파일이 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "analysis_unit_id",
        "case_id",
        "case_name",
        "comment_type",
        "parent_text",
        "analysis_text",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {sorted(missing_columns)}"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "중복 sample_id가 있습니다."
        )

    return df


# =====================================================================
# 4. 드롭다운 추가
# =====================================================================
def add_dropdown(
    sheet,
    column_name: str,
    values: list[str],
) -> None:
    header_index = {
        cell.value: cell.column
        for cell in sheet[1]
    }

    column_number = header_index[column_name]
    column_letter = get_column_letter(column_number)

    validation = DataValidation(
        type="list",
        formula1=f'"{",".join(values)}"',
        allow_blank=True,
    )

    validation.error = (
        "드롭다운 목록에 있는 값만 입력하십시오."
    )
    validation.errorTitle = "잘못된 라벨"

    sheet.add_data_validation(validation)

    validation.add(
        f"{column_letter}2:{column_letter}{sheet.max_row}"
    )


# =====================================================================
# 5. Excel 서식 설정
# =====================================================================
def apply_style(sheet) -> None:
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="BDD7EE",
    )

    input_fill = PatternFill(
        fill_type="solid",
        fgColor="FFF2CC",
    )

    input_columns = {
        "sentiment",
        "target",
        "stance",
        "is_sarcasm_mockery",
        "coder_note",
    }

    header_index = {
        cell.value: cell.column
        for cell in sheet[1]
    }

    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for row in sheet.iter_rows(
        min_row=2,
        max_row=sheet.max_row,
    ):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    for column_name in input_columns:
        column_number = header_index[column_name]

        for row_number in range(2, sheet.max_row + 1):
            sheet.cell(
                row=row_number,
                column=column_number,
            ).fill = input_fill

    width_map = {
        "sample_id": 12,
        "analysis_unit_id": 18,
        "case_id": 10,
        "case_name": 32,
        "comment_type": 15,
        "parent_text": 60,
        "analysis_text": 60,
        "sentiment": 16,
        "target": 18,
        "stance": 18,
        "is_sarcasm_mockery": 22,
        "coder_note": 40,
    }

    for column_name, width in width_map.items():
        column_number = header_index[column_name]
        column_letter = get_column_letter(column_number)

        sheet.column_dimensions[column_letter].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


# =====================================================================
# 6. 코더 파일 생성
# =====================================================================
def create_workbook(
    df: pd.DataFrame,
    output_path,
) -> None:
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Coding"

    sheet.append(OUTPUT_COLUMNS)

    for _, row in df.iterrows():
        sheet.append(
            [
                row["sample_id"],
                row["analysis_unit_id"],
                row["case_id"],
                row["case_name"],
                row["comment_type"],
                row["parent_text"],
                row["analysis_text"],
                "",
                "",
                "",
                "",
                "",
            ]
        )

    apply_style(sheet)

    add_dropdown(
        sheet,
        "sentiment",
        SENTIMENT_VALUES,
    )

    add_dropdown(
        sheet,
        "target",
        TARGET_VALUES,
    )

    add_dropdown(
        sheet,
        "stance",
        STANCE_VALUES,
    )

    add_dropdown(
        sheet,
        "is_sarcasm_mockery",
        SARCASM_VALUES,
    )

    workbook.save(output_path)


# =====================================================================
# 7. 실행
# =====================================================================
def main() -> None:
    if CODER1_PATH.exists():
        raise FileExistsError(
            "coder1_coding.xlsx가 이미 존재합니다."
        )

    df = load_sample()

    create_workbook(
        df,
        CODER1_PATH,
    )

    print("=" * 60)
    print("인간 코딩 Excel 생성 완료")
    print("=" * 60)
    print(f"코딩 표본: {len(df)}개")
    print(f"파일: {CODER1_PATH}")


if __name__ == "__main__":
    main()