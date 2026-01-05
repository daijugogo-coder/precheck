from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import unicodedata

import pandas as pd


def list_excel_files(folder: Path) -> list[Path]:
    """
    同一フォルダ内のExcelを列挙する。
    ただし、自分自身が出力した merge-CL-*.xlsx / omit-CL-*.xlsx は除外する。
    """
    exts = (".xlsx", ".xlsm", ".xls")
    files: list[Path] = []

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if p.name.startswith("merge-CL-") or p.name.startswith("omit-CL-"):
            continue
        files.append(p)

    return files


def drop_ag_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    ExcelのAG列 = 33列目（1-based）= index 32（0-based）
    常に不要なので、存在すれば必ず削除する。
    """
    ag_index = 32
    if df.shape[1] <= ag_index:
        return df
    return df.drop(df.columns[ag_index], axis=1)


def normalize_columns(cols) -> list[str]:
    """列構造比較用（列名揺れは無い前提なので最小限）"""
    return ["" if c is None else str(c).strip() for c in cols]


def describe_first_difference(expected: list[str], actual: list[str]) -> str:
    if len(expected) != len(actual):
        return f"Column count mismatch: expected={len(expected)} actual={len(actual)}"

    for i, (e, a) in enumerate(zip(expected, actual), start=1):
        if e != a:
            return f"First mismatch at column {i}: expected='{e}' actual='{a}'"

    return "Unknown difference"


def normalize_text(s: str) -> str:
    """半角・全角の揺れを吸収するために正規化する（NFKC）"""
    return unicodedata.normalize("NFKC", s)


def split_docomo_shop_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    D列（4列目）に「ドコモショップ（半角全角問わず）」を含む行を残し、
    それ以外を omit（除外）として分離する。
    """
    col_index = 3  # D列（0-based）
    keyword = normalize_text("ドコモショップ")

    series = (
        df.iloc[:, col_index]
        .astype("string")
        .fillna("")
        .map(normalize_text)
    )

    mask = series.str.contains(keyword, na=False)

    kept_df = df[mask].copy()
    omitted_df = df[~mask].copy()

    return kept_df, omitted_df


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    files = list_excel_files(base_dir)
    if not files:
        print("ERROR: No Excel files found in the current folder.", file=sys.stderr)
        return 2

    # ログ安定のためソート（結合順に意味はない）
    files = sorted(files)

    dataframes: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    baseline_file: str | None = None

    for file in files:
        try:
            # 先頭シート固定
            df = pd.read_excel(file, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"ERROR: failed to read {file.name}: {e}", file=sys.stderr)
            return 1

        before_cols = df.shape[1]
        df = drop_ag_column(df)
        after_cols = df.shape[1]

        current_columns = normalize_columns(df.columns)

        if expected_columns is None:
            expected_columns = current_columns
            baseline_file = file.name
        else:
            if current_columns != expected_columns:
                print("ERROR: column schema mismatch detected. Processing stopped.", file=sys.stderr)
                print(f"  Baseline file : {baseline_file}", file=sys.stderr)
                print(f"  Mismatch file : {file.name}", file=sys.stderr)
                print(f"  Detail        : {describe_first_difference(expected_columns, current_columns)}", file=sys.stderr)
                print("  Expected columns:", file=sys.stderr)
                print(f"    {expected_columns}", file=sys.stderr)
                print("  Actual columns:", file=sys.stderr)
                print(f"    {current_columns}", file=sys.stderr)
                return 3

        dataframes.append(df)
        print(f"OK: {file.name} rows={len(df)} cols={before_cols}->{after_cols}")

    merged_df = pd.concat(dataframes, ignore_index=True)

    # フィルタ（残す/除外を分離）
    before_rows = len(merged_df)
    kept_df, omitted_df = split_docomo_shop_rows(merged_df)
    kept_rows = len(kept_df)
    omitted_rows = len(omitted_df)

    print(
        f"FILTER: E列 contains 'ドコモショップ' "
        f"kept {kept_rows} / omitted {omitted_rows} (from {before_rows})"
    )

    # 出力（同フォルダ、ファイル名は固定規則）
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    out_keep = base_dir / f"merge-CL-{timestamp}.xlsx"
    out_omit = base_dir / f"omit-CL-{timestamp}.xlsx"

    kept_df.to_excel(out_keep, index=False, engine="openpyxl")
    if omitted_rows > 0:
        omitted_df.to_excel(out_omit, index=False, engine="openpyxl")

    print("DONE:")
    print(f"  kept   : {out_keep.name} ({kept_rows} rows)")
    print(f"  omitted: {out_omit.name} ({omitted_rows} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
