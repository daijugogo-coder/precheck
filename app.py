import streamlit as st
import pandas as pd
import io
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, date
from io import StringIO

# ページ設定
st.set_page_config(page_title="事前チェックシステム", page_icon="📦", layout="wide")

# --- Hide Streamlit default menu/footer/header (public-friendly) ---
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      [data-testid="stToolbar"] {visibility: hidden; height: 0px;}
      [data-testid="stDecoration"] {visibility: hidden; height: 0px;}
      [data-testid="stStatusWidget"] {visibility: hidden;}
      [data-testid="stHeaderActionElements"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 事前チェックシステム")
st.markdown("---")

# ----------------------------
# Constants (column names etc.)
# ----------------------------
COL_TONYA = "取次店コード"
COL_SHOHIN = "商品コード"
COL_TMS = "TMS商品CD"
COL_HOKAN = "保管場所CD"
COL_JIGYO = "事業CD"
COL_INV_BEFORE = "受払前在庫数"
COL_INV_AFTER = "受払後在庫数"


# ----------------------------
# Utilities
# ----------------------------
def force_str_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame全体を安全に文字列化
    NaN / None / pd.NA 全部 "" にする
    """
    if df is None:
        return df
    return df.fillna("").astype(str)


def force_str_series(s: pd.Series) -> pd.Series:
    """
    Seriesを安全に文字列化
    """
    return s.fillna("").astype(str)


def normalize_key_series(s: pd.Series) -> pd.Series:
    """PowerQueryのTrim相当: 前後空白（全角含む）・BOM等を除去し、突合キーのブレを防ぐ。"""
    out = s.astype(str)
    out = out.str.replace("\ufeff", "", regex=False)
    out = out.str.replace("　", " ", regex=False)  # 全角スペース→半角
    out = out.str.strip()
    # 文字列化で混入する表現を空に寄せる
    out = out.replace({"nan": "", "None": "", "NaN": ""})
    return out


# セッションステートの初期化
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# file_uploader reset key
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


def safe_rerun() -> None:
    """Rerun the Streamlit script in a way compatible with multiple Streamlit versions."""
    # Newer Streamlit
    if hasattr(st, "rerun"):
        try:
            st.rerun()
            return
        except Exception:
            pass
    # Older Streamlit
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    # Fallback: raise the internal RerunException (very old versions)
    try:
        from streamlit.runtime.scriptrunner.script_runner import RerunException

        raise RerunException()
    except Exception:
        # As last resort, stop to prevent further UI actions
        try:
            st.stop()
        except Exception:
            return


def load_csv_with_encoding(file, use_lf=True, encoding="cp932") -> pd.DataFrame:
    try:
        content = file.read()
        decoded_content = content.decode(encoding)

        if use_lf:
            df = pd.read_csv(
                io.StringIO(decoded_content),
                dtype=str,
                keep_default_na=False,
                lineterminator="\n",
            )
        else:
            df = pd.read_csv(
                io.StringIO(decoded_content), dtype=str, keep_default_na=False
            )

        # ← 追加（これが効く）
        df = force_str_df(df)

        return df

    except Exception as e:
        st.error(f"CSVファイルの読み込みエラー: {str(e)}")
        return None


def load_master_files(master_857001, master_857002, master_857003, master_13000=None) -> Dict[str, pd.DataFrame]:

    def _nfkc(s) -> str:
        if pd.isna(s):
            return ""
        return unicodedata.normalize("NFKC", str(s)).replace("\ufeff", "").strip()

    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [_nfkc(str(c)) for c in df.columns]
        return df

    def normalize_text_series(s: pd.Series) -> pd.Series:
        return (
            s.fillna("").astype(str)
             .map(lambda x: _nfkc(x).replace(" ", "").replace("　", ""))
        )

    def pick_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
        colmap: Dict[str, str] = {}
        for dst, candidates in mapping.items():
            found = None
            for c in candidates:
                if c in df.columns:
                    found = c
                    break
            if found is None:
                raise KeyError(f"マスタ列 '{dst}' が見つかりません。候補={candidates} / 実列={df.columns.tolist()}")
            colmap[dst] = found

        out = df[[colmap[k] for k in mapping.keys()]].copy()
        out.columns = list(mapping.keys())
        return out

    masters: Dict[str, pd.DataFrame] = {}

    # --- 857001 ---
    if master_857001:
        df = load_csv_with_encoding(master_857001, use_lf=False, encoding="utf-8-sig")
        if df is not None and not df.empty:
            df = normalize_columns(df)
            df = pick_columns(df, {
                "取次店コード": ["変換前コード401", "変換前コード値01", "変換前コード01"],
                "店舗倉庫区分": ["コード1", "コード値1"],
                "事業CDＢＫ": ["コード2", "コード値2"],
                "保管場所CD": ["コード4", "コード値4"],
            })

            for c in ["取次店コード", "店舗倉庫区分", "事業CDＢＫ", "保管場所CD"]:
                df[c] = normalize_text_series(df[c])

            df["事業CD"] = df["取次店コード"].apply(
                lambda x: "13000" if str(x).startswith("TG") else "15000"
            )

            df = df.drop_duplicates(subset=["取次店コード"])
            masters["857001"] = df

    # --- 857002 ---
    if master_857002:
        df = load_csv_with_encoding(master_857002, use_lf=False, encoding="utf-8-sig")
        if df is not None and not df.empty:
            df = normalize_columns(df)
            df = pick_columns(df, {
                "商品コード": ["変換前コード401", "変換前コード値01", "変換前コード01"],
                "事業CD": ["変換前コード402", "変換前コード値02", "変換前コード02"],
                "TMS商品CD": ["コード1", "コード値1"],
            })

            for c in ["商品コード", "事業CD", "TMS商品CD"]:
                df[c] = normalize_text_series(df[c])

            df = df.drop_duplicates(subset=["商品コード", "事業CD"])
            masters["857002"] = df

    # --- 857003 ---
    if master_857003:
        df = load_csv_with_encoding(master_857003, use_lf=False, encoding="utf-8-sig")
        if df is not None and not df.empty:
            df = normalize_columns(df)
            masters["857003"] = df

    # --- 13000 ---
    if master_13000:
        df = load_csv_with_encoding(master_13000, use_lf=False, encoding="utf-8-sig")
        if df is not None and not df.empty:
            df = normalize_columns(df)

            for col in ["パック商品CD", "内訳商品CD", "内訳商品名称"]:
                if col in df.columns:
                    df[col] = normalize_text_series(df[col])

            masters["13000"] = df

    return masters

def drop_ag_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    ExcelのAG列 = 33列目（1-based）= index 32（0-based）
    存在すれば削除する。CSVでも列数が多ければ同様に動作する。
    """
    ag_index = 32
    if df.shape[1] <= ag_index:
        return df
    return df.drop(df.columns[ag_index], axis=1)


def normalize_text(s: str) -> str:
    """半角・全角の揺れを吸収するために正規化する（NFKC）"""
    return unicodedata.normalize("NFKC", str(s))


def split_docomo_shop_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    D列（4列目）に「ドコモショップ（半角全角問わず）」を含む行を残し、
    それ以外を omit（除外）として分離する。
    安全のため、対象列が存在しない場合は元データをそのまま返す。
    """
    col_index = 3  # D列（0-based）
    if df.shape[1] <= col_index:
        # 対象列がないので分離は行わない
        return df.copy(), pd.DataFrame()
    keyword = normalize_text("ドコモショップ")
    series = df.iloc[:, col_index].astype("string").fillna("").map(normalize_text)
    mask = series.str.contains(keyword, na=False)
    kept_df = df[mask].copy()
    omitted_df = df[~mask].copy()
    return kept_df, omitted_df


# --- main.py のチェック機能取り込み ---
TARGET_COL_25 = 24
TARGET_COL_38 = 37
DATE_COL_9 = 8
DATE_COL_17 = 16
DATE_TIME_RE = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$")


@dataclass
class ErrorDetail:
    row: int
    store_name: str
    slip_number: str
    col_38: str


@dataclass
class DateIssue:
    record_no: int
    start_physical_line: int
    severity: str
    issue_type: str
    col9: str
    col17: str
    note: str


@dataclass
class DateSummary:
    total_checked_cells: int
    count_col9_ok: int
    count_error: int
    issues: List[DateIssue]


def csv_reader_from_text(csv_text: str):
    return csv.reader(StringIO(csv_text, newline=""))


def load_csv_from_text(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
    return force_str_df(df)


def load_current_inventory_excel(uploaded_file) -> pd.DataFrame:
    """現在庫照会（商品別）ExcelをDataFrameとして読み込む。

    - 1枚目シートを読み込む（PowerQuery運用と同じ前提）
    - 列名の前後空白/BOM/全角スペースを除去
    - 以降の処理（compare_with_current_inventory）が期待する列名
      ['保管場所CD','事業CD','商品CD','実在庫数量'] を含むことを前提とする
    """
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        data = uploaded_file.getvalue()
    except Exception:
        # 念のため
        data = uploaded_file.read()

    df = pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str)

    # 列名正規化（Trim相当 + BOM除去）
    cols = []
    for c in df.columns:
        s = str(c)
        s = s.replace("\ufeff", "")
        s = s.replace("　", " ").strip()
        cols.append(s)
    df.columns = cols

    return df


def parse_dt_str(s: str) -> Optional[datetime]:
    t = s.strip()
    if not DATE_TIME_RE.match(t):
        return None
    try:
        return datetime.strptime(t, "%Y/%m/%d %H:%M:%S")
    except Exception:
        return None


def build_error_csv_bytes(details: List[ErrorDetail]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["行番号(物理行)", "店舗名", "伝票番号", "金額(38列目)"])
    for d in details:
        w.writerow([d.row, d.store_name, d.slip_number, d.col_38])
    return buf.getvalue().encode("utf-8")


def build_date_issue_csv_bytes(issues: List[DateIssue]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "レコード番号",
            "開始物理行(参考)",
            "重要度",
            "種別",
            "9列目",
            "17列目",
            "補足",
        ]
    )
    for it in issues:
        w.writerow(
            [
                it.record_no,
                it.start_physical_line,
                it.severity,
                it.issue_type,
                it.col9,
                it.col17,
                it.note,
            ]
        )
    return buf.getvalue().encode("utf-8")


def check_and_analyze(
    csv_text: str,
) -> Tuple[bool, List[ErrorDetail], int, int, DateSummary]:  # NOSONAR
    """
    売上CSVの事前検査。
    - NG条件（25列=Z00014 かつ 38列が 3000/5000 以外）を検出
    - 9列の日付フォーマットチェック
    ※ 最小修正：全行をチェックするため、returnはループ外に移動
    """
    error_details: List[ErrorDetail] = []
    total_data_records = 0
    total_physical_lines = 0
    total_checked_cells = 0
    count_col9_ok = 0
    count_error = 0
    issues: List[DateIssue] = []

    reader = csv_reader_from_text(csv_text)
    prev_end_line = 0

    for record_no, row in enumerate(reader, start=1):
        start_physical_line = prev_end_line + 1
        end_physical_line = reader.line_num
        prev_end_line = end_physical_line
        total_physical_lines = end_physical_line

        # skip header
        if record_no == 1:
            continue
        total_data_records += 1

        # NGチェック 25/38
        if len(row) >= (TARGET_COL_38 + 1):
            col_3 = row[2].strip() if len(row) > 2 else ""
            col_11 = row[10].strip() if len(row) > 10 else ""
            col_25 = row[TARGET_COL_25].strip() if len(row) > TARGET_COL_25 else ""
            col_38 = row[TARGET_COL_38].strip() if len(row) > TARGET_COL_38 else ""
            if col_25 == "Z00014" and col_38 not in {"3000", "5000"}:
                error_details.append(
                    ErrorDetail(
                        row=start_physical_line,
                        store_name=col_3,
                        slip_number=col_11,
                        col_38=col_38,
                    )
                )

        # date checks
        col9 = row[DATE_COL_9].strip() if len(row) > DATE_COL_9 else ""
        dt9 = parse_dt_str(col9)
        if dt9 is None:
            count_error += 1
            issues.append(
                DateIssue(
                    record_no=total_data_records,
                    start_physical_line=start_physical_line,
                    severity="ERROR",
                    issue_type="COL9_MISSING_OR_INVALID",
                    col9=col9,
                    col17="",
                    note="9列目に yyyy/mm/dd hh:mm:ss が必要です。",
                )
            )
        else:
            count_col9_ok += 1
        total_checked_cells += 1

    # ループ終了後にサマリー作成して返す（←最小修正）
    date_summary = DateSummary(
        total_checked_cells=total_checked_cells,
        count_col9_ok=count_col9_ok,
        count_error=count_error,
        issues=issues,
    )
    return (
        (len(error_details) > 0),
        error_details,
        total_data_records,
        total_physical_lines,
        date_summary,
    )


### --- 取り込みここまで ---


def process_shiire_data(
    df: pd.DataFrame, masters: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    仕入データの完全処理（Power Query準拠）
    """
    # ★修正前
    # if df is None or df.empty:
    #     return pd.DataFrame()

    # ★修正後
    if df is None:
        return pd.DataFrame()

    # 型変換
    df["受払前在庫数"] = (
        pd.to_numeric(df["受払前在庫数"], errors="coerce").fillna(0).astype(int)
    )
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0).astype(int)
    df["受払後在庫数"] = (
        pd.to_numeric(df["受払後在庫数"], errors="coerce").fillna(0).astype(int)
    )
    # （以下は既存のまま）
    # マスタ857001とマージ（Inner Join）
    if "857001" in masters:
        df = df.merge(masters["857001"], on="取次店コード", how="inner")
    # 店舗倉庫区分でフィルタ
    df = df[df["店舗倉庫区分"] == "1"]
    # マスタ857002とマージ（Left Outer Join）
    if "857002" in masters:
        # 型を統一（文字列型に変換）
        df["事業CD"] = df["事業CD"].astype(str)
        master_857002 = masters["857002"].copy()
        master_857002["事業CD"] = master_857002["事業CD"].astype(str)
        df = df.merge(
            master_857002,
            on=["事業CD", "商品コード"],
            how="left",
            suffixes=("", "_master"),
        )
        # TMS商品CDがnullなら商品コードを使用
        df["TMS商品CD"] = df["TMS商品CD"].fillna(df["商品コード"])
    else:
        df["TMS商品CD"] = df["商品コード"]
    # 数量をバックアップ
    df["数量bk"] = df["数量"]
    # 受払種別でフィルタ
    valid_types = [
        "倉庫へ返品",
        "入荷",
        "入荷(システム自動)",
        "返品キャンセル",
        "返品不備",
    ]
    df = df[df["受払種別"].isin(valid_types)]
    # 除外商品コード
    exclude_codes = [
        "ZUA292",
        "ZUA34Q",
        "ZUA34R",
        "ZUA34S",
        "ZUA34T",
        "ZUA34U",
        "ZUA34V",
        "ZUA34W",
    ]
    for code in exclude_codes:
        df = df[~df["商品コード"].str.contains(code, na=False)]
    return df


def process_shiire_individual(df: pd.DataFrame) -> pd.DataFrame:
    """仕入データ（個体情報）"""
    # ← 最小修正：OR を追加（いずれかが非空なら対象）
    individual = df[
        (df["IMEI"].notna() & (df["IMEI"] != ""))
        | (df["ICCID"].notna() & (df["ICCID"] != ""))
        | (df["その他シリアル"].notna() & (df["その他シリアル"] != ""))
    ].copy()
    # カテゴリ中がＵＳＩＭカードでない
    individual = individual[individual["カテゴリ中"] != "ＵＳＩＭカード"]
    # 数量計算: 倉庫へ返品なら-1、それ以外は1
    individual["数量"] = individual["受払種別"].apply(
        lambda x: -1 if x == "倉庫へ返品" else 1
    )
    # グループ化
    result = (
        individual.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def process_shiire_accessory(df: pd.DataFrame) -> pd.DataFrame:
    """仕入データ（アクセサリ）"""
    # IMEI、ICCID、その他シリアル全てが空
    accessory = df[
        (df["IMEI"].isna() | (df["IMEI"] == ""))
        & (df["ICCID"].isna() | (df["ICCID"] == ""))
        & (df["その他シリアル"].isna() | (df["その他シリアル"] == ""))
    ].copy()
    # 数量bkを使用
    accessory["数量"] = accessory["数量bk"]
    # グループ化
    result = (
        accessory.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def process_ido_data(
    df: pd.DataFrame, masters: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    移動データの完全処理（Power Query準拠）
    """
    # ★修正前
    # if df is None or df.empty:
    #     return pd.DataFrame()

    # ★修正後
    if df is None:
        return pd.DataFrame()

    # 型変換
    df["入庫予定数"] = (
        pd.to_numeric(df["入庫予定数"], errors="coerce").fillna(0).astype(int)
    )
    df["未入庫数"] = (
        pd.to_numeric(df["未入庫数"], errors="coerce").fillna(0).astype(int)
    )
    # （以下は既存のまま）
    # 不要な列を削除（空の列）
    df = df.loc[:, ~df.columns.str.startswith("_")]
    df = df.loc[:, df.columns != ""]
    # 移動元取次店コードでマスタ857001とマージ（Inner Join）
    if "857001" in masters:
        moto_master = masters["857001"].copy()
        moto_master = moto_master.rename(
            columns={
                "取次店コード": "移動元取次店コード",
                "店舗倉庫区分": "店舗倉庫区分",
                "事業CD": "移動元事業CD",
                "保管場所CD": "移動元保管場所CD",
            }
        )
        df = df.merge(moto_master, on="移動元取次店コード", how="inner")
    # 移動先取次店コードでマスタ857001とマージ（Inner Join）
    if "857001" in masters:
        saki_master = masters["857001"][["取次店コード", "事業CD", "保管場所CD"]].copy()
        saki_master = saki_master.rename(
            columns={
                "取次店コード": "移動先取次店コード",
                "事業CD": "移動先事業CD",
                "保管場所CD": "移動先保管場所CD",
            }
        )
        df = df.merge(saki_master, on="移動先取次店コード", how="inner")
    # マスタ857002とマージ
    if "857002" in masters:
        # 型を統一（文字列型に変換）
        df["移動元事業CD"] = df["移動元事業CD"].astype(str)
        master_857002 = masters["857002"].copy()
        master_857002["事業CD"] = master_857002["事業CD"].astype(str)
        df = df.merge(
            master_857002,
            left_on=["移動元事業CD", "商品コード"],
            right_on=["事業CD", "商品コード"],
            how="left",
            suffixes=("", "_master"),
        )
        df["TMS商品CD"] = df["TMS商品CD"].fillna(df["商品コード"])
        if "事業CD_master" in df.columns:
            df = df.drop(columns=["事業CD_master"])
    else:
        df["TMS商品CD"] = df["商品コード"]
    return df


def process_ido_shukko(df: pd.DataFrame) -> pd.DataFrame:
    """移動データ（出庫）"""
    # カテゴリ中がＵＳＩＭカードでない
    shukko = df[~df["カテゴリ中"].str.contains("ＵＳＩＭカード", na=False)].copy()
    # 数量計算: 入庫予定数 * -1
    shukko["数量"] = shukko["入庫予定数"] * -1
    shukko["取次店コード"] = shukko["移動元取次店コード"]
    shukko["取次店名"] = shukko["移動元取次店名"]
    # グループ化
    result = (
        shukko.groupby(
            [
                "取次店コード",
                "取次店名",
                "移動元事業CD",
                "移動元保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(
        columns={
            "移動元事業CD": "事業CD",
            "移動元保管場所CD": "保管場所CD",
            "数量": "変動数",
        }
    )
    # 保管場所CDが空でない
    result = result[result["保管場所CD"] != ""]
    return result


def process_ido_nyuko(df: pd.DataFrame) -> pd.DataFrame:
    """移動データ（入庫）"""
    # カテゴリ中がＵＳＩＭカードでない
    nyuko = df[~df["カテゴリ中"].str.contains("ＵＳＩＭカード", na=False)].copy()
    # 数量計算: 入庫予定数 * 1
    nyuko["数量"] = nyuko["入庫予定数"] * 1
    nyuko["取次店コード"] = nyuko["移動先取次店コード"]
    nyuko["取次店名"] = nyuko["移動先取次店名"]
    # グループ化
    result = (
        nyuko.groupby(
            [
                "取次店コード",
                "取次店名",
                "移動先事業CD",
                "移動先保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(
        columns={
            "移動先事業CD": "事業CD",
            "移動先保管場所CD": "保管場所CD",
            "数量": "変動数",
        }
    )
    # 保管場所CDが空でない
    result = result[result["保管場所CD"] != ""]
    return result


def process_uri_data(
    df: pd.DataFrame, masters: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    売上データの完全処理（Power Query準拠）
    """
    # ★修正前
    # if df is None or df.empty:
    #     return pd.DataFrame()

    # ★修正後
    if df is None:
        return pd.DataFrame()

    # マスタ857001とマージ（Left Outer Join）
    if "857001" in masters:
        df = df.merge(masters["857001"], on="取次店コード", how="left")
    # （以下は既存のまま）
    # 商品構成マスタ_13000（パック商品CD→内訳商品CD）で展開（PowerQuery準拠）
    # - 商品コードがパック商品CDに該当する場合、内訳商品CDに置き換える（行は内訳数だけ増える）
    if "13000" in masters:
        m13000 = masters["13000"].copy()
        # 念のため必須列が揃っているか確認
        if "パック商品CD" in m13000.columns and "内訳商品CD" in m13000.columns:
            df = df.merge(
                m13000, left_on="商品コード", right_on="パック商品CD", how="left"
            )
    df["商品コードbk"] = df["商品コード"]
    if "内訳商品CD" in df.columns:
        df["商品コード"] = df["内訳商品CD"].where(
            (df["内訳商品CD"].notna()) & (df["内訳商品CD"] != ""), df["商品コードbk"]
        )
    # マスタ857002とマージ
    if "857002" in masters:
        # 型を統一（文字列型に変換）
        df["事業CD"] = df["事業CD"].astype(str)
        master_857002 = masters["857002"].copy()
        master_857002["事業CD"] = master_857002["事業CD"].astype(str)
        df = df.merge(
            master_857002,
            on=["事業CD", "商品コード"],
            how="left",
            suffixes=("", "_master"),
        )
        df["TMS商品CD"] = df["TMS商品CD"].fillna(df["商品コード"])
    else:
        df["TMS商品CD"] = df["商品コード"]
    # 数量計算: 収納種別が「販売」なら-1、それ以外は1
    df["数量"] = df["収納種別"].apply(lambda x: -1 if x == "販売" else 1)
    return df


def process_uri_individual(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（個体情報）"""
    # 必要な列が無い場合は空文字列列を作成（入力データによっては列名が存在しないことがある）
    needed = [
        "メーカー",
        "業務種別",
        "店舗名",
        "取次店コード",
        "事業CD",
        "保管場所CD",
        "商品コード",
        "TMS商品CD",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
    # メーカーがある かつ Apple Inc.-SBS と ｱｯﾌﾟﾙ-SBS 以外
    individual = df[
        (df["メーカー"].notna())
        & (df["メーカー"] != "")
        & (df["メーカー"] != "Apple Inc.-SBS")
        & (df["メーカー"] != "ｱｯﾌﾟﾙ-SBS")
    ].copy()
    # 業務種別にＵＳＩＭを含まない
    individual = individual[~individual["業務種別"].str.contains("ＵＳＩＭ", na=False)]
    individual["取次店名"] = individual["店舗名"]
    # 数量列が無ければ作成（通常は process_uri_data で作成される）
    if "数量" not in individual.columns:
        if "収納種別" in individual.columns:
            individual["数量"] = individual["収納種別"].apply(
                lambda x: -1 if x == "販売" else 1
            )
        else:
            individual["数量"] = 0
    # グループ化
    result = (
        individual.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def process_uri_sb_accessory(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（SBアクセサリ）"""
    # 必要な列が無い場合は空文字列列を作成
    if "取次店コード" not in df.columns:
        df["取次店コード"] = ""
    if "メーカー" not in df.columns:
        df["メーカー"] = ""
    # 取次店コードがTGで始まる
    sb_acc = df[df["取次店コード"].str.startswith("TG", na=False)].copy()
    # メーカーが Apple Inc.-SBS または ｱｯﾌﾟﾙ-SBS ← 最小修正：OR を追加
    sb_acc = sb_acc[
        (sb_acc["メーカー"] == "Apple Inc.-SBS") | (sb_acc["メーカー"] == "ｱｯﾌﾟﾙ-SBS")
    ]
    sb_acc["取次店名"] = sb_acc["店舗名"] if "店舗名" in sb_acc.columns else ""
    # 数量列が無ければ作成
    if "数量" not in sb_acc.columns:
        if "収納種別" in sb_acc.columns:
            sb_acc["数量"] = sb_acc["収納種別"].apply(
                lambda x: -1 if x == "販売" else 1
            )
        else:
            sb_acc["数量"] = 0
    # グループ化
    result = (
        sb_acc.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def process_uri_service(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（サービス）"""
    # 必要な列が無い場合は空文字列列を作成
    if "商品分類" not in df.columns:
        df["商品分類"] = ""
    if "店舗名" not in df.columns:
        df["店舗名"] = ""
    # 商品分類が「サービス」
    service = df[df["商品分類"] == "サービス"].copy()
    service["取次店名"] = service["店舗名"]
    # 数量列が無ければ作成
    if "数量" not in service.columns:
        if "収納種別" in service.columns:
            service["数量"] = service["収納種別"].apply(
                lambda x: -1 if x == "販売" else 1
            )
        else:
            service["数量"] = 0
    # グループ化
    result = (
        service.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def process_tana_data(
    df: pd.DataFrame, masters: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    棚卸データの完全処理（Power Query準拠）
    """
    # ★修正前
    # if df is None or df.empty:
    #     return pd.DataFrame()

    # ★修正後
    if df is None:
        return pd.DataFrame()

    # マスタ857001とマージ（Inner Join）
    if "857001" in masters:
        df = df.merge(masters["857001"], on="取次店コード", how="inner")
    # 型変換
    df["受払前在庫数"] = (
        pd.to_numeric(df["受払前在庫数"], errors="coerce").fillna(0).astype(int)
    )
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0).astype(int)
    df["受払後在庫数"] = (
        pd.to_numeric(df["受払後在庫数"], errors="coerce").fillna(0).astype(int)
    )
    # （以下は既存のまま）
    # 不要な列を削除
    df = df.loc[:, ~df.columns.str.startswith("_")]
    df = df.loc[:, df.columns != ""]
    # 店舗倉庫区分でフィルタ
    df = df[df["店舗倉庫区分"] == "1"]
    # マスタ857002とマージ
    if "857002" in masters:
        # 型を統一（文字列型に変換）
        df["事業CD"] = df["事業CD"].astype(str)
        master_857002 = masters["857002"].copy()
        master_857002["事業CD"] = master_857002["事業CD"].astype(str)
        df = df.merge(
            master_857002,
            on=["事業CD", "商品コード"],
            how="left",
            suffixes=("", "_master"),
        )
        df["TMS商品CD"] = df["TMS商品CD"].fillna(df["商品コード"])
    else:
        df["TMS商品CD"] = df["商品コード"]
    # 除外商品コード
    exclude_codes = [
        "ZUA292",
        "ZUA34Q",
        "ZUA34R",
        "ZUA34S",
        "ZUA34T",
        "ZUA34U",
        "ZUA34V",
        "ZUA34W",
    ]
    for code in exclude_codes:
        df = df[~df["商品コード"].str.contains(code, na=False)]
    return df


def process_tana_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """棚卸データ（グループ化）"""
    # カテゴリ中がＵＳＩＭカードでない
    grouped = df[~df["カテゴリ中"].str.contains("ＵＳＩＭカード", na=False)].copy()
    # グループ化
    result = (
        grouped.groupby(
            [
                "取次店コード",
                "取次店名",
                "事業CD",
                "保管場所CD",
                "商品コード",
                "TMS商品CD",
            ],
            dropna=False,
        )["数量"]
        .sum()
        .reset_index()
    )
    result = result.rename(columns={"数量": "変動数"})
    return result


def combine_all_data(
    shiire_ind,
    shiire_acc,
    ido_shukko,
    ido_nyuko,
    uri_ind,
    uri_sb,
    uri_service,
    tana_grouped,
) -> pd.DataFrame:
    """
    全データを結合してTMS商品CDで集計（GINIEPOS変動数）
    """
    all_dfs = []
    for df in [
        shiire_ind,
        shiire_acc,
        ido_shukko,
        ido_nyuko,
        uri_ind,
        uri_sb,
        uri_service,
        tana_grouped,
    ]:
        if df is not None and not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    # 全て結合
    combined = pd.concat(all_dfs, ignore_index=True)
    # TMS商品CDでグループ化して合計
    result = (
        combined.groupby(
            ["取次店コード", "取次店名", "事業CD", "保管場所CD", "TMS商品CD"],
            dropna=False,
        )["変動数"]
        .sum()
        .reset_index()
    )
    # ソート
    result = result.sort_values("取次店コード").reset_index(drop=True)
    return result


def compare_with_current_inventory(
    giniepos_df: pd.DataFrame,
    current_df: pd.DataFrame,
    master_857003: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    GINIEPOS変動数と現在庫照会を比較して
    ・在庫不足
    ・突合不可
    を判定して返す
    """

    # ---- 現在庫照会から必要な列のみ取得 ----
    required_cols = ["保管場所CD", "事業CD", "商品CD", "実在庫数量"]
    missing = [c for c in required_cols if c not in current_df.columns]
    if missing:
        raise KeyError(
            f"現在庫照会ファイルに必要な列が見つかりません: {missing} / 実列={list(current_df.columns)}"
        )

    current_summary = current_df[required_cols].copy()

    # ---- キー列の正規化（PowerQuery の Trim 相当）----
    for c in ["事業CD", "保管場所CD", "商品CD"]:
        current_summary[c] = normalize_key_series(current_summary[c])
    for c in ["事業CD", "保管場所CD", "TMS商品CD"]:
        giniepos_df[c] = normalize_key_series(giniepos_df[c])

    # ---- 在庫数量の文字→数値化 ----
    current_summary["CL実在庫数"] = (
        current_summary["実在庫数量"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("　", " ", regex=False)
        .str.strip()
    )
    current_summary["CL実在庫数"] = pd.to_numeric(
        current_summary["CL実在庫数"], errors="coerce"
    ).fillna(0)

    # ---- マージ（Left Outer Join）----
    result = giniepos_df.merge(
        current_summary[["保管場所CD", "事業CD", "商品CD", "CL実在庫数"]],
        left_on=["保管場所CD", "事業CD", "TMS商品CD"],
        right_on=["保管場所CD", "事業CD", "商品CD"],
        how="left",
    )

    # 「現在庫に行が存在するかどうか」は商品CDの有無で判定
    unmatched_current = result["商品CD"].isna()

    # CL実在庫数はここでもう一度数値化して、無いところは 0 として扱う
    result["CL実在庫数"] = pd.to_numeric(result["CL実在庫数"], errors="coerce").fillna(
        0
    )

    # ---- 857003（事業CD×TMS商品CD）存在チェック ----
    exist_in_857003 = pd.Series(False, index=result.index)

    if master_857003 is not None and not master_857003.empty:
        m857 = master_857003.copy()

        jigyo_candidates = ["事業CD", "変換前コード値02"]
        tms_candidates = ["TMS商品CD", "変換前コード値01"]

        jigyo_col = next((c for c in jigyo_candidates if c in m857.columns), None)
        tms_col = next((c for c in tms_candidates if c in m857.columns), None)

        if jigyo_col is not None and tms_col is not None:
            key_df = m857[[jigyo_col, tms_col]].copy()
            key_df.columns = ["事業CD", "TMS商品CD"]

            key_df["事業CD"] = normalize_key_series(key_df["事業CD"])
            key_df["TMS商品CD"] = normalize_key_series(key_df["TMS商品CD"])
            key_df = key_df.drop_duplicates()

            key_set = set(
                (str(j), str(t)) for j, t in zip(key_df["事業CD"], key_df["TMS商品CD"])
            )

            res_jigyo = normalize_key_series(result["事業CD"])
            res_tms = normalize_key_series(result["TMS商品CD"])
            res_keys = list(zip(res_jigyo.astype(str), res_tms.astype(str)))

            exist_in_857003 = pd.Series(
                [(k in key_set) for k in res_keys],
                index=result.index,
            )

    # ---- 判定（変動数 + 在庫）を全行で計算 ----
    result["判定"] = result["変動数"] + result["CL実在庫数"]

    # 判定区分初期値は OK
    result["判定区分"] = "OK"

    # 在庫不足：現在庫の有無に関係なく、合計がマイナスなら在庫不足
    result.loc[result["判定"] < 0, "判定区分"] = "在庫不足"

    # TMS商品CD が Z00014 / POS- を含むもの（突合不可除外対象）
    tms_series = result["TMS商品CD"].fillna("").astype(str)
    exclude_for_unmatch = tms_series.str.contains(
        "Z00014", na=False
    ) | tms_series.str.contains("POS-", na=False)

    # 新しい「突合不可」条件:
    #   現在庫にない かつ 857003 にもない かつ Z00014/POS- を含まない
    #   かつ すでに在庫不足と判定されていない行
    result.loc[
        unmatched_current
        & (~exist_in_857003)
        & (~exclude_for_unmatch)
        & (result["判定区分"] != "在庫不足"),
        "判定区分",
    ] = "突合不可"

    tms_series = result['TMS商品CD'].fillna("").astype(str)
    # 除外（PowerQuery 準拠）
    result = result[~result["TMS商品CD"].str.contains("BB-RQ8POU1740", na=False)]
    result = result[~result["TMS商品CD"].str.contains("ZUA292", na=False)]

    # さらに除外（最終結果から Z00014 / POS- を除外：既存仕様踏襲）
    result = result[~result["TMS商品CD"].str.contains("Z00014", na=False)]
    result = result[~result["TMS商品CD"].str.contains("POS-", na=False)]

    # 返却は「在庫不足」または「突合不可」のみ（OK は返さない）
    result = result[result["判定区分"].isin(["在庫不足", "突合不可"])]

    return result


def _decode_upload_text(upload_file) -> str:
    data = upload_file.getvalue()
    try:
        return data.decode("utf-8-sig")
    except Exception:
        return data.decode("cp932", errors="replace")


def _build_charge_error_df(err_details: List[ErrorDetail]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "行番号(物理行)": d.row,
                "店舗名": d.store_name,
                "伝票番号": d.slip_number,
                "金額(38列目)": d.col_38,
            }
            for d in err_details
        ]
    )


def _build_date_issue_df(date_summary: DateSummary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "レコード番号": it.record_no,
                "開始行(物理行)": it.start_physical_line,
                "終了行(物理行)": it.start_physical_line,  # 旧ロジック互換（終了行は開始行と同じ扱い）
                "9列目の値": it.col9,
                "補足": it.note,
            }
            for it in date_summary.issues
        ]
    )


def _run_uri_prechecks(uri_file) -> Tuple[Dict[str, Any], str]:
    result_part = {
        "charge": {"status": "未実行", "table": None, "message": ""},
        "date": {"status": "未実行", "table": None, "message": ""},
    }

    uri_text = _decode_upload_text(uri_file)
    err_flag, err_details, _, _, date_summary = check_and_analyze(uri_text)

    # チャージ金額チェック
    if err_flag:
        df_err = _build_charge_error_df(err_details)
        result_part["charge"]["status"] = "NG"
        result_part["charge"]["table"] = df_err
        result_part["charge"]["message"] = f"NG {len(df_err)}件"
    else:
        result_part["charge"]["status"] = "OK"
        result_part["charge"]["message"] = "OK"

    # 日付チェック
    if date_summary.count_error > 0:
        df_issue = _build_date_issue_df(date_summary)
        result_part["date"]["status"] = "NG"
        result_part["date"]["table"] = df_issue
        result_part["date"]["message"] = f"NG {len(df_issue)}件"
    else:
        result_part["date"]["status"] = "OK"
        result_part["date"]["message"] = "OK"

    return result_part, uri_text


def _run_inventory_check(
    shiire_file,
    ido_file,
    uri_text: str,
    tana_file,
    current_file,
    master_857001_file,
    master_857002_file,
    master_857003_file,
    master_13000_file,
) -> pd.DataFrame:
    masters = load_master_files(
        master_857001_file, master_857002_file, master_857003_file, master_13000_file
    )

    shiire_df = load_csv_with_encoding(shiire_file, use_lf=False, encoding="cp932")
    ido_df = load_csv_with_encoding(ido_file, use_lf=False, encoding="cp932")
    tana_df = load_csv_with_encoding(tana_file, use_lf=False, encoding="cp932")

    uri_df = load_csv_from_text(uri_text)
    current_df = load_current_inventory_excel(current_file)

    shiire_processed = process_shiire_data(shiire_df, masters)
    shiire_individual = process_shiire_individual(shiire_processed)
    shiire_accessory = process_shiire_accessory(shiire_processed)

    ido_processed = process_ido_data(ido_df, masters)
    ido_shukko = process_ido_shukko(ido_processed)
    ido_nyuko = process_ido_nyuko(ido_processed)

    uri_processed = process_uri_data(uri_df, masters)
    uri_individual = process_uri_individual(uri_processed)
    uri_sb = process_uri_sb_accessory(uri_processed)
    uri_service = process_uri_service(uri_processed)

    tana_processed = process_tana_data(tana_df, masters)
    tana_grouped = process_tana_grouped(tana_processed)

    merged_hendo = combine_all_data(
        shiire_individual,
        shiire_accessory,
        ido_shukko,
        ido_nyuko,
        uri_individual,
        uri_sb,
        uri_service,
        tana_grouped,
    )

    # まずは通常どおり在庫不足／突合不可判定を実行
    inv_result = compare_with_current_inventory(
        merged_hendo, current_df, masters.get("857003")
    )

    # --- ここが今回の追加ロジック ---
    # 最下流の段階で「倉庫（コード値1=2）」を在庫不足チェック対象から除外する
    if inv_result is not None and not inv_result.empty:
        m857001 = masters.get("857001")
        if (
            m857001 is not None
            and not m857001.empty
            and "取次店コード" in m857001.columns
        ):
            # 857001側：取次店コード＋店舗倉庫区分だけ使う
            m857 = m857001[["取次店コード", "店舗倉庫区分"]].copy()
            m857["取次店コード"] = normalize_key_series(m857["取次店コード"])

            # 在庫不足結果側にも正規化キーを持たせて結合
            if "取次店コード" in inv_result.columns:
                inv = inv_result.copy()
                inv["取次店コード_norm"] = normalize_key_series(inv["取次店コード"])

                inv = inv.merge(
                    m857,
                    left_on="取次店コード_norm",
                    right_on="取次店コード",
                    how="left",
                    suffixes=("", "_m857"),
                )

                # 店舗倉庫区分 = '2'（倉庫） かつ 判定区分 = '在庫不足' を除外
                mask_warehouse_shortage = (inv.get("判定区分") == "在庫不足") & (
                    inv.get("店舗倉庫区分") == "2"
                )
                inv = inv[~mask_warehouse_shortage]

                # 補助列を削除（画面側には出さない）
                inv = inv.drop(
                    columns=["取次店コード_norm", "取次店コード_m857"], errors="ignore"
                )

                inv_result = inv

    return inv_result


def run_full_check(
    shiire_file,
    ido_file,
    uri_file,
    tana_file,
    current_file,
    master_857001_file,
    master_857002_file,
    master_857003_file,
    master_13000_file,
    progress_cb=None,
):
    result = {
        "charge": {
            "label": "チャージ金額チェック",
            "status": "未実行",
            "table": None,
            "message": "",
        },
        "date": {
            "label": "売上日付チェック",
            "status": "未実行",
            "table": None,
            "message": "",
        },
        "inv": {
            "label": "在庫不足チェック",
            "status": "未実行",
            "table": None,
            "message": "",
        },
    }

    def _progress(msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    # ---- URI事前チェック ----
    try:
        _progress("売上CSVの事前チェック中...")
        uri_part, uri_text = _run_uri_prechecks(uri_file)
        for k in ["charge", "date"]:
            result[k].update(uri_part[k])

        if result["charge"]["status"] == "NG" or result["date"]["status"] == "NG":
            return result

    except Exception as e:
        result["charge"]["status"] = "NG"
        result["charge"]["message"] = f"売上ファイル解析エラー: {e}"
        return result

    # ---- 在庫不足チェック ----
    try:
        _progress("在庫不足チェック中...")
        inv_result = _run_inventory_check(
            shiire_file,
            ido_file,
            uri_text,
            tana_file,
            current_file,
            master_857001_file,
            master_857002_file,
            master_857003_file,
            master_13000_file,
        )

        if inv_result is not None and not inv_result.empty:
            result["inv"]["status"] = "NG"
            # 件数内訳（在庫不足 / 突合不可）
            shortage_cnt = (
                int((inv_result.get("判定区分") == "在庫不足").sum())
                if "判定区分" in inv_result.columns
                else len(inv_result)
            )
            unmatched_cnt = (
                int((inv_result.get("判定区分") == "突合不可").sum())
                if "判定区分" in inv_result.columns
                else 0
            )
            result["inv"]["table"] = inv_result
            if "判定区分" in inv_result.columns:
                result["inv"][
                    "message"
                ] = f"在庫不足 {shortage_cnt}件 / 突合不可 {unmatched_cnt}件"
            else:
                result["inv"]["message"] = f"NG {len(inv_result)}件"
        else:
            result["inv"]["status"] = "OK"
            result["inv"]["table"] = inv_result
            result["inv"]["message"] = "OK"

    except KeyError as e:
        # 必須列不足など：判定NGではなく「処理エラー」と明示（社内公開向け）
        result["inv"]["status"] = "NG"
        result["inv"]["table"] = pd.DataFrame()
        result["inv"]["message"] = f"在庫不足チェック（処理エラー）: {e}"

    except Exception as e:
        result["inv"]["status"] = "NG"
        result["inv"]["table"] = pd.DataFrame()
        result["inv"]["message"] = f"在庫不足チェック（処理エラー）: {e}"

    return result


# ファイルアップロードセクション
st.markdown(
    """
<style>
/* アップロード見出しをコンパクトに */
.precheck-upload-title{
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0.2rem 0 0.6rem 0;
  line-height: 1.2;
}
.precheck-step{
  display:inline-block;
  font-size:0.95rem;
  font-weight:700;
  padding:2px 10px;
  border-radius:999px;
  background:#eef2ff;
  margin-right:10px;
}
</style>
<div class="precheck-upload-title"><span class="precheck-step">1</span>ファイルアップロード</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "9ファイルの解析は、PCの負荷やファイルサイズによって30〜60秒以上かかる場合があります。反応が遅くても少し待ってください。"
)

# file_uploader を広くする（落としやすくする）
st.markdown(
    """
<style>
/* Make the file drop zone taller and easier to use (robust across Streamlit versions) */
div[data-testid="stFileUploader"]{
    min-height: 240px;
}
div[data-testid="stFileUploader"] > div{
    min-height: 240px;
}
div[data-testid="stFileUploader"] section{
    padding-top: 24px;
    padding-bottom: 24px;
}
div[data-testid="stFileUploader"] section > div{
    min-height: 240px;
    display: flex;
    align-items: center;
}
/* Newer Streamlit builds use a dedicated dropzone testid */
div[data-testid="stFileUploaderDropzone"]{
    min-height: 240px !important;
    padding-top: 24px;
    padding-bottom: 24px;
    display: flex;
    align-items: center;
}
</style>
""",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "ファイルを選択またはドラッグ&ドロップ",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="在庫変動データ4ファイル + 現在庫照会 + マスタ4ファイル = 計9ファイル",
    key=f"uploader_{st.session_state.uploader_version}",
)

# クリア（結果だけ消して初期状態を戻す）
if st.button(
    "🔄 クリア",
    key="clear_btn",
    help="結果とアップロード状態をクリアして初期表示に戻します",
):
    st.session_state.processed_data = None
    st.session_state.last_full_sig = None
    st.session_state.uploader_version += 1
    safe_rerun()

# ファイル名からファイルを振り分け
shiire_file = None
ido_file = None
uri_file = None
tana_file = None
current_file = None
master_857001_file = None
master_857002_file = None
master_857003_file = None
master_13000_file = None

if uploaded_files:
    for file in uploaded_files:
        filename = file.name
        if "SHI" in filename.upper():
            shiire_file = file
        elif "IDO" in filename.upper():
            ido_file = file
        elif "URI" in filename.upper():
            uri_file = file
        elif "TNA" in filename.upper():
            tana_file = file
        elif "現在庫" in filename or "ZAIKO" in filename.upper():
            current_file = file
        elif "857001" in filename:
            master_857001_file = file
        elif "857002" in filename:
            master_857002_file = file
        elif "857003" in filename:
            master_857003_file = file
        elif "13000" in filename:
            master_13000_file = file
        else:
            st.warning(f"⚠️ 不明なファイル: {filename}")

# 必要ファイル数チェック
total_files = sum(
    [
        shiire_file is not None,
        ido_file is not None,
        uri_file is not None,
        tana_file is not None,
        current_file is not None,
        master_857001_file is not None,
        master_857002_file is not None,
        master_857003_file is not None,
        master_13000_file is not None,
    ]
)

if total_files < 9:
    st.warning(f"⚠️ {total_files}/9ファイルが認識されました。全9ファイル必要です。")
else:
    # 9ファイル揃ったら自動で処理（途中ログは出さない）
    # 同じファイルセットで二重実行しない
    sig_parts = [
        getattr(shiire_file, "name", None),
        getattr(ido_file, "name", None),
        getattr(uri_file, "name", None),
        getattr(tana_file, "name", None),
        getattr(current_file, "name", None),
        getattr(master_857001_file, "name", None),
        getattr(master_857002_file, "name", None),
        getattr(master_857003_file, "name", None),
    ]
    sig = tuple(sig_parts)
    if "last_full_sig" not in st.session_state:
        st.session_state.last_full_sig = None

    if st.session_state.last_full_sig != sig or st.session_state.processed_data is None:
        st.session_state.last_full_sig = sig
        progress_box = st.empty()

        def _ui_progress(msg: str) -> None:
            progress_box.info(msg)

        with st.spinner("解析中です（30〜60秒かかる場合があります）..."):
            st.session_state.processed_data = run_full_check(
                shiire_file,
                ido_file,
                uri_file,
                tana_file,
                current_file,
                master_857001_file,
                master_857002_file,
                master_857003_file,
                master_13000_file,
                progress_cb=_ui_progress,
            )
        st.success("処理が完了しました")
        progress_box.empty()

# ----------------------------
# 確認結果（3つのチェックを左→右で表示）
# ----------------------------
if st.session_state.processed_data:
    st.markdown("---")
    st.header("2️⃣ 確認結果")

    data = st.session_state.processed_data

    def _status_badge(status: str) -> str:
        if status == "OK":
            return "<span style='font-size:42px; font-weight:800; color:#1f7a1f;'>OK</span>"
        if status == "NG":
            return "<span style='font-size:42px; font-weight:800; color:#d11a2a;'>NG</span>"
        return (
            "<span style='font-size:42px; font-weight:800; color:#666;'>未実行</span>"
        )

    cols = st.columns(3)
    for i, key in enumerate(["charge", "date", "inv"]):
        with cols[i]:
            st.markdown(f"**{data[key]['label']}**", unsafe_allow_html=True)
            st.markdown(_status_badge(data[key]["status"]), unsafe_allow_html=True)
            if data[key].get("message"):
                st.caption(data[key]["message"])

    # エラー詳細（NGのみ、表示順も左→右）
    st.markdown("---")
    any_ng = any(data[k]["status"] == "NG" for k in ["charge", "date", "inv"])
    if any_ng:
        st.subheader("📌 エラー詳細")
        for key in ["charge", "date", "inv"]:
            if data[key]["status"] != "NG":
                continue
            st.markdown(f"### {data[key]['label']}")
            tbl = data[key].get("table")
            if tbl is None:
                st.write(data[key].get("message", ""))
                continue
            if hasattr(tbl, "empty") and tbl.empty:
                st.write("（該当なし）")
                continue
            st.dataframe(tbl, width="stretch", height=300)
    else:
        st.success("✅ すべてOKです")
