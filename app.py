import streamlit as st
import pandas as pd
import io
import unicodedata
from typing import Dict, List
import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, date
from io import StringIO
from typing import Optional, Tuple

# ページ設定
st.set_page_config(
    page_title="在庫不足チェックシステム",
    page_icon="📦",
    layout="wide"
)

st.title("📦 在庫不足チェックシステム")
st.markdown("---")

# セッションステートの初期化
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None


def safe_rerun() -> None:
    """Rerun the Streamlit script in a way compatible with multiple Streamlit versions."""
    # Preferred API
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass

    # Fallback: raise the internal RerunException
    try:
        from streamlit.runtime.scriptrunner.script_runner import RerunException
        raise RerunException()
    except Exception:
        # As last resort, use stop to prevent further UI actions
        try:
            st.stop()
        except Exception:
            return

def load_csv_with_encoding(file, use_lf=True, encoding='cp932') -> pd.DataFrame:
    """
    CSVファイルを読み込む（エンコーディングと改行コードを指定）
    """
    try:
        content = file.read()
        decoded_content = content.decode(encoding)
        if use_lf:
            df = pd.read_csv(io.StringIO(decoded_content), lineterminator='\n')
        else:
            df = pd.read_csv(io.StringIO(decoded_content))
        return df
    except Exception as e:
        st.error(f"CSVファイルの読み込みエラー: {str(e)}")
        return None

def load_master_files(master_857001, master_857002, master_857003) -> Dict[str, pd.DataFrame]:
    """
    3つのマスタファイルを読み込む（UTF-8、CRLF改行）
    """
    masters = {}
    
    if master_857001:
        df = load_csv_with_encoding(master_857001, use_lf=False, encoding='utf-8')
        if df is not None:
            # マスタ857001の加工
            df = df[['変換前コード値01', 'コード値1', 'コード値2', 'コード値4']].copy()
            df = df.rename(columns={
                '変換前コード値01': '取次店コード',
                'コード値1': '店舗倉庫区分',
                'コード値2': '事業CDｂｋ',
                'コード値4': '保管場所CD'
            })
            # 事業CDの計算: TGで始まるなら13000、それ以外は15000
            df['事業CD'] = df['取次店コード'].apply(lambda x: '13000' if str(x).startswith('TG') else '15000')
            df = df.drop_duplicates(subset=['取次店コード'])
            masters['857001'] = df
            st.success(f"✅ マスタ857001読み込み完了: {len(df)}行")
    
    if master_857002:
        df = load_csv_with_encoding(master_857002, use_lf=False, encoding='utf-8')
        if df is not None:
            # マスタ857002の加工
            df = df[['変換前コード値01', '変換前コード値02', 'コード値1']].copy()
            df = df.rename(columns={
                '変換前コード値01': '商品コード',
                '変換前コード値02': '事業CD',
                'コード値1': 'TMS商品CD'
            })
            df = df.drop_duplicates(subset=['商品コード', '事業CD'])
            masters['857002'] = df
            st.success(f"✅ マスタ857002読み込み完了: {len(df)}行")
    
    if master_857003:
        df = load_csv_with_encoding(master_857003, use_lf=False, encoding='utf-8')
        if df is not None:
            masters['857003'] = df
            st.success(f"✅ マスタ857003読み込み完了: {len(df)}行")
    
    return masters


### --- 以下 main.py から取り込んだユーティリティ関数 (売上前処理用) ---
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
    return unicodedata.normalize("NFKC", s)


def split_docomo_shop_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    w.writerow(["レコード番号", "開始物理行(参考)", "重要度", "種別", "9列目", "17列目", "補足"])
    for it in issues:
        w.writerow([it.record_no, it.start_physical_line, it.severity, it.issue_type, it.col9, it.col17, it.note])
    return buf.getvalue().encode("utf-8")


def check_and_analyze(csv_text: str) -> Tuple[bool, List[ErrorDetail], int, int, DateSummary]:
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
                error_details.append(ErrorDetail(row=start_physical_line, store_name=col_3, slip_number=col_11, col_38=col_38))

        # date checks
        col9 = row[DATE_COL_9].strip() if len(row) > DATE_COL_9 else ""
        dt9 = parse_dt_str(col9)
        if dt9 is None:
            count_error += 1
            issues.append(DateIssue(record_no=total_data_records, start_physical_line=start_physical_line, severity="ERROR", issue_type="COL9_MISSING_OR_INVALID", col9=col9, col17="", note="9列目に yyyy/mm/dd hh:mm:ss が必要です。"))
        else:
            count_col9_ok += 1
            total_checked_cells += 1

    date_summary = DateSummary(total_checked_cells=total_checked_cells, count_col9_ok=count_col9_ok, count_error=count_error, issues=issues)

    return (len(error_details) > 0), error_details, total_data_records, total_physical_lines, date_summary

# --- end of main.py checks ---

### --- 取り込みここまで ---

def process_shiire_data(df: pd.DataFrame, masters: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    仕入データの完全処理（Power Query準拠）
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 型変換
    df['受払前在庫数'] = pd.to_numeric(df['受払前在庫数'], errors='coerce').fillna(0).astype(int)
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').fillna(0).astype(int)
    df['受払後在庫数'] = pd.to_numeric(df['受払後在庫数'], errors='coerce').fillna(0).astype(int)
    
    # マスタ857001とマージ（Inner Join）
    if '857001' in masters:
        df = df.merge(masters['857001'], on='取次店コード', how='inner')
    
    # 店舗倉庫区分でフィルタ
    df = df[df['店舗倉庫区分'] == '1']
    
    # マスタ857002とマージ（Left Outer Join）
    if '857002' in masters:
        # 型を統一（文字列型に変換）
        df['事業CD'] = df['事業CD'].astype(str)
        master_857002 = masters['857002'].copy()
        master_857002['事業CD'] = master_857002['事業CD'].astype(str)
        
        df = df.merge(master_857002, on=['事業CD', '商品コード'], how='left', suffixes=('', '_master'))
        # TMS商品CDがnullなら商品コードを使用
        df['TMS商品CD'] = df['TMS商品CD'].fillna(df['商品コード'])
    else:
        df['TMS商品CD'] = df['商品コード']
    
    # 数量をバックアップ
    df['数量bk'] = df['数量']
    
    # 受払種別でフィルタ
    valid_types = ['倉庫へ返品', '入荷', '入荷(システム自動)', '返品キャンセル', '返品不備']
    df = df[df['受払種別'].isin(valid_types)]
    
    # 除外商品コード
    exclude_codes = ['ZUA292', 'ZUA34Q', 'ZUA34R', 'ZUA34S', 'ZUA34T', 'ZUA34U', 'ZUA34V', 'ZUA34W']
    for code in exclude_codes:
        df = df[~df['商品コード'].str.contains(code, na=False)]
    
    return df

def process_shiire_individual(df: pd.DataFrame) -> pd.DataFrame:
    """仕入データ（個体情報）"""
    # IMEI、ICCID、その他シリアルのいずれかがある
    individual = df[
        (df['IMEI'].notna() & (df['IMEI'] != '')) |
        (df['ICCID'].notna() & (df['ICCID'] != '')) |
        (df['その他シリアル'].notna() & (df['その他シリアル'] != ''))
    ].copy()
    
    # カテゴリ中がＵＳＩＭカードでない
    individual = individual[individual['カテゴリ中'] != 'ＵＳＩＭカード']
    
    # 数量計算: 倉庫へ返品なら-1、それ以外は1
    individual['数量'] = individual['受払種別'].apply(lambda x: -1 if x == '倉庫へ返品' else 1)
    
    # グループ化
    result = individual.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_shiire_accessory(df: pd.DataFrame) -> pd.DataFrame:
    """仕入データ（アクセサリ）"""
    # IMEI、ICCID、その他シリアル全てが空
    accessory = df[
        (df['IMEI'].isna() | (df['IMEI'] == '')) &
        (df['ICCID'].isna() | (df['ICCID'] == '')) &
        (df['その他シリアル'].isna() | (df['その他シリアル'] == ''))
    ].copy()
    
    # 数量bkを使用
    accessory['数量'] = accessory['数量bk']
    
    # グループ化
    result = accessory.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_ido_data(df: pd.DataFrame, masters: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    移動データの完全処理（Power Query準拠）
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 型変換
    df['入庫予定数'] = pd.to_numeric(df['入庫予定数'], errors='coerce').fillna(0).astype(int)
    df['未入庫数'] = pd.to_numeric(df['未入庫数'], errors='coerce').fillna(0).astype(int)
    
    # 不要な列を削除（空の列）
    df = df.loc[:, ~df.columns.str.startswith('_')]
    df = df.loc[:, df.columns != '']
    
    # 移動元取次店コードでマスタ857001とマージ（Inner Join）
    if '857001' in masters:
        moto_master = masters['857001'].copy()
        moto_master = moto_master.rename(columns={
            '取次店コード': '移動元取次店コード',
            '店舗倉庫区分': '店舗倉庫区分',
            '事業CD': '移動元事業CD',
            '保管場所CD': '移動元保管場所CD'
        })
        df = df.merge(moto_master, on='移動元取次店コード', how='inner')
    
    # 移動先取次店コードでマスタ857001とマージ（Inner Join）
    if '857001' in masters:
        saki_master = masters['857001'][['取次店コード', '事業CD', '保管場所CD']].copy()
        saki_master = saki_master.rename(columns={
            '取次店コード': '移動先取次店コード',
            '事業CD': '移動先事業CD',
            '保管場所CD': '移動先保管場所CD'
        })
        df = df.merge(saki_master, on='移動先取次店コード', how='inner')
    
    # マスタ857002とマージ
    if '857002' in masters:
        # 型を統一（文字列型に変換）
        df['移動元事業CD'] = df['移動元事業CD'].astype(str)
        master_857002 = masters['857002'].copy()
        master_857002['事業CD'] = master_857002['事業CD'].astype(str)
        
        df = df.merge(
            master_857002,
            left_on=['移動元事業CD', '商品コード'],
            right_on=['事業CD', '商品コード'],
            how='left',
            suffixes=('', '_master')
        )
        df['TMS商品CD'] = df['TMS商品CD'].fillna(df['商品コード'])
        if '事業CD_master' in df.columns:
            df = df.drop(columns=['事業CD_master'])
    else:
        df['TMS商品CD'] = df['商品コード']
    
    return df

def process_ido_shukko(df: pd.DataFrame) -> pd.DataFrame:
    """移動データ（出庫）"""
    # カテゴリ中がＵＳＩＭカードでない
    shukko = df[~df['カテゴリ中'].str.contains('ＵＳＩＭカード', na=False)].copy()
    
    # 数量計算: 入庫予定数 * -1
    shukko['数量'] = shukko['入庫予定数'] * -1
    shukko['取次店コード'] = shukko['移動元取次店コード']
    shukko['取次店名'] = shukko['移動元取次店名']
    
    # グループ化
    result = shukko.groupby(
        ['取次店コード', '取次店名', '移動元事業CD', '移動元保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={
        '移動元事業CD': '事業CD',
        '移動元保管場所CD': '保管場所CD',
        '数量': '変動数'
    })
    
    # 保管場所CDが空でない
    result = result[result['保管場所CD'] != '']
    
    return result

def process_ido_nyuko(df: pd.DataFrame) -> pd.DataFrame:
    """移動データ（入庫）"""
    # カテゴリ中がＵＳＩＭカードでない
    nyuko = df[~df['カテゴリ中'].str.contains('ＵＳＩＭカード', na=False)].copy()
    
    # 数量計算: 入庫予定数 * 1
    nyuko['数量'] = nyuko['入庫予定数'] * 1
    nyuko['取次店コード'] = nyuko['移動先取次店コード']
    nyuko['取次店名'] = nyuko['移動先取次店名']
    
    # グループ化
    result = nyuko.groupby(
        ['取次店コード', '取次店名', '移動先事業CD', '移動先保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={
        '移動先事業CD': '事業CD',
        '移動先保管場所CD': '保管場所CD',
        '数量': '変動数'
    })
    
    # 保管場所CDが空でない
    result = result[result['保管場所CD'] != '']
    
    return result

def process_uri_data(df: pd.DataFrame, masters: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    売上データの完全処理（Power Query準拠）
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # マスタ857001とマージ（Left Outer Join）
    if '857001' in masters:
        df = df.merge(masters['857001'], on='取次店コード', how='left')
    
    # 商品構成マスタとのマージは省略（マスタがない場合）
    # 商品コードをそのまま使用
    df['商品コードbk'] = df['商品コード']
    
    # マスタ857002とマージ
    if '857002' in masters:
        # 型を統一（文字列型に変換）
        df['事業CD'] = df['事業CD'].astype(str)
        master_857002 = masters['857002'].copy()
        master_857002['事業CD'] = master_857002['事業CD'].astype(str)
        
        df = df.merge(master_857002, on=['事業CD', '商品コード'], how='left', suffixes=('', '_master'))
        df['TMS商品CD'] = df['TMS商品CD'].fillna(df['商品コード'])
    else:
        df['TMS商品CD'] = df['商品コード']
    
    # 数量計算: 収納種別が「販売」なら-1、それ以外は1
    df['数量'] = df['収納種別'].apply(lambda x: -1 if x == '販売' else 1)
    
    return df

def process_uri_individual(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（個体情報）"""
    # 必要な列が無い場合は空文字列列を作る（入力データによっては列名が存在しないことがある）
    needed = ['メーカー', '業務種別', '店舗名', '取次店コード', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD']
    for col in needed:
        if col not in df.columns:
            df[col] = ''

    # メーカーがある かつ Apple Inc.-SBSとｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝでない
    individual = df[
        (df['メーカー'].notna()) &
        (df['メーカー'] != '') &
        (df['メーカー'] != 'Apple Inc.-SBS') &
        (df['メーカー'] != 'ｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝ')
    ].copy()
    
    # 業務種別にＵＳＩＭを含まない
    individual = individual[~individual['業務種別'].str.contains('ＵＳＩＭ', na=False)]
    
    individual['取次店名'] = individual['店舗名']

    # 数量列が無ければ作成（通常は process_uri_data で作成される）
    if '数量' not in individual.columns:
        if '収納種別' in individual.columns:
            individual['数量'] = individual['収納種別'].apply(lambda x: -1 if x == '販売' else 1)
        else:
            individual['数量'] = 0

    # グループ化
    result = individual.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_uri_sb_accessory(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（SBアクセサリ）"""
    # 必要な列が無い場合は空文字列列を作る
    if '取次店コード' not in df.columns:
        df['取次店コード'] = ''
    if 'メーカー' not in df.columns:
        df['メーカー'] = ''

    # 取次店コードがTGで始まる
    sb_acc = df[df['取次店コード'].str.startswith('TG', na=False)].copy()
    
    # メーカーがApple Inc.-SBSまたはｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝ
    sb_acc = sb_acc[
        (sb_acc['メーカー'] == 'Apple Inc.-SBS') |
        (sb_acc['メーカー'] == 'ｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝ')
    ]
    
    sb_acc['取次店名'] = sb_acc['店舗名']

    # 数量列が無ければ作成
    if '数量' not in sb_acc.columns:
        if '収納種別' in sb_acc.columns:
            sb_acc['数量'] = sb_acc['収納種別'].apply(lambda x: -1 if x == '販売' else 1)
        else:
            sb_acc['数量'] = 0

    # グループ化
    result = sb_acc.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_uri_service(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（サービス）"""
    # 必要な列が無い場合は空文字列列を作る
    if '商品分類' not in df.columns:
        df['商品分類'] = ''
    if '店舗名' not in df.columns:
        df['店舗名'] = ''

    # 商品分類が「サービス」
    service = df[df['商品分類'] == 'サービス'].copy()
    
    service['取次店名'] = service['店舗名']

    # 数量列が無ければ作成
    if '数量' not in service.columns:
        if '収納種別' in service.columns:
            service['数量'] = service['収納種別'].apply(lambda x: -1 if x == '販売' else 1)
        else:
            service['数量'] = 0

    # グループ化
    result = service.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_tana_data(df: pd.DataFrame, masters: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    棚卸データの完全処理（Power Query準拠）
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # マスタ857001とマージ（Inner Join）
    if '857001' in masters:
        df = df.merge(masters['857001'], on='取次店コード', how='inner')
    
    # 型変換
    df['受払前在庫数'] = pd.to_numeric(df['受払前在庫数'], errors='coerce').fillna(0).astype(int)
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').fillna(0).astype(int)
    df['受払後在庫数'] = pd.to_numeric(df['受払後在庫数'], errors='coerce').fillna(0).astype(int)
    
    # 不要な列を削除
    df = df.loc[:, ~df.columns.str.startswith('_')]
    df = df.loc[:, df.columns != '']
    
    # 店舗倉庫区分でフィルタ
    df = df[df['店舗倉庫区分'] == '1']
    
    # マスタ857002とマージ
    if '857002' in masters:
        # 型を統一（文字列型に変換）
        df['事業CD'] = df['事業CD'].astype(str)
        master_857002 = masters['857002'].copy()
        master_857002['事業CD'] = master_857002['事業CD'].astype(str)
        
        df = df.merge(master_857002, on=['事業CD', '商品コード'], how='left', suffixes=('', '_master'))
        df['TMS商品CD'] = df['TMS商品CD'].fillna(df['商品コード'])
    else:
        df['TMS商品CD'] = df['商品コード']
    
    # 除外商品コード
    exclude_codes = ['ZUA292', 'ZUA34Q', 'ZUA34R', 'ZUA34S', 'ZUA34T', 'ZUA34U', 'ZUA34V', 'ZUA34W']
    for code in exclude_codes:
        df = df[~df['商品コード'].str.contains(code, na=False)]
    
    return df

def process_tana_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """棚卸データ（グループ化）"""
    # カテゴリ中がＵＳＩＭカードでない
    grouped = df[~df['カテゴリ中'].str.contains('ＵＳＩＭカード', na=False)].copy()
    
    # グループ化
    result = grouped.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def combine_all_data(shiire_ind, shiire_acc, ido_shukko, ido_nyuko, 
                     uri_ind, uri_sb, uri_service, tana_grouped) -> pd.DataFrame:
    """
    全データを結合してTMS商品CDで集計（GINIEPOS変動数）
    """
    all_dfs = []
    
    for df in [shiire_ind, shiire_acc, ido_shukko, ido_nyuko, uri_ind, uri_sb, uri_service, tana_grouped]:
        if df is not None and not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    # 全て結合
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # TMS商品CDでグループ化して合計
    result = combined.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', 'TMS商品CD'],
        dropna=False
    )['変動数'].sum().reset_index()
    
    # ソート
    result = result.sort_values('取次店コード').reset_index(drop=True)
    
    return result

def compare_with_current_inventory(giniepos_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """
    GINIEPOS変動数と現在庫照会を比較（判定結果）
    """
    if giniepos_df is None or giniepos_df.empty or current_df is None or current_df.empty:
        return pd.DataFrame()
    
    # 現在庫照会から必要な列のみ取得
    current_summary = current_df[['保管場所CD', '事業CD', '商品CD', '実在庫数量']].copy()
    
    # 型を統一（文字列型に変換）
    giniepos_df['事業CD'] = giniepos_df['事業CD'].astype(str)
    current_summary['事業CD'] = current_summary['事業CD'].astype(str)
    current_summary['保管場所CD'] = current_summary['保管場所CD'].astype(str)
    current_summary['商品CD'] = current_summary['商品CD'].astype(str)
    giniepos_df['保管場所CD'] = giniepos_df['保管場所CD'].astype(str)
    giniepos_df['TMS商品CD'] = giniepos_df['TMS商品CD'].astype(str)
    
    # マージ（Left Outer Join）
    result = giniepos_df.merge(
        current_summary,
        left_on=['保管場所CD', '事業CD', 'TMS商品CD'],
        right_on=['保管場所CD', '事業CD', '商品CD'],
        how='left'
    )
    
    # CL実在庫数と呼ぶ
    result = result.rename(columns={'実在庫数量': 'CL実在庫数'})
    
    # nullは0に置換
    result['CL実在庫数'] = result['CL実在庫数'].fillna(0)
    
    # 判定 = 変動数 + CL実在庫数
    result['判定'] = result['変動数'] + result['CL実在庫数']
    
    # パスマネ等を除外（判定前）
    result = result[~result['TMS商品CD'].str.contains('BB-RQ8POU1740', na=False)]
    result = result[~result['TMS商品CD'].str.contains('ZUA292', na=False)]
    
    # 在庫不足の判定（判定 < 0）
    result = result[result['判定'] < 0]
    
    # POS-、Z00014を除外（在庫不足抽出後）
    result = result[~result['TMS商品CD'].str.contains('Z00014', na=False)]
    result = result[~result['TMS商品CD'].str.contains('POS-', na=False)]
    
    return result

# ファイルアップロードセクション
st.header("1️⃣ ファイルアップロード")

# 実行モードの選択
mode = st.selectbox("実行モードを選択してください", ["Full (8 files)", "Sales only (single sales file)"])

col1, col2 = st.columns([4, 1])

with col1:
    if mode == "Full (8 files)":
        st.info("📁 必要な8つのファイルをまとめてドラッグ&ドロップしてください")
    else:
        st.info("📁 売上ファイルのみでチェックします。売上ファイルと現在庫（Excel）、必要なマスタをアップロードしてください")

with col2:
    if st.button("🔄 クリア", key='clear_btn', help="アップロードしたファイルをクリア"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        safe_rerun()

if mode == "Full (8 files)":
    uploaded_files = st.file_uploader(
        "ファイルを選択またはドラッグ&ドロップ",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="在庫変動データ4ファイル + 現在庫照会 + マスタ3ファイル = 計8ファイル"
    )
else:
    # Sales-only mode: 単一の売上ファイル、現在庫、マスタ（任意）を受け取る
    sales_file = st.file_uploader(
        "売上ファイルを選択（CSV）",
        type=['csv'],
        accept_multiple_files=False,
        key='sales_only'
    )
    current_file = st.file_uploader(
        "現在庫照会ファイルを選択（Excel）",
        type=['xlsx', 'xls'],
        accept_multiple_files=False,
        key='current_only'
    )
    st.markdown("---")
    st.markdown("**必要に応じてマスタファイル（857001, 857002, 857003）をアップロード**")
    master_857001_file = st.file_uploader("マスタ857001 (取次店)", type=['csv'], key='m857001')
    master_857002_file = st.file_uploader("マスタ857002 (商品)", type=['csv'], key='m857002')
    master_857003_file = st.file_uploader("マスタ857003 (仕入先)", type=['csv'], key='m857003')

# ファイル名からファイルを振り分け
shiire_file = None
ido_file = None
uri_file = None
tana_file = None
current_file = None
master_857001_file = None
master_857002_file = None
master_857003_file = None

if uploaded_files:
    st.subheader("📋 アップロード済みファイル")
    
    for file in uploaded_files:
        filename = file.name
        
        if 'SHI' in filename.upper():
            shiire_file = file
            st.success(f"✅ 仕入データ: {filename}")
        elif 'IDO' in filename.upper():
            ido_file = file
            st.success(f"✅ 移動データ: {filename}")
        elif 'URI' in filename.upper():
            uri_file = file
            st.success(f"✅ 売上データ: {filename}")
        elif 'TNA' in filename.upper():
            tana_file = file
            st.success(f"✅ 棚卸データ: {filename}")
        elif '現在庫' in filename or 'ZAIKO' in filename.upper():
            current_file = file
            st.success(f"✅ 現在庫照会: {filename}")
        elif '857001' in filename:
            master_857001_file = file
            st.success(f"✅ マスタ857001（取次店）: {filename}")
        elif '857002' in filename:
            master_857002_file = file
            st.success(f"✅ マスタ857002（商品コード）: {filename}")
        elif '857003' in filename:
            master_857003_file = file
            st.success(f"✅ マスタ857003（仕入先）: {filename}")
        else:
            st.warning(f"⚠️ 不明なファイル: {filename}")
    
    # ファイル数チェック
    total_files = sum([
        shiire_file is not None,
        ido_file is not None,
        uri_file is not None,
        tana_file is not None,
        current_file is not None,
        master_857001_file is not None,
        master_857002_file is not None,
        master_857003_file is not None
    ])
    
    if total_files < 8:
        st.warning(f"⚠️ {total_files}/8ファイルが認識されました。全8ファイル必要です。")
    else:
        st.success("✅ 全8ファイルが揃いました！")

st.markdown("---")

# 処理実行ボタン
if st.button("🚀 在庫チェック実行", type="primary", use_container_width=True):
    if not all([shiire_file, ido_file, uri_file, tana_file, current_file]):
        st.error("⚠️ 在庫変動データと現在庫照会ファイルは必須です")
    else:
        with st.spinner("処理中..."):
            try:
                # マスタファイルの読み込み
                st.info("📚 マスタファイル読み込み中...")
                masters = load_master_files(master_857001_file, master_857002_file, master_857003_file)
                
                # CSVファイルの読み込み（LF改行、Shift-JIS）
                st.info("📂 在庫変動データ読み込み中...")
                shiire_df = load_csv_with_encoding(shiire_file, use_lf=True, encoding='cp932')
                ido_df = load_csv_with_encoding(ido_file, use_lf=True, encoding='cp932')
                uri_df = load_csv_with_encoding(uri_file, use_lf=True, encoding='cp932')
                if uri_df is None:
                    uri_df = pd.DataFrame()
                tana_df = load_csv_with_encoding(tana_file, use_lf=True, encoding='cp932')
                
                # 現在庫ファイルの読み込み
                st.info("📊 現在庫照会読み込み中...")
                current_df = pd.read_excel(current_file)
                st.success(f"✅ 現在庫照会: {len(current_df)}行")
                
                # 仕入データ処理
                st.info("🔄 仕入データ処理中...")
                shiire_processed = process_shiire_data(shiire_df, masters)
                shiire_individual = process_shiire_individual(shiire_processed)
                shiire_accessory = process_shiire_accessory(shiire_processed)
                st.success(f"✅ 仕入（個体）: {len(shiire_individual)}行、仕入（アクセサリ）: {len(shiire_accessory)}行")
                
                # 移動データ処理
                st.info("🔄 移動データ処理中...")
                ido_processed = process_ido_data(ido_df, masters)
                ido_shukko = process_ido_shukko(ido_processed)
                ido_nyuko = process_ido_nyuko(ido_processed)
                st.success(f"✅ 移動（出庫）: {len(ido_shukko)}行、移動（入庫）: {len(ido_nyuko)}行")
                
                # 売上データ処理
                st.info("🔄 売上データ処理中...")

                # 1) main.py 相当の CSV レベルのチェックを先に実行（生データを検査）
                try:
                    raw_bytes = uri_file.getvalue()
                    text = raw_bytes.decode('cp932')
                except Exception:
                    text = None

                if text:
                    try:
                        err_flag, err_details, total_records, total_physical_lines, date_summary = check_and_analyze(text)
                        if err_flag:
                            st.error("❌ 売上ファイルに NG 条件が見つかりました。処理を中止します。")
                            st.write(f"NG件数: {len(err_details)} 件")
                            err_csv = build_error_csv_bytes(err_details)
                            st.download_button("NG行一覧をダウンロード (UTF-8)", data=err_csv, file_name=f"{uri_file.name}_ng.csv")
                            # 日付指摘もダウンロード可能
                            ds_bytes = build_date_issue_csv_bytes(date_summary.issues)
                            st.download_button("日付チェック指摘をダウンロード (UTF-8)", data=ds_bytes, file_name=f"{uri_file.name}_date_issues.csv")
                            st.stop()
                        else:
                            # 日付チェックの警告などを表示（あれば）
                            if date_summary.issues:
                                st.warning(f"日付チェックで指摘があります（{len(date_summary.issues)} 件）。ダウンロードして確認してください。")
                                ds_bytes = build_date_issue_csv_bytes(date_summary.issues)
                                st.download_button("日付チェック指摘をダウンロード (UTF-8)", data=ds_bytes, file_name=f"{uri_file.name}_date_issues.csv")
                    except Exception as e:
                        st.warning(f"売上ファイルの事前チェックで例外: {e}")

                # 2) 既存の前処理: AG列削除とドコモショップ抽出
                try:
                    before_rows = len(uri_df)
                    uri_df = drop_ag_column(uri_df)
                    kept_df, omitted_df = split_docomo_shop_rows(uri_df)
                    kept_rows = len(kept_df)
                    omitted_rows = len(omitted_df)
                    st.info(f"🔎 売上前処理: {before_rows}行 -> ドコモショップ抽出 {kept_rows}行 (除外 {omitted_rows}行)")
                    uri_df = kept_df
                except Exception as e:
                    st.warning(f"売上前処理で注意: {e}")

                uri_processed = process_uri_data(uri_df, masters)
                uri_individual = process_uri_individual(uri_processed)
                uri_sb_accessory = process_uri_sb_accessory(uri_processed)
                uri_service = process_uri_service(uri_processed)
                st.success(f"✅ 売上（個体）: {len(uri_individual)}行、売上（SBアクセサリ）: {len(uri_sb_accessory)}行、売上（サービス）: {len(uri_service)}行")
                
                # 棚卸データ処理
                st.info("🔄 棚卸データ処理中...")
                tana_processed = process_tana_data(tana_df, masters)
                tana_grouped = process_tana_grouped(tana_processed)
                st.success(f"✅ 棚卸: {len(tana_grouped)}行")
                
                # 全データ結合
                st.info("🔗 データ結合・集計中...")
                giniepos_hendo = combine_all_data(
                    shiire_individual, shiire_accessory,
                    ido_shukko, ido_nyuko,
                    uri_individual, uri_sb_accessory, uri_service,
                    tana_grouped
                )
                st.success(f"✅ GINIEPOS変動数: {len(giniepos_hendo)}行")
                
                # 現在庫との比較
                st.info("🔍 在庫過不足チェック中...")
                result_df = compare_with_current_inventory(giniepos_hendo, current_df)
                
                # 結果をセッションステートに保存
                st.session_state.processed_data = result_df
                
                st.success("✅ 処理完了！")
                
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                st.exception(e)

# 結果表示セクション
if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
    st.markdown("---")
    st.header("2️⃣ 在庫不足結果")
    
    result_df = st.session_state.processed_data
    
    # サマリー情報
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("在庫不足件数", f"{len(result_df)}件")
    with col2:
        total_hendo = result_df['変動数'].sum()
        st.metric("変動数合計", f"{total_hendo:,}")
    with col3:
        total_cl = result_df['CL実在庫数'].sum()
        st.metric("CL実在庫合計", f"{int(total_cl):,}")
    
    # データテーブル表示
    st.subheader("📋 詳細リスト")
    
    display_cols = ['取次店コード', '取次店名', 'TMS商品CD', '変動数', 'CL実在庫数', '判定']
    available_cols = [col for col in display_cols if col in result_df.columns]
    
    st.dataframe(
        result_df[available_cols].sort_values('判定'),
        use_container_width=True,
        height=400
    )
    
    # CSVダウンロードボタン
    csv = result_df[available_cols].to_csv(index=False, encoding='cp932')
    st.download_button(
        label="📥 結果をCSVでダウンロード",
        data=csv,
        file_name="在庫不足結果.csv",
        mime="text/csv",
        use_container_width=True
    )

elif st.session_state.processed_data is not None:
    st.success("✅ 在庫不足はありません！")

# フッター
st.markdown("---")
st.markdown("**在庫不足チェックシステム** | Python + Streamlit版 | Power Query完全準拠")
