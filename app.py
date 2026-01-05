import streamlit as st
import pandas as pd
import io
from typing import Dict, List

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
    
    # グループ化
    result = individual.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_uri_sb_accessory(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（SBアクセサリ）"""
    # 取次店コードがTGで始まる
    sb_acc = df[df['取次店コード'].str.startswith('TG', na=False)].copy()
    
    # メーカーがApple Inc.-SBSまたはｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝ
    sb_acc = sb_acc[
        (sb_acc['メーカー'] == 'Apple Inc.-SBS') |
        (sb_acc['メーカー'] == 'ｿﾌﾄﾊﾞﾝｸｾﾚｸｼｮﾝ')
    ]
    
    sb_acc['取次店名'] = sb_acc['店舗名']
    
    # グループ化
    result = sb_acc.groupby(
        ['取次店コード', '取次店名', '事業CD', '保管場所CD', '商品コード', 'TMS商品CD'],
        dropna=False
    )['数量'].sum().reset_index()
    result = result.rename(columns={'数量': '変動数'})
    
    return result

def process_uri_service(df: pd.DataFrame) -> pd.DataFrame:
    """売上データ（サービス）"""
    # 商品分類が「サービス」
    service = df[df['商品分類'] == 'サービス'].copy()
    
    service['取次店名'] = service['店舗名']
    
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

col1, col2 = st.columns([4, 1])

with col1:
    st.info("📁 必要な8つのファイルをまとめてドラッグ&ドロップしてください")

with col2:
    if st.button("🔄 クリア", use_container_width=True, help="アップロードしたファイルをクリア"):
        st.session_state.clear()
        st.rerun()

uploaded_files = st.file_uploader(
    "ファイルを選択またはドラッグ&ドロップ",
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True,
    help="在庫変動データ4ファイル + 現在庫照会 + マスタ3ファイル = 計8ファイル"
)

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
