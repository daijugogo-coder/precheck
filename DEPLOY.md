# Streamlit Cloudデプロイ手順

## 1. GitHubリポジトリの作成

### ローカルでGit初期化

```bash
cd "c:\Users\316926\Desktop\SB在庫不足チェック"
git init
git add .
git commit -m "Initial commit: 在庫不足チェックシステム"
```

### GitHubでリポジトリ作成

1. https://github.com にアクセス
2. 右上の「+」→「New repository」をクリック
3. リポジトリ名: `zaiko-fusoku-check`（またはお好きな名前）
4. Description: `在庫不足チェックシステム - Streamlit版`
5. **Private**を選択（社内ツールのため）
6. 「Create repository」をクリック

### GitHubにプッシュ

```bash
git remote add origin https://github.com/[あなたのユーザー名]/zaiko-fusoku-check.git
git branch -M main
git push -u origin main
```

## 2. Streamlit Cloudへのデプロイ

### アカウント作成

1. https://share.streamlit.io にアクセス
2. 「Sign up」→「Continue with GitHub」でGitHubアカウントと連携
3. Streamlitがリポジトリにアクセスする権限を許可

### アプリのデプロイ

1. Streamlit Cloudダッシュボードで「New app」をクリック
2. 以下を設定：
   - **Repository**: `[あなたのユーザー名]/zaiko-fusoku-check`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `zaiko-fusoku-check`（カスタマイズ可能）
3. 「Deploy!」をクリック

### デプロイ完了

数分後、以下のようなURLでアクセス可能になります：
```
https://[あなたのユーザー名]-zaiko-fusoku-check.streamlit.app
```

## 3. 重要な注意事項

### ⚠️ セキュリティ

- **データファイルをGitHubにアップしない**
  - `.gitignore`で除外済み（*.csv, *.xlsx等）
  - 機密情報を含むファイルは絶対にプッシュしないこと

- **Privateリポジトリを使用**
  - 社内ツールなので必ずPrivateに設定

### 📁 データの扱い

- Streamlit Cloudアプリではファイルをアップロードして使用
- ファイルはセッション中のみメモリに保持
- セッション終了後は自動削除される

## 4. アップデート方法

コードを修正した後：

```bash
git add .
git commit -m "機能追加: XXX"
git push
```

→ Streamlit Cloudが自動的に再デプロイ（数分）

## 5. トラブルシューティング

### デプロイエラーが出る場合

1. `requirements.txt`の内容確認
2. Streamlit Cloudのログを確認
3. ローカルで動作確認

### アクセス制限をかけたい場合

Streamlit Cloudの設定で：
- Email認証を有効化
- 特定のメールドメインのみ許可（例: @yourcompany.com）

## 6. 参考リンク

- Streamlit Cloud公式: https://docs.streamlit.io/streamlit-community-cloud
- GitHub公式: https://docs.github.com/
