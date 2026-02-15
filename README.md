# Omi × Groq Whisper STT ブリッジサーバー

Omiウェアラブルの音声を **Groq API の Whisper Large-v3** で文字起こしするための中継サーバーです。  
オンデバイスWhisperの「thank you for watching」問題やハルシネーション（幻覚）をフィルタリングする機能も内蔵しています。

## 仕組み

```
Omiデバイス → (BLE) → Omiアプリ → (WebSocket) → このサーバー(Railway)
                                                        ↓
                                                   Groq Whisper API
                                                        ↓
                                                   文字起こし結果
                                                        ↓
                                                   Omiアプリに返却
```

---

## 必要なもの

| 項目 | 費用 | 備考 |
|------|------|------|
| Groq APIキー | 無料 | レートリミットあり |
| Railway アカウント | Hobby $5/月 | 最初に$5のトライアルクレジットあり |
| GitHub アカウント | 無料 | Railwayへのデプロイに使用 |

---

## Step 1: Groq APIキーを取得する

1. [console.groq.com](https://console.groq.com) にアクセス
2. **Googleアカウント** などでサインアップ（クレジットカード不要）
3. ログイン後、左メニューの **「API Keys」** をクリック
4. **「Create API Key」** ボタンをクリック
5. 名前を入力（例: `omi-stt`）→ **Submit**
6. 表示されたキー（`gsk_` で始まる文字列）を **コピーしてメモ帳などに保存**
   - ⚠️ このキーは一度しか表示されません！

---

## Step 2: GitHubにリポジトリを作る

1. [github.com](https://github.com) にログイン
2. 右上の **「+」** → **「New repository」**
3. 以下を入力:
   - **Repository name**: `omi-groq-stt`
   - **Visibility**: `Private`（推奨）
   - 他はそのままでOK
4. **「Create repository」** をクリック

### ファイルをアップロード

GitHub のリポジトリページで:

1. **「uploading an existing file」** リンクをクリック
2. 以下の4つのファイルをドラッグ＆ドロップ:
   - `server.py`
   - `requirements.txt`
   - `Dockerfile`
   - `railway.toml`
3. 下部の **「Commit changes」** をクリック

---

## Step 3: Railwayにデプロイする

### 3-1. Railwayアカウント作成

1. [railway.app](https://railway.app) にアクセス
2. **「Login」** → **GitHubアカウント** で認証

### 3-2. 新しいプロジェクトを作成

1. ダッシュボード右上の **「+ New Project」** をクリック
2. **「Deploy from GitHub Repo」** を選択
3. **「Configure GitHub App」** が出たら、クリックしてRailwayにGitHubへのアクセスを許可
4. さきほど作った **`omi-groq-stt`** リポジトリを選択
5. **「Deploy Now」** をクリック

### 3-3. 環境変数を設定する

デプロイが始まったら（まだビルド中でもOK）:

1. プロジェクト画面で、デプロイされたサービス（四角いカード）をクリック
2. 上のタブから **「Variables」** を選択
3. **「+ New Variable」** をクリックして以下を追加:

| Variable | Value |
|----------|-------|
| `GROQ_API_KEY` | `gsk_...`（Step 1で取得したキー） |
| `WHISPER_MODEL` | `whisper-large-v3` |
| `LANGUAGE` | `ja` |
| `PROMPT` | `日本語の会話です` |
| `CHUNK_SECONDS` | `5` |

4. 保存すると自動で再デプロイされます

### 3-4. 公開URLを取得する

1. サービスカードをクリック → **「Settings」** タブ
2. **「Networking」** セクションまでスクロール
3. **「Generate Domain」** ボタンをクリック
4. 生成されたURL（例: `omi-groq-stt-production-xxxx.up.railway.app`）をコピー

### 3-5. 動作確認

ブラウザで以下にアクセス:

```
https://omi-groq-stt-production-xxxx.up.railway.app/health
```

以下のようなJSONが返ればOK:
```json
{"status": "ok", "model": "whisper-large-v3", "language": "ja"}
```

---

## Step 4: OmiアプリでカスタムSTTを設定する

1. Omiアプリを開く
2. **設定（Settings）** → **Developer Settings** に進む
3. **Custom STT** / **Speech-to-Text Provider** の項目を探す
4. **WebSocket URL** に以下を入力:

```
wss://omi-groq-stt-production-xxxx.up.railway.app/listen
```

> ⚠️ `https://` ではなく **`wss://`** で始めてください（WebSocket Secure）

5. 設定を保存

これで完了です！  
Omiで録音を開始すると、音声がRailwayのサーバー経由でGroq Whisperに送られ、  
文字起こし結果がOmiアプリに返ってきます。

---

## カスタマイズ

### 環境変数の説明

| 環境変数 | デフォルト | 説明 |
|----------|-----------|------|
| `GROQ_API_KEY` | （必須） | Groq APIキー |
| `WHISPER_MODEL` | `whisper-large-v3` | モデル名。`whisper-large-v3-turbo` で速度優先 |
| `LANGUAGE` | `ja` | 言語コード。`ja`=日本語、`en`=英語 |
| `PROMPT` | `日本語の会話です` | Whisperへのコンテキストヒント（224トークン以内） |
| `CHUNK_SECONDS` | `5` | 何秒ごとにGroqに送信するか |

### PROMPTのカスタマイズ例

用途に応じてPROMPTを変更すると精度が上がります:

- 日常会話: `日本語の日常会話です`
- 会議: `日本語のビジネスミーティングです。議題、アクションアイテムが含まれます`
- 技術系: `日本語の技術的な会話です。プログラミング用語が含まれます`

---

## トラブルシューティング

### 「Railwayのビルドが失敗する」
- **Variables** タブで `GROQ_API_KEY` が正しくセットされているか確認
- **Deployments** タブでビルドログを確認

### 「Omiアプリに文字が出ない」
- WebSocket URLが `wss://` で始まっているか確認（`https://` ではない）
- `/health` にアクセスして `{"status": "ok"}` が返るか確認
- Railwayの **Deployments** タブで最新ログを確認

### 「429エラーが出る」（Groqレートリミット）
- Groq無料枠のリクエスト制限に達しています
- `CHUNK_SECONDS` を `10` に増やして送信頻度を下げる
- 数分待ってリトライ

### 「文字起こしは出るが精度が悪い」
- `LANGUAGE` が `ja` になっているか確認
- `PROMPT` を具体的な内容に変更してみる
- `WHISPER_MODEL` を `whisper-large-v3`（v3-turboではない方）に設定

### 「opusコーデックで動かない」
- Omiのcodecがopusの場合、OGG Opusコンテナに自動変換して
  Groqに送信しますが、フレームサイズの不一致で失敗する場合があります
- Omiアプリ側でPCMコーデックに変更できる場合は `pcm16` 推奨

---

## 費用の目安

| 項目 | 費用 |
|------|------|
| Groq API（無料枠） | $0 |
| Railway（Hobby） | ~$5/月 |
| **合計** | **~$5/月** |

※ Railwayは使った分だけの従量課金（$5のクレジット内で収まる可能性あり）  
※ Groqの無料枠で1日8時間程度は十分カバーできる見込みですが、  
  レートリミットに当たる場合は `CHUNK_SECONDS` を増やして調整してください

---

## ハルシネーションフィルター

このサーバーには、Whisperのよくある幻覚パターンを自動除去するフィルターが内蔵されています:

- 「Thank you for watching」系
- 韓国語・ロシア語のYouTube定型文
- URL・サイト名の挿入
- 無音区間の誤認識（no_speech_prob > 0.7）
- 極端に短い/繰り返しテキスト

フィルターパターンは `server.py` の `HALLUCINATION_PATTERNS` リストで追加・変更できます。
