FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ビルド時に Silero VAD モデルをダウンロードしてキャッシュ（初回起動を高速化）
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=True)"

COPY . .

# Railway が PORT 環境変数を設定するので、それを使う
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
