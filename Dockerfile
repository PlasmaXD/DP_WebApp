# ベースイメージを指定
FROM python:3.11

# 作業ディレクトリを作成
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git build-essential cmake python3-dev

# SEAL-Pythonの依存関係をインストール
RUN pip install numpy pybind11

# SEAL-Pythonをクローンしてビルド
RUN git clone https://github.com/Huelse/SEAL-Python.git /app/SEAL-Python
WORKDIR /app/SEAL-Python
RUN git submodule update --init --recursive
RUN cd SEAL && cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF && cmake --build build
RUN python3 setup.py build_ext -i

# 必要なPythonパッケージをインストール
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# アプリケーションのコードをコピー
COPY . /app

# Streamlitアプリを起動
CMD ["streamlit", "run", "compare_dp.py"]
