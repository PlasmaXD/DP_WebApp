# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pydp.algorithms.laplacian as dp  # PyDPのインポート

def add_differential_privacy(data, epsilon=1.0):
    noisy_data = data.copy()
    
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):  # 数値データにのみ適用
            # Laplace機構を使用してノイズを加える
            scale = 1 / epsilon  # スケールパラメータを設定

            noisy_column = []
            for value in data[column]:
                # ノイズを加えた値を取得
                noisy_value = value + np.random.laplace(0, scale)
                noisy_column.append(noisy_value)
            
            # ノイズを加えたデータを新しい列に置き換え
            noisy_data[column] = noisy_column
    
    return noisy_data

# CSVファイルを読み込む
input_csv = 'input.csv'
output_csv = 'output_noisy.csv'

# ファイルのエンコーディングを指定して読み込み
data = pd.read_csv(input_csv, encoding='utf-8-sig')

# 数値列を浮動小数点数に変換
for column in data.columns:
    if data[column].dtype == 'object':
        try:
            data[column] = data[column].str.replace(',', '').astype(float)
        except ValueError:
            continue

# 差分プライバシーを適用
epsilon = 1.0
noisy_data = add_differential_privacy(data, epsilon)

# ノイズを加えたデータを新しいCSVファイルに書き出す
noisy_data.to_csv(output_csv, index=False)

print(f"差分プライバシーを適用したデータを {output_csv} に保存しました。")
