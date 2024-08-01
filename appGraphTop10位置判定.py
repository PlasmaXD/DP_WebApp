import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydp.algorithms.laplacian import BoundedMean

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'

def add_differential_privacy(data, epsilon=1.0, scaling_factor=1000):
    noisy_data = data.copy()
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # スケーリングして整数に変換
            scaled_data = (data[column] * scaling_factor).astype(int)
            lower_bound = int(scaled_data.min())
            upper_bound = int(scaled_data.max())

            # BoundedMeanのインスタンスを作成
            bounded_mean = BoundedMean(
                epsilon=epsilon,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                l0_sensitivity=1,
                linf_sensitivity=1
            )

            # 各値にノイズを加える
            noisy_values = [bounded_mean.quick_result([val]) for val in scaled_data]
            # 元のスケールに戻す
            noisy_data[column] = [val / scaling_factor for val in noisy_values]
            
    return noisy_data

def calculate_top10_accuracy(original_data, noisy_data):
    """トップ10の一致率を計算"""
    top10_max_accuracy = 0
    top10_min_accuracy = 0
    
    for column in original_data.columns:
        if pd.api.types.is_numeric_dtype(original_data[column]):
            # 大きい順のトップ10の一致率
            original_top10_max = original_data[column].nlargest(10).index
            noisy_top10_max = noisy_data[column].nlargest(10).index
            max_match_count = sum(1 for o, n in zip(original_top10_max, noisy_top10_max) if o == n)
            top10_max_accuracy += max_match_count / 10
            
            # 小さい順のトップ10の一致率
            original_top10_min = original_data[column].nsmallest(10).index
            noisy_top10_min = noisy_data[column].nsmallest(10).index
            min_match_count = sum(1 for o, n in zip(original_top10_min, noisy_top10_min) if o == n)
            top10_min_accuracy += min_match_count / 10

            # デバッグ用出力
            print(f"Column: {column}")
            print(f"Original Top10 Max: {original_top10_max.tolist()}")
            print(f"Noisy Top10 Max: {noisy_top10_max.tolist()}")
            print(f"Original Top10 Min: {original_top10_min.tolist()}")
            print(f"Noisy Top10 Min: {noisy_top10_min.tolist()}")
            print(f"Max Match Count: {max_match_count}, Min Match Count: {min_match_count}\n")

    num_columns = len(original_data.select_dtypes(include=np.number).columns)
    return (top10_max_accuracy / num_columns) * 100, (top10_min_accuracy / num_columns) * 100

def plot_top10_accuracies_vs_epsilon(epsilon_values, top10_max_accuracies, top10_min_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, top10_max_accuracies, marker='o', linestyle='-', color='blue', label='トップ10大きい順一致率')
    plt.plot(epsilon_values, top10_min_accuracies, marker='o', linestyle='-', color='green', label='トップ10小さい順一致率')
    plt.title('ノイズ量 (ε) とトップ10の一致率')
    plt.xlabel('ε')
    plt.ylabel('一致率 (%)')
    plt.ylim(0, 110)  # 一致率の範囲を0から100%に設定
    plt.xticks(epsilon_values)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def main():
    st.title('差分プライバシー 株価データ分析アプリ')

    # 株価データのサンプル（実際にはCSVから読み込む）
    data = pd.DataFrame({
        "日付": ["2021/01/04", "2021/01/05", "2021/01/06", "2021/01/07", "2021/01/08", "2021/01/12", "2021/01/13", "2021/01/14", "2021/01/15", "2021/01/18"],
        "終値": [27258.38, 27158.63, 27055.94, 27490.13, 28139.03, 28164.34, 28456.59, 28698.26, 28519.18, 28242.21],
        "始値": [27575.57, 27151.38, 27102.85, 27340.46, 27720.14, 28004.37, 28140.10, 28442.73, 28777.47, 28238.68],
        "高値": [27602.11, 27279.78, 27196.40, 27624.73, 28139.03, 28287.37, 28503.43, 28979.53, 28820.50, 28349.97],
        "安値": [27042.32, 27073.46, 27002.18, 27340.46, 27667.75, 27899.45, 28133.59, 28411.58, 28477.03, 28111.54],
    })

    # 数値データのみを抽出
    stock_data = data.drop(columns=["日付"])

    # スケーリングファクターを定義
    scaling_factor = 1000  # スケーリングファクターを固定値に設定

    if 'noisy_data' not in st.session_state or st.button("ノイズを加える"):
        epsilon = st.slider("プライバシー保護レベル ε", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        st.session_state.noisy_data = add_differential_privacy(stock_data, epsilon, scaling_factor)
        
        st.write("ノイズが加えられたデータ:")
        st.write(st.session_state.noisy_data)

    # グラフの表示: εに対するトップ10の一致率
    epsilon_values = np.arange(0.01, 1.1, 0.1)
    top10_max_accuracies, top10_min_accuracies = [], []

    for epsilon in epsilon_values:
        noisy_data = add_differential_privacy(stock_data, epsilon, scaling_factor)
        max_acc, min_acc = calculate_top10_accuracy(stock_data, noisy_data)
        top10_max_accuracies.append(max_acc)
        top10_min_accuracies.append(min_acc)

    plot_top10_accuracies_vs_epsilon(epsilon_values, top10_max_accuracies, top10_min_accuracies)

if __name__ == "__main__":
    main()
