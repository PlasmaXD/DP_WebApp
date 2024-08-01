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
            # データを整数にスケーリング
            scaled_data = (data[column] * scaling_factor).astype(int)
            lower_bound = scaled_data.min()
            upper_bound = scaled_data.max()

            # BoundedMeanのインスタンスを作成
            bounded_mean = BoundedMean(epsilon, 0, lower_bound, upper_bound)
            # 各値にノイズを加える
            noisy_values = [bounded_mean.quick_result([val]) for val in scaled_data]
            # データを元のスケールに戻す
            noisy_data[column] = [val / scaling_factor for val in noisy_values]
            
    return noisy_data

def calculate_accuracies(original_data, noisy_data):
    """最大値、最小値、中央値のインデックス一致率を計算"""
    max_accuracy = 0
    min_accuracy = 0
    median_accuracy = 0
    
    for column in original_data.columns:
        if pd.api.types.is_numeric_dtype(original_data[column]):
            # 最大値の一致率
            if original_data[column].idxmax() == noisy_data[column].idxmax():
                max_accuracy += 1
            # 最小値の一致率
            if original_data[column].idxmin() == noisy_data[column].idxmin():
                min_accuracy += 1
            # 中央値の一致率
            if original_data[column].median() == noisy_data[column].median():
                median_accuracy += 1

    num_columns = len(original_data.select_dtypes(include=np.number).columns)
    return max_accuracy / num_columns, min_accuracy / num_columns, median_accuracy / num_columns

def calculate_mean_std_differences(original_data, noisy_data):
    """平均値と標準偏差のズレを計算"""
    mean_differences = []
    std_differences = []
    
    for column in original_data.columns:
        if pd.api.types.is_numeric_dtype(original_data[column]):
            original_mean = original_data[column].mean()
            noisy_mean = noisy_data[column].mean()
            mean_differences.append(abs(original_mean - noisy_mean))
            
            original_std = original_data[column].std()
            noisy_std = noisy_data[column].std()
            std_differences.append(abs(original_std - noisy_std))
    
    return np.mean(mean_differences), np.mean(std_differences)

def plot_histograms(original_data, noisy_data, column, bin_count):
    plt.figure(figsize=(10, 4))
    data_range = (min(original_data[column].min(), noisy_data[column].min()), 
                  max(original_data[column].max(), noisy_data[column].max()))  # データの範囲を取得

    plt.hist(original_data[column], bins=bin_count, range=data_range, alpha=0.5, color='blue', label='元のデータ')
    plt.hist(noisy_data[column], bins=bin_count, range=data_range, alpha=0.5, color='red', label='ノイズ付加データ')
    plt.title(f'ヒストグラム（{column}）')
    plt.xlabel('値')
    plt.ylabel('頻度')
    plt.legend()
    st.pyplot(plt)

def plot_accuracies_vs_epsilon(epsilon_values, max_accuracies, min_accuracies, median_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, max_accuracies, marker='o', linestyle='-', color='blue', label='最大値一致率')
    plt.plot(epsilon_values, min_accuracies, marker='o', linestyle='-', color='green', label='最小値一致率')
    plt.plot(epsilon_values, median_accuracies, marker='o', linestyle='-', color='orange', label='中央値一致率')
    plt.title('ノイズ量 (ε) と一致率')
    plt.xlabel('ε')
    plt.ylabel('一致率')
    plt.ylim(0, 1.1)  # 一致率の範囲を0から100%に設定
    plt.xticks(epsilon_values)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_differences_vs_epsilon(epsilon_values, mean_differences, std_differences):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, mean_differences, marker='o', linestyle='-', color='red', label='平均値のズレ')
    plt.plot(epsilon_values, std_differences, marker='o', linestyle='-', color='purple', label='標準偏差のズレ')
    plt.title('ノイズ量 (ε) とズレ')
    plt.xlabel('ε')
    plt.ylabel('ズレ')
    plt.ylim(0, max(max(mean_differences), max(std_differences)) * 1.1)  # ズレの範囲を設定
    plt.xticks(epsilon_values)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def main():
    st.title('差分プライバシー データ変換アプリ')
    
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("元のデータ:")
        st.write(data.head())
        
        if 'noisy_data' not in st.session_state or st.button("ノイズを加える"):
            epsilon = st.slider("プライバシー保護レベル ε (小さいほどプライバシー保護が強化されますが、データの変動が大きくなります)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
            st.session_state.noisy_data = add_differential_privacy(data, epsilon)
            st.session_state.bin_count = 30
        
        st.write("ノイズが加えられたデータ:")
        st.write(st.session_state.noisy_data.head())
        
        # ヒストグラムの選択と表示
        column = st.selectbox("ヒストグラムを表示する列を選択してください:", data.columns)
        bin_count = st.slider("ビンの数を選択", min_value=5, max_value=100, value=st.session_state.bin_count)
        st.session_state.bin_count = bin_count  # 更新されたビンの数を保存
        plot_histograms(data, st.session_state.noisy_data, column, st.session_state.bin_count)
        
        # グラフの表示: εに対する一致率とズレ
        epsilon_values = np.arange(0.01, 1.1, 0.1)
        max_accuracies, min_accuracies, median_accuracies = [], [], []
        mean_differences, std_differences = [], []
        
        for epsilon in epsilon_values:
            noisy_data = add_differential_privacy(data, epsilon)
            max_acc, min_acc, median_acc = calculate_accuracies(data, noisy_data)
            mean_diff, std_diff = calculate_mean_std_differences(data, noisy_data)
            
            max_accuracies.append(max_acc)
            min_accuracies.append(min_acc)
            median_accuracies.append(median_acc)
            mean_differences.append(mean_diff)
            std_differences.append(std_diff)
        
        plot_accuracies_vs_epsilon(epsilon_values, max_accuracies, min_accuracies, median_accuracies)
        plot_differences_vs_epsilon(epsilon_values, mean_differences, std_differences)
        
        # CSVとしてダウンロード
        csv = st.session_state.noisy_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="加工後のデータをダウンロードする",
            data=csv,
            file_name='noisy_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
