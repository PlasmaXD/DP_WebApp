import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'

def add_laplace_noise(data, epsilon=1.0, sensitivity=1.0):
    noisy_data = data.copy()

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # Laplaceノイズを生成して加える
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, data[column].shape)
            noisy_data[column] += noise

    return noisy_data

def calculate_accuracies(original_data, noisy_data, match_type='position'):
    """トップ10の一致率を計算"""
    max_accuracy = 0
    min_accuracy = 0
    
    for column in original_data.columns:
        if pd.api.types.is_numeric_dtype(original_data[column]):
            # 大きい順のトップ10の一致率
            original_top10_max = original_data[column].nlargest(10).index
            noisy_top10_max = noisy_data[column].nlargest(10).index
            if match_type == 'position':
                max_match_count = sum(1 for o, n in zip(original_top10_max, noisy_top10_max) if o == n)
            else:  # 'content'
                max_match_count = len(set(original_top10_max).intersection(set(noisy_top10_max)))
            max_accuracy += max_match_count / 10
            
            # 小さい順のトップ10の一致率
            original_top10_min = original_data[column].nsmallest(10).index
            noisy_top10_min = noisy_data[column].nsmallest(10).index
            if match_type == 'position':
                min_match_count = sum(1 for o, n in zip(original_top10_min, noisy_top10_min) if o == n)
            else:  # 'content'
                min_match_count = len(set(original_top10_min).intersection(set(noisy_top10_min)))
            min_accuracy += min_match_count / 10

    num_columns = len(original_data.select_dtypes(include=np.number).columns)
    return max_accuracy / num_columns, min_accuracy / num_columns

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

def plot_accuracies_vs_epsilon(epsilon_values, max_accuracies, min_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, max_accuracies, marker='o', linestyle='-', color='blue', label='トップ10大きい順一致率')
    plt.plot(epsilon_values, min_accuracies, marker='o', linestyle='-', color='green', label='トップ10小さい順一致率')
    plt.title('ノイズ量 (ε) とトップ10の一致率')
    plt.xlabel('ε')
    plt.ylabel('一致率 (%)')
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
        
        # 数値データのみを抽出
        numeric_data = data.select_dtypes(include=np.number)
        
        # 一致の判定方法を選択
        match_type = st.radio("一致の判定方法", ('position', 'content'))
        
        if 'noisy_data' not in st.session_state or st.button("ノイズを加える"):
            epsilon = st.slider("プライバシー保護レベル ε (小さいほどプライバシー保護が強化されますが、データの変動が大きくなります)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
            sensitivity = st.number_input("感度を入力してください", min_value=1.0, max_value=1000.0, value=1.0, step=0.1)
            st.session_state.noisy_data = add_laplace_noise(numeric_data, epsilon, sensitivity)
        
        st.write("ノイズが加えられたデータ:")
        st.write(st.session_state.noisy_data.head())
        
        # グラフの表示: εに対する一致率とズレ
        epsilon_values = np.arange(0.01, 1.1, 0.1)
        max_accuracies, min_accuracies = [], []
        mean_differences, std_differences = [], []
        
        for epsilon in epsilon_values:
            noisy_data = add_laplace_noise(numeric_data, epsilon, sensitivity)
            max_acc, min_acc = calculate_accuracies(numeric_data, noisy_data, match_type)
            mean_diff, std_diff = calculate_mean_std_differences(numeric_data, noisy_data)
            
            max_accuracies.append(max_acc)
            min_accuracies.append(min_acc)
            mean_differences.append(mean_diff)
            std_differences.append(std_diff)
        
        if not numeric_data.empty:
            plot_accuracies_vs_epsilon(epsilon_values, max_accuracies, min_accuracies)
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
