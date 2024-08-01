import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydp.algorithms.laplacian import BoundedMean


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


def show_statistics(data):
    st.write("基本統計情報:")
    st.write(data.describe())

plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォントの設定

def plot_histogram(data, column, bin_count):
    plt.figure(figsize=(10, 4))
    data_range = (data[column].min(), data[column].max())  # データの範囲を取得
    plt.hist(data[column], bins=bin_count, range=data_range, alpha=0.75, color='blue')
    plt.title(f'ヒストグラム（{column}）')
    plt.xlabel('値')
    plt.ylabel('頻度')
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
        
        # 統計情報の表示
        show_statistics(st.session_state.noisy_data)
        
        # ヒストグラムの選択と表示
        column = st.selectbox("ヒストグラムを表示する列を選択してください:", st.session_state.noisy_data.columns)
        bin_count = st.slider("ビンの数を選択", min_value=5, max_value=100, value=st.session_state.bin_count)
        st.session_state.bin_count = bin_count  # 更新されたビンの数を保存
        plot_histogram(st.session_state.noisy_data, column, st.session_state.bin_count)
        
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
