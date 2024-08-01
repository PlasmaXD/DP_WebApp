import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydp.algorithms.laplacian import BoundedMean
def add_differential_privacy(data, epsilon=1.0):
    noisy_data = data.copy()
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):  # 数値データにのみ適用
            # PyDPを使用してノイズを加える
            # lower_boundとupper_boundを整数にキャスト
            lower_bound = int(data[column].min())
            upper_bound = int(data[column].max())

            # BoundedMeanのインスタンスを整数の範囲で作成
            bounded_mean = BoundedMean(epsilon=epsilon, delta=0.0001, lower_bound=lower_bound, upper_bound=upper_bound)
            noisy_column = [bounded_mean.quick_result([int(val)]) for val in data[column]]
            noisy_data[column] = noisy_column
            
    return noisy_data


def main():
    st.title('差分プライバシー データ変換アプリ')

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("元のデータ:")
        st.write(data.head())

        if 'noisy_data' not in st.session_state or st.button("ノイズを加える"):
            epsilon = st.slider("プライバシー保護レベル ε", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
            st.session_state.noisy_data = add_differential_privacy(data, epsilon)
        
        st.write("ノイズが加えられたデータ:")
        st.write(st.session_state.noisy_data.head())

if __name__ == "__main__":
    main()

