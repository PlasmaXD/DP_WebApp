import streamlit as st
import pandas as pd
import numpy as np

def add_differential_privacy(data, epsilon=1.0):
    noisy_data = data.copy()
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):  # 数値データにのみ適用
            scale = 1 / epsilon
            noisy_column = data[column] + np.random.laplace(0, scale, size=len(data[column]))
            noisy_data[column] = noisy_column
            
    return noisy_data

def main():
    st.title('差分プライバシー データ変換アプリ')
    
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("元のデータ:")
        st.write(data.head())
        
        epsilon = st.slider("プライバシー保護レベル ε (小さいほどプライバシー保護が強化されますが、データの変動が大きくなります)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
        
        if st.button("ノイズを加える"):
            noisy_data = add_differential_privacy(data, epsilon)
            st.write("ノイズが加えられたデータ:")
            st.write(noisy_data.head())
            
            # CSVとしてダウンロード
            csv = noisy_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="加工後のデータをダウンロードする",
                data=csv,
                file_name='noisy_data.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
