import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from seal import *

# 通常の差分プライバシー
def add_differential_privacy(data, epsilon=1.0):
    noisy_data = data.copy()
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):  # 数値データにのみ適用
            scale = 1 / epsilon
            noisy_column = data[column] + np.random.laplace(0, scale, size=len(data[column]))
            noisy_data[column] = noisy_column
    return noisy_data

# 準同型暗号を用いた差分プライバシー
def encrypt_data(data, encoder, encryptor, scale):
    encrypted_data = []
    for value in data:
        if isinstance(value, (int, float)):  # 数値データのみを処理
            plain = encoder.encode(value, scale)
            encrypted_value = encryptor.encrypt(plain)
            encrypted_data.append(encrypted_value)
    return encrypted_data

def decrypt_data(encrypted_data, decryptor, encoder):
    decrypted_data = []
    for value in encrypted_data:
        plain = decryptor.decrypt(value)
        decoded = encoder.decode(plain)
        decrypted_data.append(decoded[0])
    return decrypted_data

def add_differential_privacy_homomorphic(encrypted_data, epsilon, scale, encoder, evaluator):
    noisy_data = []
    for value in encrypted_data:
        noise = np.random.laplace(0, scale / epsilon)
        plain_noise = encoder.encode(noise, scale)
        noisy_value = evaluator.add_plain(value, plain_noise)
        noisy_data.append(noisy_value)
    return noisy_data

def analyze_data(data):
    mean = np.mean(data)
    variance = np.var(data)
    stdev = np.std(data)
    return mean, variance, stdev

plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォントの設定

def plot_histogram(data, title):
    plt.figure(figsize=(10, 4))
    plt.hist(data, bins=30, alpha=0.75, color='blue')
    plt.title(title)
    plt.xlabel('値')
    plt.ylabel('頻度')
    st.pyplot(plt)

def main():
    st.title('差分プライバシーの処理時間比較アプリ')

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("元のデータ:")
        st.write(data.head())

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        column = st.selectbox("ノイズを加える列を選択してください", numeric_columns)

        epsilon = st.slider("プライバシー保護レベル ε (小さいほどプライバシー保護が強化されますが、データの変動が大きくなります)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

        if st.button("ノイズを加える"):
            # 通常の差分プライバシー
            start_time = time.time()
            noisy_data_dp = add_differential_privacy(data[[column]], epsilon)
            dp_time = time.time() - start_time

            # 準同型暗号を用いた差分プライバシー
            parms = EncryptionParameters(scheme_type.ckks)
            poly_modulus_degree = 8192
            parms.set_poly_modulus_degree(poly_modulus_degree)
            parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
            context = SEALContext(parms)
            encoder = CKKSEncoder(context)
            scale = 2.0 ** 40

            keygen = KeyGenerator(context)
            public_key = keygen.create_public_key()
            secret_key = keygen.secret_key()
            relin_keys = keygen.create_relin_keys()
            encryptor = Encryptor(context, public_key)
            evaluator = Evaluator(context)
            decryptor = Decryptor(context, secret_key)

            original_data = data[column].to_numpy()

            start_time = time.time()
            encrypted_data = encrypt_data(original_data, encoder, encryptor, scale)
            noisy_encrypted_data = add_differential_privacy_homomorphic(encrypted_data, epsilon=epsilon, scale=scale, encoder=encoder, evaluator=evaluator)
            decrypted_data = decrypt_data(noisy_encrypted_data, decryptor, encoder)
            hed_time = time.time() - start_time

            # 結果の表示
            st.write(f"通常の差分プライバシーの処理時間: {dp_time:.4f} 秒")
            st.write(f"準同型暗号を用いた差分プライバシーの処理時間: {hed_time:.4f} 秒")

            st.write("通常の差分プライバシーが適用されたデータ:")
            st.write(noisy_data_dp.head())

            mean, variance, stdev = analyze_data(noisy_data_dp[column])
            st.write(f"平均: {mean}")
            st.write(f"分散: {variance}")
            st.write(f"標準偏差: {stdev}")

            st.write("ヒストグラム（通常の差分プライバシー）:")
            plot_histogram(noisy_data_dp[column], '通常の差分プライバシー')

            st.write("準同型暗号を用いた差分プライバシーが適用されたデータ（先頭5件）:")
            st.write(decrypted_data[:5])

            mean, variance, stdev = analyze_data(decrypted_data)
            st.write(f"平均: {mean}")
            st.write(f"分散: {variance}")
            st.write(f"標準偏差: {stdev}")

            st.write("ヒストグラム（準同型暗号を用いた差分プライバシー）:")
            plot_histogram(decrypted_data, '準同型暗号を用いた差分プライバシー')

if __name__ == "__main__":
    main()
