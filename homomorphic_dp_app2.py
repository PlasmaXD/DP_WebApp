import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seal import *
from pydp.algorithms.laplacian import BoundedMean

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
def analyze_data(data):
    mean = np.mean(data)
    variance = np.var(data)
    stdev = np.std(data)
    return mean, variance, stdev
# def add_differential_privacy(encrypted_data, epsilon, scale, encoder, evaluator, context):
#     noisy_data = []
#     for value in encrypted_data:
#         bounded_mean = BoundedMean(epsilon, 0, -1000000, 1000000)  # 適切な範囲を設定
#         noise = bounded_mean.quick_result([0])[0]  # ノイズ生成
#         plain_noise = encoder.encode(noise, scale)
#         noisy_value = evaluator.add_plain(value, plain_noise)
#         noisy_data.append(noisy_value)
#     return noisy_data

# def add_differential_privacy(encrypted_data, epsilon, scale, encoder, evaluator, context):
#     noisy_data = []
#     scale_for_noise = 1 / epsilon  # ラプラスノイズのスケールを定義

#     for value in encrypted_data:
#         noise = np.random.laplace(0, scale_for_noise)  # ラプラスノイズを生成
#         plain_noise = encoder.encode(noise, scale)  # ノイズを暗号化
#         noisy_value = evaluator.add_plain(value, plain_noise)  # ノイズを加える
#         noisy_data.append(noisy_value)
#     return noisy_data

def add_differential_privacy(encrypted_data, epsilon, encoder, evaluator, scale):
    noisy_data = []
    # BoundedMeanを使用してノイズを加える
    bounded_mean = BoundedMean(epsilon, 0, -1000, 1000)  # 適当な範囲を設定

    for value in encrypted_data:
        # 0を基準としてノイズを計算
        noise = bounded_mean.quick_result([0])
        # ノイズをエンコード（スケールを指定して呼び出す）
        plain_noise = encoder.encode(noise, scale)
        # 暗号化された値に平文ノイズを加える
        noisy_value = evaluator.add_plain(value, plain_noise)
        noisy_data.append(noisy_value)
    return noisy_data


def main():
    st.title('差分プライバシー データ変換アプリ')

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("元のデータ:")
        st.write(data.head())

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        column = st.selectbox("ノイズを加える列を選択してください", numeric_columns)

        epsilon = st.slider("プライバシー保護レベル ε (小さいほどプライバシー保護が強化されますが、データの変動が大きくなります)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

        if st.button("ノイズを加える"):
            parms = EncryptionParameters(scheme_type.ckks)
            poly_modulus_degree = 8192
            parms.set_poly_modulus_degree(poly_modulus_degree)
            parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
            context = SEALContext(parms)
            encoder = CKKSEncoder(context)
            scale = 2.0 ** 40

            keygen = KeyGenerator(context)
            public_key = keygen.create_public_key()  # 正しいメソッドを使用
            secret_key = keygen.secret_key()  # 秘密鍵を取得
            encryptor = Encryptor(context, public_key)
            evaluator = Evaluator(context)
            decryptor = Decryptor(context, secret_key)


            original_data = data[column].to_numpy()

            encrypted_data = encrypt_data(original_data, encoder, encryptor, scale)
            # 関数呼び出しの部分
            noisy_encrypted_data = add_differential_privacy(encrypted_data, epsilon, encoder, evaluator, scale)

            decrypted_data = decrypt_data(noisy_encrypted_data, decryptor, encoder)

            st.write("ノイズが加えられたデータ:")
            st.write(decrypted_data[:5])

            mean, variance, stdev = analyze_data(decrypted_data)
            st.write(f"平均: {mean}")
            st.write(f"分散: {variance}")
            st.write(f"標準偏差: {stdev}")

            # ヒストグラムの表示
            st.write("ヒストグラム:")
            plt.rcParams['font.family'] = 'IPAGothic'  # 日本語フォントの設定
            fig, ax = plt.subplots()
            ax.hist(decrypted_data, bins=30, alpha=0.75, color='blue')
            ax.set_title(f'ヒストグラム（{column}）')
            ax.set_xlabel('値')
            ax.set_ylabel('頻度')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
