from seal import *
import numpy as np

def print_vector(vector):
    print('[ ', end='')
    for i in range(0, min(8, len(vector))):
        print(f'{vector[i]:.6f}', end=', ')
    print('... ]')

def encrypt_data(data, encoder, encryptor, scale):
    encrypted_data = []
    for value in data:
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

def add_differential_privacy(encrypted_data, epsilon, scale, encoder, evaluator):
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

def main():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    context = SEALContext(parms)
    encoder = CKKSEncoder(context)
    slot_count = encoder.slot_count()
    scale = 2.0 ** 40

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    encrypted_data = encrypt_data(data, encoder, encryptor, scale)
    print("Encrypted data:")
    print(encrypted_data)

    noisy_encrypted_data = add_differential_privacy(encrypted_data, epsilon=1.0, scale=scale, encoder=encoder, evaluator=evaluator)
    print("Noisy encrypted data:")
    print(noisy_encrypted_data)

    decrypted_data = decrypt_data(noisy_encrypted_data, decryptor, encoder)
    print("Decrypted data with differential privacy:")
    print_vector(decrypted_data)

    # 統計分析の実行
    mean, variance, stdev = analyze_data(decrypted_data)
    print(f"Mean: {mean}, Variance: {variance}, Standard Deviation: {stdev}")

if __name__ == "__main__":
    main()
