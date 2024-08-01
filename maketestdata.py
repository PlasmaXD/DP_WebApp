import pandas as pd
import numpy as np

# ランダムな日付を生成
dates = pd.date_range(start="2021-01-01", periods=100, freq='B')  # 営業日のみ

# ランダムな株価データを生成
np.random.seed(42)  # 乱数シードを設定して再現性を確保
opening_prices = np.random.uniform(100, 200, size=len(dates))
high_prices = opening_prices + np.random.uniform(0, 10, size=len(dates))
low_prices = opening_prices - np.random.uniform(0, 10, size=len(dates))
closing_prices = np.random.uniform(low_prices, high_prices)

# データフレームを作成
stock_data = pd.DataFrame({
    '日付': dates,
    '始値': opening_prices,
    '高値': high_prices,
    '安値': low_prices,
    '終値': closing_prices
})

# CSVファイルとして保存
stock_data.to_csv("test_stock_data.csv", index=False, encoding='utf-8-sig')
