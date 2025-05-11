import yfinance as yf
import pandas as pd
import time

tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'XOM', 'CVX', 'COP', 
           'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 'MRK', 'KO', 'PG', 
           'PEP', 'UNP', 'CAT', 'LIN', 'NEM', 'NEE', 'DUK']

start_date = "2021-01-01"
end_date = "2025-01-01"
split_date = "2024-01-01"

data_list = []

# Download adjusted close price for each ticker
for ticker in tickers:
    success = False
    attempts = 0
    while not success and attempts < 5:
        try:
            print(f"Downloading {ticker} (Attempt {attempts + 1})...")
            df = yf.download(ticker, start=start_date, end=end_date)['Close']
            if df.empty:
                raise ValueError("Empty data returned.")
            df.name = ticker
            data_list.append(df)
            success = True
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            attempts += 1
            time.sleep(30 + attempts * 10)

if data_list:
    data = pd.concat(data_list, axis=1)
    data.index.name = 'Date'

    data.to_csv("stock_close_prices_full.csv")

    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    train_data.to_csv("stock_close_prices_train.csv")
    test_data.to_csv("stock_close_prices_test.csv")

    print("✅ All data saved:")
    print(" - Full data: stock_close_prices_full.csv")
    print(" - Training set: stock_close_prices_train.csv")
    print(" - Test set: stock_close_prices_test.csv")
else:
    print("❌ No data successfully downloaded.")