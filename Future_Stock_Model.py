import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

def fetch_and_predict():
    # 입력값 가져오기
    symbol = symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    # 입력값 검증
    if not symbol or not start_date or not end_date:
        messagebox.showerror("Error", "All fields are required!")
        return

    try:
        # 데이터 다운로드
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            messagebox.showerror("Error", f"No data found for symbol: {symbol}")
            return

        if ('Close', symbol) not in data.columns:
            messagebox.showerror("Error", "Unable to find 'Close' data for the given symbol.")
            return

        closing_prices = data[('Close', symbol)]
        closing_prices = pd.DataFrame(closing_prices)
        closing_prices.columns = ['Close']

        # 데이터 분할
        train_data = closing_prices[:int(len(closing_prices) * 0.8)]
        test_data = closing_prices[int(len(closing_prices) * 0.8):]

        # 데이터 정규화
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        # 데이터셋 생성 함수
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        # 시퀀스 길이 설정
        time_step = 60

        # 학습 데이터셋 생성
        X_train, y_train = create_dataset(scaled_train_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 테스트 데이터셋 생성
        X_test, y_test = create_dataset(scaled_test_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM 모델 생성
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # 모델 컴파일
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 모델 학습 - 정확성을 높이기 위해 에포크 수와 배치 크기 조정
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=2)

        # 예측
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # 데이터 역정규화
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # 결과 시각화
        plt.figure(figsize=(14, 5))
        plt.plot(closing_prices.index, closing_prices['Close'], label='Actual Price')
        plt.plot(train_data.index[time_step + 1:], train_predict, label='Train Predict')
        plt.plot(test_data.index[time_step + 1:], test_predict, label='Test Predict')

        # 마지막 날짜 이후 60일 동안의 예측
        last_60_days = closing_prices[-time_step:].values
        last_60_days_scaled = scaler.transform(last_60_days)

        X_last = []
        X_last.append(last_60_days_scaled)
        X_last = np.array(X_last)
        X_last = np.reshape(X_last, (X_last.shape[0], X_last.shape[1], 1))

        pred_price = []
        for _ in range(60):
            pred = model.predict(X_last)
            pred_price.append(pred[0])
            X_last = np.append(X_last[:, 1:, :], [pred], axis=1)

        pred_price = scaler.inverse_transform(pred_price)

        prediction_dates = pd.date_range(start=closing_prices.index[-1], periods=61)[1:]
        plt.plot(prediction_dates, pred_price, label='Predicted Price for the next 60 days')

        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI 설정
root = tk.Tk()
root.title("Stock Prediction with LSTM")
root.geometry("400x350")

# 입력 필드
symbol_label = tk.Label(root, text="Stock Symbol:")
symbol_label.pack(pady=5)
symbol_entry = tk.Entry(root)
symbol_entry.pack(pady=5)

start_date_label = tk.Label(root, text="Start Date:")
start_date_label.pack(pady=5)
start_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
start_date_entry.pack(pady=5)

end_date_label = tk.Label(root, text="End Date:")
end_date_label.pack(pady=5)
end_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
end_date_entry.pack(pady=5)

# 실행 버튼
submit_button = tk.Button(root, text="Fetch and Predict", command=fetch_and_predict)
submit_button.pack(pady=20)

# GUI 실행
root.mainloop()
