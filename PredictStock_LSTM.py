import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime, timedelta
import re

def update_period_label(value):
    period_label.config(text=f"{value}년")

def on_hover(button, color):
    button['background'] = color

def on_leave(button, color):
    button['background'] = color

def reset_gui():
    """Resets the GUI inputs to their default state."""
    input_entry.delete(0, tk.END)
    slider.set(10)
    update_period_label(10)

def validate_input(char, action):
    """Validates input to allow only uppercase letters, numbers, and dots.
    Also allows special keys like backspace to function properly.
    """
    if action == "1":  # Insert action
        if re.match(r"^[A-Z0-9.]*$", char):
            return True
        elif re.match(r"^[a-zㄱ-ㅎㅏ-ㅣ가-힣]*$", char):
            input_entry.insert(tk.END, char.upper())
            return False
        else:
            return False
    return True

def fetch_and_predict():
    symbol = input_entry.get()
    period_years = int(slider.get())

    if not symbol:
        messagebox.showerror("Error", "Symbol field is required!")
        return

    # Show analyzing message
    analyzing_message.set(f"{symbol}을(를) 분석 중입니다...")
    analyze_button.update()

    start_date = (datetime.now() - timedelta(days=period_years * 365)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            messagebox.showerror("Error", f"No data found for symbol: {symbol}")
            return

        closing_prices = data['Close']
        closing_prices = pd.DataFrame(closing_prices)
        closing_prices.columns = ['Close']

        train_data = closing_prices[:int(len(closing_prices) * 0.8)]
        test_data = closing_prices[int(len(closing_prices) * 0.8):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X_train, y_train = create_dataset(scaled_train_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        X_test, y_test = create_dataset(scaled_test_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=2)

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

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(closing_prices.index, closing_prices['Close'], label='Actual Price', color='blue')
        ax.plot(prediction_dates, pred_price, label='Predicted Price for the next 60 days', color='red')
        ax.set_xlim(left=date2num(closing_prices.index[0]))
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        ax.xaxis.set_major_formatter(DateFormatter("%Y.%m.%d"))
        plt.xticks(rotation=45)

        # Highlight buy or sell button based on price trend
        if pred_price[-1] > pred_price[0]:
            buy_button.config(bg="green", activebackground="darkgreen")
        else:
            sell_button.config(bg="red", activebackground="darkred")

        # 상호작용 기능: 스크롤 확대
        def on_scroll(event):
            if event.inaxes == ax:
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()

                xdata = event.xdata
                ydata = event.ydata
                scale_factor = 1.1 if event.button == 'down' else 0.9

                new_xlim = [max(date2num(closing_prices.index[0]), xdata - (xdata - cur_xlim[0]) * scale_factor),
                            xdata + (cur_xlim[1] - xdata) * scale_factor]
                new_ylim = [max(0, ydata - (ydata - cur_ylim[0]) * scale_factor),
                            ydata + (cur_ylim[1] - ydata) * scale_factor]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()

        # 상호작용 기능: 드래그 이동
        press = [None, None]

        def on_press(event):
            if event.button == 2:  # Middle mouse button
                press[0] = event.xdata
                press[1] = event.ydata

        def on_motion(event):
            if press[0] is not None:
                dx = press[0] - event.xdata
                dy = press[1] - event.ydata

                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()

                new_xlim = [max(date2num(closing_prices.index[0]), cur_xlim[0] + dx), cur_xlim[1] + dx]
                new_ylim = [max(0, cur_ylim[0] + dy), cur_ylim[1] + dy]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()

        def on_release(event):
            press[0], press[1] = None, None

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)

        plt.show(block=True)

        # 플롯 창 닫힌 후 초기화
        reset_gui()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

    finally:
        # Remove analyzing message
        analyzing_message.set("")

root = tk.Tk()
root.title("Predict Stock")
root.geometry("500x450")
root.configure(bg="#181818")

analyzing_message = tk.StringVar()
analyzing_label = tk.Label(root, textvariable=analyzing_message, font=("Noto Sans KR Bold", 12), bg="#181818", fg="orange")
analyzing_label.pack(pady=5)

title_label = tk.Label(root, text="Predict Stock", font=("Noto Sans Bold", 16), bg="#181818", fg="white")
title_label.pack(pady=10)

input_frame = tk.Frame(root, bg="lightgray", width=300, height=40)
input_frame.pack(pady=10)
input_frame.pack_propagate(False)

vcmd = (root.register(validate_input), '%S', '%d')
input_entry = tk.Entry(input_frame, font=("Noto Sans KR", 12), validate="key", validatecommand=vcmd, justify="center")
input_entry.pack(fill="both", expand=True, padx=5, pady=5)

input_hint_frame = tk.Frame(root, bg="#181818")
input_hint_frame.pack()

input_hint_bold = tk.Label(input_hint_frame, text="종목의 티커를 입력하세요", font=("Noto Sans KR Bold", 12), bg="#181818", fg="white")
input_hint_bold.pack()
input_hint_light = tk.Label(input_hint_frame, text="(ex. 테슬라: TSLA, 삼성전자: 005930.KS)", font=("Noto Sans KR Light", 10), bg="#181818", fg="white")
input_hint_light.pack()

analysis_frame = tk.Frame(root, bg="#181818")
analysis_frame.pack(pady=10)

period_label = tk.Label(analysis_frame, text="10년", font=("Noto Sans KR Medium", 12), bg="#181818", fg="white")
period_label.pack(side="right", padx=10)

style = ttk.Style()
style.theme_use("default")
style.configure("Horizontal.TScale", 
    background="#181818", 
    troughcolor="gray", 
    sliderthickness=14, 
    sliderlength=20,
    sliderrelief="flat")
style.map("Horizontal.TScale",
    slidercolor=[("!pressed", "white"), ("pressed", "white")])

slider = ttk.Scale(
    analysis_frame,
    from_=1,
    to=10,
    orient="horizontal",
    style="Horizontal.TScale",
    command=lambda v: update_period_label(int(float(v)))
)
slider.set(10)
slider.pack(side="right", fill="x", expand=True)

analysis_label = tk.Label(analysis_frame, text="분석 기간", font=("Noto Sans KR Bold", 12), bg="#181818", fg="white")
analysis_label.pack(side="left", padx=10)

button_frame = tk.Frame(root, bg="#181818")
button_frame.pack(pady=10)

def create_rounded_button(parent, text, font, bg, activebackground, width, command=None):
    return tk.Button(
        parent,
        text=text,
        font=font,
        bg=bg,
        activebackground=activebackground,
        width=width,
        command=command,
        relief="flat",
        borderwidth=0,
        highlightthickness=0
    )

analyze_button = create_rounded_button(
    button_frame, "분석하기", ("Noto Sans KR Bold", 12), "orange", "darkorange", 15, fetch_and_predict
)

analyze_button.pack(side="top", pady=20)

buy_button = create_rounded_button(
    button_frame,
    text="매수",
    font=("Noto Sans KR", 12),
    bg="lightgray",
    activebackground="green",
    width=10
)

buy_button.bind("<Enter>", lambda e: on_hover(buy_button, "green"))
buy_button.bind("<Leave>", lambda e: on_leave(buy_button, "lightgray"))

sell_button = create_rounded_button(
    button_frame,
    text="매도",
    font=("Noto Sans KR", 12),
    bg="lightgray",
    activebackground="red",
    width=10
)

sell_button.bind("<Enter>", lambda e: on_hover(sell_button, "red"))
sell_button.bind("<Leave>", lambda e: on_leave(sell_button, "lightgray"))

reset_button = create_rounded_button(
    button_frame,
    text="초기화",
    font=("Noto Sans KR", 12),
    bg="lightgray",
    activebackground="blue",
    width=10,
    command=reset_gui
)

reset_button.bind("<Enter>", lambda e: on_hover(reset_button, "lightblue"))
reset_button.bind("<Leave>", lambda e: on_leave(reset_button, "lightgray"))

buy_button.pack(side="left", padx=10)
sell_button.pack(side="right", padx=10)
reset_button.pack(side="right", padx=10)

# Run the application
root.mainloop()
