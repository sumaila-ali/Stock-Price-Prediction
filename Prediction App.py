#=====PACKAGES=====

#base gui package
import tkinter as tk
from tkinter import ttk, messagebox

# for number operations before shipping it to the LSTM model. 
import numpy as np
#for dataframe creation and operations
import pandas as pd
#Data visualisation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Data preprocessing -- Standizing the data to a standard normal distribution
from sklearn.preprocessing import StandardScaler

# Computing the RMSE for the model. 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

#pytorch model and parameters
import torch
import torch.nn as nn
import torch.optim as optim

# Google sheet data interactions and processes. 
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Scheduled-based operations. 
import time

#custom tkinter for beautifying the GUI. 
import customtkinter as ctk

# Determines if the LSMT model will be built with GPU or CPU. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== GOOGLE SHEET SETUP ======
CREDENTIALS_FILE = 'LSTM Model Use-Case/stock-prediction-app-459317-075704cd682e.json' # needs to be updated if you have your own data source.
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1dsVEvP0KD-6E86Etg8166Wlt0NEZvW0IwBPqaNAq8OY/edit?gid=0#gid=0'  # Should be replaced if you are using your own data source
HELPER_SHEET = 'Helper'
DATA_SOURCE_SHEET = 'Data Source'


# ==== FUNCTIONS ==== 


'''Writes the ticker input to the helper sheet and fetches the data from the source sheet for the model. 
The data is then converted to a dataframe for preprocessing and modeliing. '''

def fetch_data_from_google_sheet(ticker):
   
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)

    # Open spreadsheet using URL
    spreadsheet = client.open_by_url(SHEET_URL)

    # Update ticker in 'Helper' sheet
    # Update ticker in 'Helper' sheet
    helper_sheet = spreadsheet.worksheet(HELPER_SHEET)
    helper_sheet.update_acell('A1', ticker)

    # Wait for formulas in 'Data Source' sheet to update
    time.sleep(10)

    # Read data from 'Data Source' sheet
    data_sheet = spreadsheet.worksheet(DATA_SOURCE_SHEET)
    data = data_sheet.get_all_values()

    # Convert to DataFrame, skip headers or note rows
    df = pd.DataFrame(data[2:], columns=["Date", "Close"])
    df = df.replace('', np.nan).dropna()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df = df.dropna()
    df.set_index("Date", inplace=True)

    return df


'''LSTM Model Class'''

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        return self.fc(out[:, -1, :])


'''Action Runner

Executes all for the app when the prodict button is pushed. 

-- Calls the fetch data function. 
-- Plots the charts
-- Preprocesses the data for modeling. 
-- Runs the predictive model
-- 

'''
def run_prediction(ticker_symbol, canvas_frame, chart_frame, progress_bar):
    try:
        progress_bar['value'] = 10
        progress_bar.update()

        df = fetch_data_from_google_sheet(ticker_symbol)
        if df.empty:
            raise ValueError("No data fetched from Google Sheet.")

        progress_bar['value'] = 30
        progress_bar.update()

        # Chart 1: Original data
        fig_original = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig_original.add_subplot(111)
        ax.plot(df['Close'], label='Close Price')
        ax.set_title(f"{ticker_symbol} Historical Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig_original, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

        # Preprocessing
        scalar = StandardScaler()
        df['Close'] = scalar.fit_transform(df['Close'].values.reshape(-1, 1))

        progress_bar['value'] = 50
        progress_bar.update()

        sequence_length = 21
        data = [df.Close[i:i + sequence_length].values for i in range(len(df) - sequence_length)]
        data = np.expand_dims(np.array(data), axis=2)

        train_size = int(0.8 * len(data))
        X_train = torch.from_numpy(data[:train_size, :-1, :]).float().to(device)
        y_train = torch.from_numpy(data[:train_size, -1, :]).float().to(device)

        X_test = torch.from_numpy(data[train_size:, :-1, :]).float().to(device)
        y_test = torch.from_numpy(data[train_size:, -1, :]).float().to(device)


        model = PredictionModel(input_dim=1, hidden_dim=150, num_layers=2, output_dim=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        last_sequence = df.Close.values[-(sequence_length - 1):]  # shape: (20,)

        # Step 2: Reshape to (1, 20, 1) to match LSTM input: (batch_size, seq_len, input_size)
        input_seq = np.expand_dims(last_sequence, axis=(0, 2))  # shape: (1, 20, 1)
        input_tensor = torch.from_numpy(input_seq).float().to(device)

        for epoch in range(100):
            model.train()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                progress_bar['value'] += 5
                progress_bar.update()

        progress_bar['value'] = 85
        progress_bar.update()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            prediction = model(input_tensor)  # shape: (1, 1)
            next_value = prediction.item()
            

        y_test_pred = scalar.inverse_transform(y_test_pred.cpu().numpy())
        y_test = scalar.inverse_transform(y_test.cpu().numpy())

        rmse = root_mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        next_value = scalar.inverse_transform(np.array([[next_value]]))
        next_value = next_value[0][0]

    

        
        fig_pred = plt.Figure(figsize=(6, 4), dpi=100)
        ax1 = fig_pred.add_subplot(211)
        ax1.plot(df.iloc[-len(y_test):].index, y_test, label="Actual")
        ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, label="Predicted")
        ax1.set_title("Prediction vs Actual")
        ax1.legend()

        ax2 = fig_pred.add_subplot(212)
        ax2.axhline(rmse, linestyle='--', color='gray', label="RMSE")
        ax2.plot(df.iloc[-len(y_test):].index, abs(y_test - y_test_pred), label="Error", color='red')
        ax2.set_title("Prediction Error")
        ax2.legend()

        for widget in chart_frame.winfo_children():
            widget.destroy()
        canvas2 = FigureCanvasTkAgg(fig_pred, master=chart_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(expand=True, fill="both")

        progress_bar['value'] = 100
        progress_bar.update()
        return rmse, r2, next_value

    except Exception as e:
        messagebox.showerror("Error", f"Data fetch or processing failed: {e}")
        progress_bar['value'] = 0
        progress_bar.update()




 # === GUI === 

def create_app():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.title("Stock Predictor")

    # Fullscreen
    root.attributes("-fullscreen", True)  
    # press Esc to exit fullscreen
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    input_frame = ctk.CTkFrame(root)
    input_frame.pack(pady=10)

    ctk.CTkLabel(input_frame, text="Enter Ticker Symbol:", font=('Arial', 14)).pack(side="left", padx=5)
    ticker_entry = ctk.CTkEntry(input_frame, width=200, font=('Arial', 14))
    ticker_entry.pack(side="left", padx=5)

    progress = ttk.Progressbar(root, length=500, mode='determinate')
    progress.pack(pady=10)

    def on_predict():
        ticker = ticker_entry.get().strip().upper()
        if ticker:
            progress['value'] = 0
            root.update_idletasks()
            rmse, r2, next = run_prediction(ticker, chart_frame1_content, chart_frame2_content, progress)
            rmse_label.configure(text=f"{rmse:.2f}")
            r2_label.configure(text=f"{r2:.2f}")
            close_label.configure(text=f"{next:.2f}")


        else:
            messagebox.showwarning("Input Error", "Please enter a ticker symbol.")

    ctk.CTkButton(input_frame, text="Predict", command=on_predict).pack(side="left", padx=10)

    # Row 1 - Original Chart
    chart_frame1 = ctk.CTkFrame(root, height=250)
    chart_frame1.pack(fill="both", expand=True, padx=10, pady=5)
    ctk.CTkLabel(chart_frame1, text="Original Chart", font=('Arial', 16)).pack(anchor="w", padx=10, pady=5)
    chart_frame1_content = ctk.CTkFrame(chart_frame1)
    chart_frame1_content.pack(fill="both", expand=True, padx=10, pady=5)

    # Row 2 - Predicted Chart
    chart_frame2 = ctk.CTkFrame(root, height=250)
    chart_frame2.pack(fill="both", expand=True, padx=10, pady=5)
    ctk.CTkLabel(chart_frame2, text="Predicted Chart", font=('Arial', 16)).pack(anchor="w", padx=10, pady=5)
    chart_frame2_content = ctk.CTkFrame(chart_frame2)
    chart_frame2_content.pack(fill="both", expand=True, padx=10, pady=5)

    # Row 3 - Bottom info panels
    bottom_panel = ctk.CTkFrame(root, height=100)
    bottom_panel.pack(fill="x", padx=10, pady=10)

    # Panel 1: Predicted Value
    panel1 = ctk.CTkFrame(bottom_panel)
    panel1.pack(side="left", expand=True, fill="both", padx=5)
    ctk.CTkLabel(panel1, text="Predicted Closing Price", font=('Arial', 40)).pack(pady=5)
    close_label = ctk.CTkLabel(panel1, text="--", font=('Arial', 40), text_color="blue")
    close_label.pack(pady=5)

    # Panel 2: RMSE
    panel2 = ctk.CTkFrame(bottom_panel)
    panel2.pack(side="left", expand=True, fill="both", padx=5)
    ctk.CTkLabel(panel2, text="RMSE", font=('Arial', 40)).pack(pady=5)
    rmse_label = ctk.CTkLabel(panel2, text="--", font=('Arial', 40), text_color="blue")
    rmse_label.pack(pady=5)

    # Panel 3: Reserved
    panel3 = ctk.CTkFrame(bottom_panel)
    panel3.pack(side="left", expand=True, fill="both", padx=5)
    ctk.CTkLabel(panel3, text="R2 Score", font=('Arial', 40)).pack(pady=5)
    r2_label = ctk.CTkLabel(panel3, text="--", font=('Arial', 40), text_color="blue")
    r2_label.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_app()