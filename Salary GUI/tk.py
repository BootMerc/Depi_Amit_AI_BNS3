import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv(r'Salary_Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 1]

# Scale features for better accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Try Linear, Ridge, and Lasso regression
regressor = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

regressor.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
lasso.fit(X_train, Y_train)

# Evaluate models
lin_pred = regressor.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

lin_mse = mean_squared_error(Y_test, lin_pred)
ridge_mse = mean_squared_error(Y_test, ridge_pred)
lasso_mse = mean_squared_error(Y_test, lasso_pred)

lin_r2 = r2_score(Y_test, lin_pred)
ridge_r2 = r2_score(Y_test, ridge_pred)
lasso_r2 = r2_score(Y_test, lasso_pred)

app = tk.Tk()
app.title("Experience vs Salary Prediction")
app.geometry("440x260")
app.config(bg="#e6f2ff")  # lighter blue shade
# Title label
title_lbl = tk.Label(app, text="Salary Prediction App", bg="#e6f2ff", fg="#003366", font=("Arial", 16, "bold"))
title_lbl.pack(pady=(16, 6))

# Frame for input
input_frame = tk.Frame(app, bg="#e6f2ff")
input_frame.pack(pady=8)

lbl = tk.Label(input_frame, text="Enter Years of Experience:", bg="#e6f2ff", fg="black", font=("Arial", 12))
lbl.grid(row=0, column=0, padx=6, pady=6)

txt_input = tk.Entry(input_frame, width=18, font=("Arial", 12))
txt_input.grid(row=0, column=1, padx=6, pady=6)
# Result label
result_lbl = tk.Label(app, text="", bg="#e6f2ff", fg="#006600", font=("Arial", 13, "bold"))
result_lbl.pack(pady=10)

def show_prediction():
    value = txt_input.get()
    result_lbl.config(text="")

    if not value:
        messagebox.showerror("Error", "Please enter a valid number!")
        return

    try:
        exp = float(value)
    except:
        messagebox.showerror("Error", "Only numeric values are allowed!")
        return

    if exp < 0 or exp > 65:
        messagebox.showerror("Error", "Please enter a value between 0 and 65.")
        return
    # Scale input
    exp_scaled = scaler.transform([[exp]])
    pred_lin = regressor.predict(exp_scaled)[0]
    pred_ridge = ridge.predict(exp_scaled)[0]
    pred_lasso = lasso.predict(exp_scaled)[0]

    result_lbl.config(text=(
        f"Linear: ${pred_lin:,.2f}\n"
        f"Ridge: ${pred_ridge:,.2f}\n"
        f"Lasso: ${pred_lasso:,.2f}"
    ))
metrics_lbl = tk.Label(app, text=(
    f"Linear MSE: {lin_mse:.2f}, R²: {lin_r2:.3f}\n"
    f"Ridge MSE: {ridge_mse:.2f}, R²: {ridge_r2:.3f}\n"
    f"Lasso MSE: {lasso_mse:.2f}, R²: {lasso_r2:.3f}"
), bg="#e6f2ff", fg="#333333", font=("Arial", 10))
metrics_lbl.pack(pady=4)

btn = tk.Button(app, text="Predict Salary", command=show_prediction, bg="#005c99", fg="white", font=("Arial", 12, "bold"), width=18, height=1)
btn.pack(pady=12)

app.mainloop()