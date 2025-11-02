import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the trained model and scaler
try:
    model = joblib.load("random_forest_crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    messagebox.showerror("Error", "Model or scaler files not found. Make sure 'random_forest_crop_model.pkl' and 'scaler.pkl' are in the same directory.")
    exit()

def predict_crop():
    """
    Predicts the crop based on user input from the GUI.
    """
    try:
        # Get values from the input fields
        n = float(entry_n.get())
        p = float(entry_p.get())
        k = float(entry_k.get())
        temperature = float(entry_temp.get())
        humidity = float(entry_humidity.get())
        ph = float(entry_ph.get())
        rainfall = float(entry_rainfall.get())

        # Create a numpy array from the inputs
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)

        # Display the result
        result_label.config(text=f"Recommended Crop: {prediction[0].capitalize()}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")

# --- GUI Setup ---
root = tk.Tk()
root.title("Crop Recommendation System")
root.geometry("400x350")

# Create a frame for the input fields
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(expand=True)

# Labels and Entry fields for each feature
labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)", "pH Value", "Rainfall (mm)"]
entries = []

for i, label_text in enumerate(labels):
    label = tk.Label(frame, text=label_text)
    label.grid(row=i, column=0, sticky="w", pady=5)
    entry = tk.Entry(frame)
    entry.grid(row=i, column=1, pady=5)
    entries.append(entry)

(entry_n, entry_p, entry_k, entry_temp, entry_humidity, entry_ph, entry_rainfall) = entries

# Prediction Button
predict_button = tk.Button(root, text="Predict Crop", command=predict_crop, font=("Helvetica", 12, "bold"))
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="Recommended Crop: ", font=("Helvetica", 14, "italic"))
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
