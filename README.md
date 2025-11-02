# IOT_SOIl - Crop Recommendation System

This project aims to recommend the most suitable crop for a given soil profile based on various environmental factors. It utilizes a Random Forest machine learning model trained on a comprehensive dataset of soil and crop data.

## Dataset

The dataset used for training the model is `Crop_recommendation.csv`. It contains the following features:

-   **N**: Nitrogen content in the soil
-   **P**: Phosphorus content in the soil
-   **K**: Potassium content in the soil
-   **temperature**: Temperature in Celsius
-   **humidity**: Relative humidity in %
-   **ph**: pH value of the soil
-   **rainfall**: Rainfall in mm

The target variable is **label**, which represents the recommended crop.

## Model

The core of this project is a Random Forest Classifier implemented in the `random_forest.ipynb` Jupyter notebook. The notebook covers the following steps:

1.  **Data Loading and Inspection**: The dataset is loaded using pandas for initial analysis and inspection.
2.  **Data Preprocessing**: The features are standardized using `StandardScaler` to ensure that all features contribute equally to the model's performance.
3.  **Train-Test Split**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
4.  **Model Training**: A Random Forest Classifier is trained on the training data.
5.  **Model Evaluation**: The model's performance is evaluated using metrics such as accuracy, a classification report, and a confusion matrix. The model achieves an accuracy of approximately 99.32%.
6.  **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal hyperparameters for the Random Forest model to improve its performance.
7.  **Feature Importance**: The importance of each feature in the decision-making process of the model is analyzed and visualized.
8.  **Model Saving**: The trained model and the scaler are saved to disk using `joblib` for future use in a production environment.

## Installation

To set up the environment and run this project, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/PIO-VIA/IA_SOIL.git
    cd IA_SOIL
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The trained model is saved as `random_forest_crop_model.pkl` and the scaler as `scaler.pkl`. You can load these files to make predictions on new data.

Example of how to load the model and make a prediction:

```python
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("random_forest_crop_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example data for prediction
# [N, P, K, temperature, humidity, ph, rainfall]
new_data = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Make a prediction
prediction = model.predict(new_data_scaled)

print(f"The recommended crop is: {prediction[0]}")
```

