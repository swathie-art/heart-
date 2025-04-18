import joblib
import numpy as np

# Load the saved model
model = joblib.load('heart_disease_model.pkl')

# Example input data (replace with real values)
# Format: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
input_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
input_array = np.array(input_data).reshape(1, -1)

# Predict
prediction = model.predict(input_array)

# Show result
if prediction[0] == 1:
    print("The person is likely to have heart disease.")
else:
    print("The person is unlikely to have heart disease.")