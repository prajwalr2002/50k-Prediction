from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your dataset and preprocess it
df = pd.read_csv(r'C:\Users\DELL\Downloads/adult.csv')
df = df.rename(columns={
    '39': 'age',
    ' State-gov': 'workclass',
    ' 77516': 'fnlwgt',
    ' Bachelors': 'Education',
    ' 13': 'education_num',
    ' Adm-clerical': 'occupation',
    ' Never-married': 'marital_status',
    ' Not-in-family': 'relationship',
    ' White': 'race',
    ' Male': 'sex',
    ' 2174': 'capital_gain',
    ' 0': 'capital_loss',
    ' 40': 'hours_per_week',
    ' United-States': 'native_country',
    ' <=50K': 'Target'
})

# Drop unnecessary columns and handle missing values
df.drop(labels=['Education', 'fnlwgt'], axis=1, inplace=True)
df.replace({' ?': np.nan}, inplace=True)
df.fillna(method='ffill', inplace=True)

# Assume categorical_columns contains the names of categorical columns in your dataset
categorical_columns = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# Use one-hot encoding for categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Split the data into features and target variable
X = df_encoded.drop('Target', axis=1)
y = df_encoded['Target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
RFC = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=7, min_samples_split=20)
RFC.fit(x_train, y_train)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predicting income
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_values = request.form.to_dict()

        # Check for empty input
        if not input_values:
            return render_template('error.html', error_message='Empty input. Please enter values.')

        # Preprocess input values to match the format used during model training
        input_df = pd.DataFrame([input_values])

        # Handle numeric input types
        input_df = input_df.apply(pd.to_numeric, errors='ignore')

        # Use one-hot encoding for categorical columns
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns)

        # Ensure the input_df_encoded has the same columns as the model was trained on
        missing_columns = set(X.columns) - set(input_df_encoded.columns)
        for column in missing_columns:
            input_df_encoded[column] = 0

        # Reorder the columns to match the order during training
        input_df_encoded = input_df_encoded[X.columns]

        # Make a prediction using the trained model
        prediction = RFC.predict(input_df_encoded)

        # Return the prediction to the user
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        # Log the exception
        app.logger.error(f"Prediction error: {str(e)}")
        # Return an error message to the userpip
        return render_template('error.html', error_message='An error occurred during prediction. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
