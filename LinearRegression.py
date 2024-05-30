# Import necessary libraries
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define a function to load and process data for linear regression
def load_and_process_data(dataset_path):
    # Load the dataset into a DataFrame
    data = pd.read_csv(dataset_path)

    # Define a custom impute function
    def impute_missing_with_mean(df):
        impute = SimpleImputer(strategy='mean')
        return impute.fit_transform(df)

    # Define a function to evaluate linear regression
    def evaluate_linear_regression(X, y):
        # Filter out non-numeric columns for linear regression
        numeric_columns = X.select_dtypes(include=['number']).columns
        X_numeric = X[numeric_columns]

        # Impute missing values with mean
        X_numeric = impute_missing_with_mean(X_numeric)

        # Initialize variables to track the best linear regression results
        best_r2_linear = -float('inf')
        best_feature_combination_linear = []

        # Convert X_numeric back to a DataFrame
        X_numeric = pd.DataFrame(X_numeric, columns=numeric_columns)

        # Generate combinations of feature names for linear regression
        feature_names = list(X_numeric.columns)
        for r in range(1, min(len(feature_names) + 1, 4)):  # Limit to 3 features
            feature_combinations = combinations(feature_names, r)
            for feature_combination in feature_combinations:
                feature_subset = X_numeric[list(feature_combination)]
                model = LinearRegression()
                model.fit(feature_subset, y)
                y_pred = model.predict(feature_subset)
                r2 = r2_score(y, y_pred)
                if r2 > best_r2_linear:
                    best_r2_linear = r2
                    best_feature_combination_linear = list(feature_combination)

        # Print the best feature combo and R-squared value
        print(f"The best feature combo is: {best_feature_combination_linear} \nThe best R^2 value produced is: {best_r2_linear:.1g}")

        # Plot a line of best fit
        plt.figure(figsize=(10, 6))
        plt.plot(y, y_pred, 'o', label="Actual vs. Predicted")
        plt.title("Actual vs. Predicted Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")

        # Fit a regression line
        coefficients = np.polyfit(y, y_pred, 1)
        line_of_best_fit = np.poly1d(coefficients)
        # By using np.polyfit with a degree of 1,
        # we're essentially fitting a straight line for linear regression which is better for consistency
        # since we are using multiple types of regression models.
        plt.plot(y, line_of_best_fit(y), label="regression Line of Best Fit", color='red')

        plt.legend()
        plt.grid(True)
        plt.show()

        return best_feature_combination_linear, best_r2_linear

    # Example usage
    target_column_name = 'PetalWidthCm'
    X = data.drop(target_column_name, axis=1)  # Assuming 'Species' is the target variable
    y = data[target_column_name]

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Evaluate Polynomial Regression when R-squared is less than 0.90
    evaluate_linear_regression(X, y)

# Example call to the function
load_and_process_data('iris.csv')  # Replace with your dataset path
