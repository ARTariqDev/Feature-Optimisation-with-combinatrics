import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(dataset_path):
    # Load the dataset into a DataFrame
    data = pd.read_csv(dataset_path)

    # Custom impute function
    def impute_missing_with_mean(df):
        impute = SimpleImputer(strategy='mean')
        return impute.fit_transform(df)

    def evaluate_linear_regression(X_train, X_predict, y):
        X_train = impute_missing_with_mean(X_train)  # Impute missing values with mean

        numeric_columns = np.arange(X_train.shape[1])  # Get all columns initially
        numeric_mask = np.all(np.isfinite(X_train), axis=0)  # Create a mask for numeric columns
        numeric_columns = numeric_columns[numeric_mask]  # Keep only numeric columns

        X_train = X_train[:, numeric_mask]  # Select numeric columns from X_train

        best_r2_linear = -float('inf')
        best_feature_combination_linear = []

        # Generate combinations of feature indices for linear regression
        feature_indices = np.arange(X_train.shape[1])
        for r in range(1, min(X_train.shape[1] + 1, 4)):  # Limit to 3 features
            feature_combinations = combinations(feature_indices, r)
            for feature_combination in feature_combinations:
                feature_subset = X_train[:, list(feature_combination)]
                model = LinearRegression()
                model.fit(feature_subset, y)
                y_pred = model.predict(feature_subset)
                r2 = r2_score(y, y_pred)
                if r2 > best_r2_linear:
                    best_r2_linear = r2
                    best_feature_combination_linear = list(feature_combination)

                # Predict using X_predict
        X_predict_numeric = X_predict[:, list(feature_combination)]
        y_pred = model.predict(X_predict_numeric)

        # Create a scatter plot
        plt.scatter(x=y_pred, y=y, marker='o', c=y, cmap='viridis', label='Data Points')
        plt.colorbar()
        plt.xlabel("Predicted Values (y_pred)")
        plt.ylabel("Categories (y)")
        plt.title("Categorical Scatter Plot of y vs. y_pred")

        # Calculate and plot the regression line
        m, b = np.polyfit(y_pred, y, 1)  # Calculate the slope and intercept of the regression line
        plt.plot(y_pred, m * y_pred + b, color='red', label='Regression Line')

        plt.legend()
        plt.show()

        return best_feature_combination_linear, best_r2_linear

    def evaluate_polynomial_regression(X_train, X_predict, y):
        if np.unique(y).size > 1:
            r2_threshold = 0.90
            best_features_linear, best_r2_linear = evaluate_linear_regression(X_train, X_predict, y)

            if best_r2_linear < r2_threshold:
                print(f"R^2 score for Linear Regression ({best_r2_linear:.2f}) is below {r2_threshold}.")
                print("Switching to Polynomial Regression.")

                poly = PolynomialFeatures(degree=2)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_predict = poly.transform(X_predict)

                model = LinearRegression()
                model.fit(X_poly_train, y)
                y_pred = model.predict(X_poly_train)
                r2 = r2_score(y, y_pred)

                print("Best Features for Polynomial Regression:", best_features_linear)
                print("Best R^2 Score for Polynomial Regression:", r2)
            else:
                print("Best Features for Linear Regression:", best_features_linear)
                print("Best R^2 Score for Linear Regression:", best_r2_linear)

    target_column_name = 'Species'
    X = data.drop(target_column_name, axis=1)
    y = data[target_column_name]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Convert X to a NumPy array
    X = X.to_numpy()

    # Call the functions with X for training and X for prediction
    evaluate_linear_regression(X, X, y)
    evaluate_polynomial_regression(X, X, y)

# Example call to the function
load_and_process_data('iris.csv')  # Replace with your dataset path
