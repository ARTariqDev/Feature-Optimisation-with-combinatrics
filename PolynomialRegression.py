import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures, LabelEncoder
import matplotlib.pyplot as plt

"""
Basic concept: use combos (nCr) and find the combo that produces the best results and use that combo in our model
We will use encoding here to simplify the problem since the classification target only has 3 distinct types
We assign an integer to each type using encoding therefore no need for logistic regression
if each classification or regression based feature is not linearly related or produces a subpar r^2 score,
we use polynomial regression.For continuous data, there is no need for encoding since it is already numeric
in addition, we will use imputing and fill any missing values with the mean although this dataset  is
mostly cleared and filtered. Speaking of which, when using regression, we filter out all non numeric 
columns for efficiency. In this way this feature selection algorithm makes it possible to 
fit a model for any classification or regression task. I have distributed the code into 3 files each showcasing 
the various regression/classification types we are using to predict a target column.Below is the code for each file:
NOTE: This data set has mostly numeric types save for id. This is why filtering non numeric data while making combos 
does not affect the model score and also because we are using basic regression types only.
"""

def load_and_process_data(dataset_path):
    # Load the dataset into a DataFrame
    data = pd.read_csv(dataset_path)

    # Custom impute function
    def impute_missing_with_mean(df):
        impute = SimpleImputer(strategy='mean')
        return impute.fit_transform(df)

    def evaluate_linear_regression(X, y):
        # Filter out non-numeric columns for linear regression
        numeric_columns = X.select_dtypes(include=['number']).columns
        X_numeric = X[numeric_columns]

        X_numeric = impute_missing_with_mean(X_numeric)  # Impute missing values with mean

        best_r2_linear = -float('inf')
        best_feature_combination_linear = []

        # Convert X_numeric back to a DataFrame
        X_numeric = pd.DataFrame(X_numeric, columns=numeric_columns)

        # Generate combinations of feature names for linear regression in a list and use a loop to fit them all to the model
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



        return best_feature_combination_linear, best_r2_linear

    def evaluate_polynomial_regression(X, y):
        if np.unique(y).size > 1:  # Check if there are multiple unique values in y
            r2_threshold = 0.90  # Adjust this threshold as needed
            best_features_linear, best_r2_linear = evaluate_linear_regression(X, y)

            if best_r2_linear < r2_threshold:
                print(f"R^2 score for Linear Regression ({best_r2_linear:.2f}) is below {r2_threshold}.")
                print("Switching to Polynomial Regression.")

                # Remove non-numeric columns
                numeric_columns = X.select_dtypes(include=['number']).columns
                X_numeric = X[numeric_columns]

                # Apply polynomial regression
                poly = PolynomialFeatures(degree=3)  # You can adjust the degree as needed
                X_poly = poly.fit_transform(X_numeric)

                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                r2 = r2_score(y, y_pred)

                print(f"Best Features for Polynomial Regression: {best_features_linear}")
                print(f"Best R^2 Score for Polynomial Regression: {r2:.1g}")

                """
                Calculate std and variance for actual and predicted values (for the graph)
                this way we can use numerics to back up our claims of model accuracy
                this in essence is kind of like graphically representing a confusion matrix
                further backed up with the actual and predicted var/std.
                """
                # Calculate std and variance for actual and predicted values
                std_actual = np.std(y)
                var_actual = np.var(y)
                std_predicted = np.std(y_pred)
                var_predicted = np.var(y_pred)
                # Plot a line of best fit
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                # axes[0] just refers to the first subplot in multiple subplots
                # Plot actual vs. predicted
                axes[0].plot(y, y_pred, 'o', label="Actual vs. Predicted(polynomial)")
                axes[0].set_title("Actual vs. Predicted Values")
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                # Fit a regression line
                coefficients = np.polyfit(y, y_pred, 1)
                line_of_best_fit = np.poly1d(coefficients)
                axes[0].plot(y, line_of_best_fit(y), label="Regression Line of Best Fit", color='red')
                axes[0].legend()
                axes[0].grid(True)

                # Plot std values  note that axes[i] = 1
                axes[1].bar(['Actual', 'Predicted'], [std_actual, std_predicted], color=['blue', 'orange'])
                axes[1].set_title("Standard Deviation (std)")
                axes[1].set_ylabel("Values")

                # Plot var values note that axes[i] = 2 so axes[0,1,2] gives us 3 different plots
                axes[2].bar(['Actual', 'Predicted'], [var_actual, var_predicted], color=['blue', 'orange'])
                axes[2].set_title("Variance (var)")
                axes[2].set_ylabel("Values")

                plt.tight_layout()
                plt.show()  # Display the plot




    # Example usage
    target_column_name = 'SepalLengthCm'
    X = data.drop(target_column_name, axis=1)  # Assuming 'Species' is the target variable
    y = data[target_column_name]

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Evaluate Polynomial Regression when R-squared is less than 0.90
    evaluate_polynomial_regression(X, y)



# Example call to the function
load_and_process_data('iris.csv')  # Replace with your dataset path
