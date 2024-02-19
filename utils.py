import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, boxcox

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, recall_score

# These functions take in a dataframe, find outliers based on Zscore,
# Modified Zscore and IQR method respectively, and then return a DataFrame with these outliers.


def find_outliers_zscore(column, df, threshold=3):
    # Identifies outliers using the Z-score method
    column_mean = df[column].mean()
    column_std = df[column].std()

    outliers_zscore = df.loc[(abs(df[column] - column_mean) / column_std) > threshold]
    return outliers_zscore


def find_outliers_modified_zscore(column, df, threshold=4):
    # Identifies outliers using the modified Z-score method
    column_median = df[column].median()
    median_absolute_deviation = (df[column] - column_median).abs().median()

    modified_zscores = 0.6745 * (df[column] - column_median) / median_absolute_deviation

    outliers_modified_zscore = df.loc[abs(modified_zscores) > threshold]
    return outliers_modified_zscore


def find_outliers_iqr(column, df):
    # Identifies outliers using the Interquartile Range (IQR) method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    outliers_iqr = df.loc[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outliers_iqr


def get_common_outliers(outliers_col1, outliers_col2, df):
    # Finds common outliers between two outlier DataFrames and returns them along with non-common outliers
    common_indices = outliers_col1.index.intersection(outliers_col2.index)
    common_outliers = df.loc[common_indices]

    outliers_col1_index = outliers_col1.index.difference(outliers_col2.index)
    outliers_col2_index = outliers_col2.index.difference(outliers_col1.index)

    non_common_outliers = df.loc[outliers_col1_index.union(outliers_col2_index)]

    return common_outliers, non_common_outliers



def apply_transformation(original_data, transformation):
    # Applies a specified transformation to the original data
    if transformation == 'log':
        return np.log1p(original_data)  # Logarithmic transformation
    elif transformation == 'sqrt':
        return np.sqrt(original_data)   # Square root transformation
    elif transformation == 'reciprocal':
        return 1 / (original_data + 1e-10)  # Reciprocal transformation
    elif transformation == 'square':
        return np.square(original_data)  # Square transformation
    elif transformation == 'boxcox':
        if (original_data <= 0).any():
            return original_data  # If any values are non-positive, return original data
        else:
            transformed_data, lambda_value = boxcox(original_data)  # Box-Cox transformation
            return transformed_data
    else:
        return original_data  # If transformation is not recognized, return original data


def generate_qq_plot(data, title, xlabel, ylabel, ax):
    # Generates a Q-Q plot for the given data and displays it on the specified axis
    probplot_data = probplot(data, plot=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def find_best_transformation(column_data, transformations):
    # Finds the best transformation for a given column of data based on skewness reduction
    data_series = pd.Series(column_data)
    skewness_before = data_series.skew()

    best_transformation = None
    best_skewness = skewness_before
    original_data = data_series.copy()

    for transformation in transformations:
        transformed_data = apply_transformation(original_data, transformation)
        transformed_data_series = pd.Series(transformed_data)
        skewness_after = transformed_data_series.skew()

        if abs(skewness_after) < abs(best_skewness):
            best_skewness = skewness_after
            best_transformation = transformation

    return best_transformation, skewness_before, best_skewness


def transform_data(data, best_transformation):
    # Applies the best transformation to the given data
    return apply_transformation(data, best_transformation)


def compare_qq_plots(original_data, transformed_data, column_name, transformation=None):
    # Generates and displays a pair of Q-Q plots comparing the original and transformed data
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Q-Q Plot for the original data
    generate_qq_plot(original_data, f'Q-Q Plot for {column_name}', 'Theoretical Quantiles', f'Sample Quantiles of {column_name}', axs[0])

    # Q-Q Plot for the transformed data (if a transformation is applied)
    if transformation:
        generate_qq_plot(transformed_data, f'Q-Q Plot for {column_name} after {transformation} transformation', 'Theoretical Quantiles', f'Sample Quantiles of {column_name} (Transformed)', axs[1])
    else:
        axs[1].set_title('No Transformation Applied')

    plt.show()


def apply_transformations_to_set(data_set, transformations):
    # Applies the best transformations to numerical columns in a DataFrame
    transformed_set = data_set.copy()
    applied_transformations = {}

    for col in transformed_set.select_dtypes(include=['number']).columns:
        best_transformation, _, _ = find_best_transformation(transformed_set[col], transformations)

        if best_transformation is not None and col not in applied_transformations:
            transformed_set[col] = transform_data(transformed_set[col], best_transformation)
            applied_transformations[col] = best_transformation

    return transformed_set, applied_transformations


def generate_classification_report_with_recall_and_f1_score(y_true, y_pred):
    # Generates a classification report with recall and F1 scores for class 1, along with accuracy score
    report = classification_report(y_true, y_pred)
    report_lines = report.split('\n')[:6]  # Extract the first six lines of the report
    class_1_recall = float(report_lines[3].split()[2])  # Extract recall score for class 1
    class_1_f1 = float(report_lines[3].split()[3])  # Extract F1 score for class 1
    accuracy = float(report_lines[5].split()[1])  # Extract accuracy score
    print("Classification Report:")
    print('\n'.join(report_lines))
    print("Class 1 Recall:", class_1_recall)
    print("Class 1 F1 Score:", class_1_f1)
    print("Accuracy Score:", accuracy)
    return class_1_recall, class_1_f1, accuracy


def generate_confusion_matrix(y_true, y_pred):
    # Generates and displays a confusion matrix based on true and predicted labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def tune_model(model, param_grid, X_train, y_train):
    # Tunes a machine learning model using Grid Search with recall as the scoring metric
    custom_scorer = make_scorer(recall_score)  # Create a scorer for recall
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=custom_scorer)  # Instantiate Grid Search with custom scoring
    grid_search.fit(X_train, y_train)  # Fit Grid Search
    print("Best Hyperparameters:", grid_search.best_params_)  # Print best hyperparameters
    print("Best Recall Score:", grid_search.best_score_)  # Print best recall score
    best_model = grid_search.best_estimator_  # Get the best model
    return best_model
