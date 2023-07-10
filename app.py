import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Split the dataset into training data and new data
train_data = iris_df.sample(frac=0.7, random_state=42)
new_data = iris_df.drop(train_data.index)

# Set a significance level
alpha = 0.05

# Perform the Kolmogorov-Smirnov test for each feature
for feature in iris_df.columns:
    ks_statistic, p_value = ks_2samp(train_data[feature], new_data[feature])

    print(f"Feature: {feature}")
    if p_value < alpha:
        print("Data drift detected. The distributions are significantly different.")
    else:
        print("No data drift detected. The distributions are not significantly different.")
    print()