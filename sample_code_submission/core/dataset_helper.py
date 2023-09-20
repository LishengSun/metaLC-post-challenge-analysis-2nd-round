import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()

    def fit(self, dataset):
        self.encoder.fit(dataset[self.categorical_features].to_numpy())
        self.scaler.fit(dataset[self.numerical_features].to_numpy())
        return self

    def transform(self, dataset):
        transformed = self.encoder.transform(
            dataset[self.categorical_features].to_numpy()).toarray()
        data = self.scaler.transform(
            dataset[self.numerical_features].to_numpy())

        X = np.nan_to_num(np.concatenate([data, transformed], axis=1), nan=0)

        return X
