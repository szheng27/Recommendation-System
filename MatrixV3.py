#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class RecommendationMeasureStevenZ:
    @staticmethod
    def random_replace(row):
        # Find non-zero indices
        non_zero_indices = row.nonzero()[1]
        # Randomly select 5 indices
        replace_indices = np.random.choice(non_zero_indices, size=min(5, len(non_zero_indices)), replace=False)
        # Replace selected indices with 0
        row[0, replace_indices] = 0
        return row

    @staticmethod
    def randomize_non_zero_values(dataset, n):
        for i, row in dataset.iterrows():
            non_zero_indices = row.index[row != 0]
            random_indices = np.random.choice(non_zero_indices, size=min(n, len(non_zero_indices)), replace=False)
            dataset.loc[i, random_indices] = 0
        return dataset
    
    @staticmethod
    def randomize_remove_n(dataset, n):
        
        dataset_copy = dataset.copy()
    
        columns_to_modify = dataset_copy.columns.drop('user_id')
        
        for i, row in dataset_copy.iterrows():
            non_zero_indices = row[columns_to_modify].index[row[columns_to_modify] != 0]
            random_indices = np.random.choice(non_zero_indices, size=min(n, len(non_zero_indices)), replace=False)
            dataset_copy.loc[i, random_indices] = 0
        
        return dataset_copy

    @staticmethod
    def generate_random_prediction(df, n):
        """
        Generate a random prediction matrix with specified number of 1 values per row.

        Parameters:
            df (DataFrame): Input DataFrame to match the shape of the generated matrix.
            n (int): Number of 1 values to generate per row.

        Returns:
            np.ndarray: Randomly generated prediction matrix with specified number of 1 per row.
        """
        rows, cols = df.shape
        rand_matrix = np.zeros((rows, cols), dtype=int)

        # Loop through each row
        for i in range(rows):
            indices = np.random.choice(cols, n, replace=False)
            rand_matrix[i, indices] = 1

        rand_pred = pd.DataFrame(rand_matrix, columns=df.columns, index=df.index)

        return rand_pred

    @staticmethod
    def calculate_recommendation_accuracy(df0, df_pred):
        """
        Calculate recommendation accuracy based on two input DataFrames.

        Parameters:
            df0 (DataFrame): Original DataFrame.
            df_pred (DataFrame): DataFrame containing predicted recommendations.

        Returns:
            float: Recommendation accuracy.
        """
        matrix_df0 = df0.values
        matrix_df_pred = df_pred.values
        matrix_sum = matrix_df0 + matrix_df_pred

        result_df = pd.DataFrame(matrix_sum, columns=df0.columns, index=df0.index)
        num_rows_true = result_df.eq(2).any(axis=1).sum()

        recommend_acc = num_rows_true / df0.shape[0]
        return recommend_acc
    
    @staticmethod
    def estimate_precision(df0, df_pred, n):
        """
        Count the total number of occurrences of the value 2 in the DataFrame.

        Parameters:
            df (DataFrame): Input DataFrame.

        Returns:
            int: Total number of occurrences of the value 2.
        """
        matrix_df0 = df0.values
        matrix_df_pred = df_pred.values
        matrix_sum = matrix_df0 + matrix_df_pred

        result_df = pd.DataFrame(matrix_sum, columns=df0.columns, index=df0.index)
        num_rows_2 = result_df.values.ravel().tolist().count(2)
        precision = num_rows_2 / (df0.shape[0]*n)
        return precision

