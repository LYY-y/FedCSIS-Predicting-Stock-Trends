import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def check_basic_info(df):
    num_rows = df.shape[0]
    print("Number of rows:", num_rows)
    print(df.describe())

def check_missing_value(df, columns_df, print_detail: False):
    #  non-available data
    rows_without_null = df[~df.isnull().any(axis=1)]
    print(f"Rows_without_missing: {len(rows_without_null)>1}")
    if print_detail:
        missing_values_count = df.isna().sum()
        missing_values_count_gt_zero = missing_values_count[missing_values_count > 0]
        missing_values_count_gt_zero_df = pd.DataFrame(missing_values_count_gt_zero, columns=['count'])
        merged_df = pd.merge(missing_values_count_gt_zero_df, columns_df, left_index=True, right_on='CODE')
        print(merged_df)


def transform_data_type(df, drop_columns=['Group']):
    string_to_float_convert_columns = df.select_dtypes(include=['object']).drop(columns=drop_columns)
    for column in string_to_float_convert_columns:
        # df[column] = df[column].str.replace(',', '.').astype(float)
        df[column] = df[column].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x).astype(float)
    return df


def process_missing_value(df, drop_columns=['Group','Perform']):
    # replace missing value with mean value or 0
    empty_columns = df.columns[df.isna().all()]
    df[empty_columns] = 0
    
    string_to_float_convert_columns = df.select_dtypes(include=['object']).drop(columns=drop_columns)
    for column in string_to_float_convert_columns:
        column_values = df[column] 
        mean_value = column_values.dropna().str.replace(',', '.').astype(float).mean() 
        df[column].fillna(mean_value, inplace=True)
    return df


def error_calculation(y_test, y_pred):
    class_labels = [-1, 0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    # Define the cost matrix
    cost_matrix = np.array([[0, 1, 2],
    [1, 0, 1],
    [2, 1, 0]])

    # Compute the error using the given formula
    err = np.sum(cm * cost_matrix) / len(y_test)
    return err