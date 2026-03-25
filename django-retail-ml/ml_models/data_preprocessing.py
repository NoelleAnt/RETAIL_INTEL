import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales_dataset.csv')
    df = pd.read_csv(data_path)
    df_processed = df.copy()

    # Handle missing values
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df_processed.loc[:, col] = df_processed[col].fillna(df_processed[col].median())

    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            df_processed.loc[:, col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # Data cleaning
    df_processed = df_processed.drop_duplicates()
    for col in numerical_cols:
        if (df_processed[col] < 0).any():
            df_processed.loc[:, col] = df_processed[col].abs()

    # Convert Date to datetime
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])

    # Data discretization
    if 'Age' in df_processed.columns:
        df_processed['Age_Group'] = pd.cut(df_processed['Age'],
                                            bins=[0, 25, 35, 50, 100],
                                            labels=['Young', 'Young Adult', 'Middle Age', 'Senior'])
    if 'Total Amount' in df_processed.columns:
        df_processed['Spending_Category'] = pd.qcut(df_processed['Total Amount'],
                                                     q=3,
                                                     labels=['Low', 'Medium', 'High'])
    if 'Price per Unit' in df_processed.columns:
        df_processed['Price_Level'] = pd.cut(df_processed['Price per Unit'],
                                              bins=[0, 50, 200, 1000],
                                              labels=['Budget', 'Standard', 'Premium'])

    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        if col not in ['Transaction ID', 'Customer ID', 'Date']:
            le = LabelEncoder()
            df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le

    # Also encode the new discretized columns
    discretized_cols = ['Age_Group', 'Spending_Category', 'Price_Level']
    for col in discretized_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le

    # Normalize numerical features
    numerical_features = [col for col in numerical_cols if col not in ['Transaction ID']]
    scaler = MinMaxScaler()
    if numerical_features:
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    return df_processed, encoders, numerical_features