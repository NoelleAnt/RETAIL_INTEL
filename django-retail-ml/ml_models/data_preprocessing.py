import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

def load_and_preprocess_data():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales_dataset.csv')
    df = pd.read_csv(data_path)
    
    df_processed = df.copy()
    
    # Handle missing values
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df_processed.drop_duplicates(inplace=True)
    
    # Handle negative values
    for col in numerical_cols:
        if (df_processed[col] < 0).any():
            df_processed[col] = df_processed[col].abs()
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        if col not in ['Transaction ID', 'Customer ID', 'Date']:
            le = LabelEncoder()
            df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
    
    # Normalize numerical features
    numerical_features = [col for col in numerical_cols if col not in ['Transaction ID']]
    scaler = MinMaxScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    return df_processed, encoders, numerical_features