import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from .data_preprocessing import load_and_preprocess_data
import numpy as np

def train_all_models():
    # Load and preprocess data
    df_processed, encoders, numerical_features = load_and_preprocess_data()
    
    # Prepare target variable
    if 'Total Amount' in df_processed.columns:
        original_amount = df_processed['Total Amount']
        median_amount = np.median(original_amount)
        df_processed['Spending_Level'] = (original_amount > median_amount).astype(int)
        y = df_processed['Spending_Level']
    
    # Select features
    feature_cols = [col for col in numerical_features if col != 'Total Amount']
    encoded_cols = [col for col in df_processed.columns if 'Encoded' in col]
    feature_cols.extend(encoded_cols[:3])
    feature_cols = list(dict.fromkeys(feature_cols))
    
    X = df_processed[feature_cols].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train 5 models
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        # Save model
        model_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        joblib.dump(model, model_path)
        
        results[model_name] = {
            'accuracy': accuracy,
            'model_path': model_path
        }
    
    return results, feature_cols, X_test, y_test

if __name__ == '__main__':
    train_all_models()