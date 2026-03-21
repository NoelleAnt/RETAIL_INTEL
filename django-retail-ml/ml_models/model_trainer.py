import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .data_preprocessing import load_and_preprocess_data

def train_all_models():
    df_processed, encoders, numerical_features = load_and_preprocess_data()

    if 'Total Amount' in df_processed.columns:
        original_amount = df_processed['Total Amount']
        median_amount = np.median(original_amount)
        df_processed['Spending_Level'] = (original_amount > median_amount).astype(int)
        y = df_processed['Spending_Level']
    else:
        raise ValueError('Total Amount column not found')

    feature_cols = [col for col in numerical_features if col != 'Total Amount']
    encoded_cols = [col for col in df_processed.columns if 'Encoded' in col]
    feature_cols.extend(encoded_cols[:3])
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df_processed[feature_cols].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }

    param_grids = {
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'Random Forest': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear'],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
        },
    }

    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)

    results = {}
    for model_name, model in models.items():
        if model_name in param_grids:
            grid = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            cv_mean = grid.best_score_
            cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
            tuned_params = grid.best_params_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, n_jobs=-1)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            tuned_params = {}

        train_acc = best_model.score(X_train, y_train)
        test_acc = best_model.score(X_test, y_test)

        model_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        joblib.dump(best_model, model_path)

        overfit_risk = max(0.0, train_acc - test_acc)
        results[model_name] = {
            'model_path': model_path,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_accuracy_mean': cv_mean,
            'cv_accuracy_std': cv_std,
            'tuned_params': tuned_params,
            'overfit_risk': overfit_risk,
        }

    print("\n===== Model evaluation summary =====")
    for model_name, stats in results.items():
        gap = stats['overfit_risk']
        if gap <= 0.05:
            recommendation = 'OK: good generalization.'
        elif gap <= 0.15:
            recommendation = 'Warning: moderate overfit; tune or regularize.'
        else:
            recommendation = 'Strong overfit risk: reduce complexity / add data.'

        print(f"{model_name}: train={stats['train_accuracy']:.3f}, "
              f"test={stats['test_accuracy']:.3f}, gap={gap:.3f} -> {recommendation}")
    print("====================================\n")

    return results, feature_cols, X_test, y_test

if __name__ == '__main__':
    train_all_models()