import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from mlxtend.frequent_patterns import fpgrowth, association_rules
from .data_preprocessing import load_and_preprocess_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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

def perform_clustering():
    df_processed, encoders, numerical_features = load_and_preprocess_data()

    # Prepare data for clustering
    cluster_features = [col for col in numerical_features if col not in ['Total Amount', 'Price per Unit']]
    cluster_features.extend([col for col in df_processed.columns if 'Encoded' in col and 'Spending' not in col and 'Price_Level' not in col])
    cluster_df = df_processed[cluster_features].dropna()

    if cluster_df.empty:
        print("No suitable features for clustering after filtering.")
        return None

    # Determine optimal number of clusters (Elbow Method and Silhouette Score)
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    max_clusters = min(10, len(cluster_df) - 1)  # Ensure max_clusters is less than number of samples

    if max_clusters < 2:
        print("Not enough data points to perform meaningful clustering (need at least 2).")
        return None

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(cluster_df)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(cluster_df, kmeans.labels_)
        silhouette_scores.append(score)

    # Choose optimal K based on highest silhouette score
    optimal_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (based on highest silhouette score): {optimal_k}")

    # Plotting the Elbow Method and Silhouette Scores
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'clustering_elbow_silhouette.png'))
    plt.close()

    # Apply K-Means clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cluster_df)
    df_processed['Cluster'] = clusters

    # Analyze cluster characteristics
    print(f"Cluster Analysis for {optimal_k} Clusters:")
    cluster_summary = df_processed.groupby('Cluster')[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].mean()
    cluster_summary_categorical = df_processed.groupby('Cluster')[['Gender_Encoded', 'Product Category_Encoded']].agg(lambda x: x.mode()[0])  # mode for categorical

    # Inverse transform encoded categorical features for better interpretation
    for col_name, encoder in encoders.items():
        if f'{col_name}_Encoded' in cluster_summary_categorical.columns:
            # Map encoded numerical mode back to original labels
            cluster_summary_categorical[col_name] = cluster_summary_categorical[f'{col_name}_Encoded'].apply(lambda x: encoder.inverse_transform([int(x)])[0])
            cluster_summary_categorical = cluster_summary_categorical.drop(columns=[f'{col_name}_Encoded'])

    print(pd.concat([cluster_summary, cluster_summary_categorical], axis=1))

    # Visualization of clusters (example using first two components)
    if len(cluster_features) >= 2:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=cluster_df.iloc[:, 0], y=cluster_df.iloc[:, 1], hue=df_processed['Cluster'], palette='viridis', legend='full')
        plt.title('Clusters Visualization (First two features)')
        plt.xlabel(cluster_features[0])
        plt.ylabel(cluster_features[1])
        plt.savefig(os.path.join(plots_dir, 'cluster_visualization.png'))
        plt.close()
    else:
        print("Not enough features to create a 2D scatter plot for cluster visualization.")

    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'kmeans_clustering.pkl')
    joblib.dump(kmeans, model_path)

    return optimal_k, cluster_summary, cluster_summary_categorical, model_path

def perform_association_mining():
    df_processed, encoders, _ = load_and_preprocess_data()

    # Association Mining
    # Prepare data for association analysis
    # Group by Transaction ID and collect all Product Categories for each transaction
    transaction_products = df_processed.groupby('Transaction ID')['Product Category'].apply(list).reset_index()

    # One-hot encode the product categories for fpgrowth algorithm
    # First, create a list of all unique products
    all_products = df_processed['Product Category'].unique()

    # Create an empty dataframe with Transaction ID as index and products as columns
    data_for_fpgrowth = pd.DataFrame(index=transaction_products['Transaction ID'], columns=all_products).fillna(False)

    # Populate the dataframe: mark True if a product is in the transaction
    for index, row in transaction_products.iterrows():
        transaction_id = row['Transaction ID']
        products_in_transaction = row['Product Category']
        for product in products_in_transaction:
            data_for_fpgrowth.loc[transaction_id, product] = True

    # Run the FPGrowth algorithm to find frequent itemsets
    frequent_itemsets_fp = fpgrowth(data_for_fpgrowth, min_support=0.01, use_colnames=True)

    # Generate association rules
    rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1)

    # Display the top rules by lift
    print("Top 10 Association Rules (FPGrowth, sorted by Lift):")
    print(rules_fp.sort_values(['lift'], ascending=False).head(10))

    # Interpretation of rules
    print("Interpretation of Association Rules (FPGrowth):")
    if not rules_fp.empty:
        top_rule_fp = rules_fp.sort_values(['lift'], ascending=False).iloc[0]
        antecedents_str_fp = ', '.join(list(top_rule_fp['antecedents']))
        consequents_str_fp = ', '.join(list(top_rule_fp['consequents']))
        print(f"  • Customers who buy {antecedents_str_fp} are {top_rule_fp['lift']:.2f} times more likely to buy {consequents_str_fp}.")
        print(f"    This rule has a confidence of {top_rule_fp['confidence']:.2f}, meaning {top_rule_fp['confidence'] * 100:.0f}% of transactions with {antecedents_str_fp} also contain {consequents_str_fp}.")
        print(f"    Its support is {top_rule_fp['support']:.2f}, indicating it appears in {top_rule_fp['support'] * 100:.0f}% of all transactions.")
    else:
        print("No significant association rules found based on the given support and lift thresholds (FPGrowth).")

    return rules_fp

def perform_regression():
    df_processed, encoders, numerical_features = load_and_preprocess_data()

    if 'Total Amount' not in df_processed.columns:
        print("Total Amount column not found for regression.")
        return None

    # Get original Total Amount before normalization
    # Since we normalized, we need to inverse transform or use original
    # But in preprocessing, we didn't save original, so let's load again
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales_dataset.csv')
    df_original = pd.read_csv(data_path)
    y_reg = df_original['Total Amount'].values

    X_reg_cols = [col for col in numerical_features if col != 'Total Amount']

    encoded_for_reg = [col for col in df_processed.columns if 'Encoded' in col and 'Spending' not in col][:2]

    if len(encoded_for_reg) > 0:
        X_reg = np.column_stack([df_processed[col].values for col in X_reg_cols] +
                                [df_processed[col].values for col in encoded_for_reg])
        all_features = X_reg_cols + encoded_for_reg
    else:
        X_reg = df_processed[X_reg_cols].values
        all_features = X_reg_cols

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train_r, y_train_r)
    y_pred = lr_model.predict(X_test_r)

    r2 = r2_score(y_test_r, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))

    print("Regression Equation:")
    equation = f"Total Amount = {lr_model.intercept_:.2f}"
    for i, feature in enumerate(all_features):
        sign = "+" if lr_model.coef_[i] >= 0 else "-"
        equation += f" {sign} {abs(lr_model.coef_[i]):.2f} × {feature}"
    print(equation)

    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    print(f"Interpretation: {r2*100:.1f}% of variance in Total Amount is explained by the model")

    # Feature impact
    coef_df = pd.DataFrame({
        'feature': all_features,
        'coefficient': lr_model.coef_,
        'abs_coef': np.abs(lr_model.coef_)
    }).sort_values('abs_coef', ascending=False)
    print("Feature Impact:")
    print(coef_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test_r, y_pred, alpha=0.5)
    axes[0].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')

    residuals = y_test_r - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_plots.png')) # type: ignore
    plt.close()

    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'linear_regression.pkl')
    joblib.dump(lr_model, model_path)

    return lr_model, r2, rmse, coef_df, model_path

def train_notebook_classification():
    df_processed, encoders, numerical_features = load_and_preprocess_data()

    # Prepare target
    if 'Spending_Category_Encoded' in df_processed.columns:
        target_col = 'Spending_Category_Encoded'
        target_labels = encoders['Spending_Category'].classes_
    else:
        # Fallback to Spending_Level
        if 'Total Amount' in df_processed.columns:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales_dataset.csv')
            df_original = pd.read_csv(data_path)
            original_amount = df_original['Total Amount']
            median_amount = np.median(original_amount)
            df_processed['Spending_Level'] = pd.cut(original_amount,
                                                    bins=[0, median_amount, float('inf')],
                                                    labels=['Low', 'High'])
            le_target = LabelEncoder()
            df_processed['Spending_Level_Encoded'] = le_target.fit_transform(df_processed['Spending_Level'])
            target_col = 'Spending_Level_Encoded'
            target_labels = le_target.classes_

    # Select features
    feature_cols = []
    for col in numerical_features[:4]:
        if col in df_processed.columns:
            feature_cols.append(col)
    encoded_cols = [col for col in df_processed.columns if 'Encoded' in col and col != target_col]
    feature_cols.extend(encoded_cols[:4])
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df_processed[feature_cols].fillna(0)
    y = df_processed[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    results = {}

    # Naïve Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    print(f"Naïve Bayes Accuracy: {nb_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, nb_pred, target_names=target_labels, zero_division=0))

    # Confusion Matrix Plot for Naïve Bayes
    from sklearn.metrics import confusion_matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_labels, yticklabels=target_labels)
    plt.title('Naïve Bayes Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(plots_dir, 'nb_confusion_matrix.png')) # type: ignore
    plt.close()

    results['Naive Bayes'] = {
        'accuracy': nb_accuracy,
        'model': nb_model,
        'predictions': nb_pred,
        'y_test': y_test
    }

    # Decision Tree
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)

    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, dt_pred, target_names=target_labels, zero_division=0))

    # Confusion Matrix Plot for Decision Tree
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Greens',
                xticklabels=target_labels, yticklabels=target_labels)
    plt.title('Decision Tree Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(plots_dir, 'dt_confusion_matrix.png')) # type: ignore
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Feature Importance:")
    print(feature_importance)

    print(f"Model Comparison:")
    print(f"Naïve Bayes Accuracy: {nb_accuracy:.4f}")
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

    best_model = "Decision Tree" if dt_accuracy >= nb_accuracy else "Naïve Bayes"
    print(f"Best Model: {best_model}")

    # Cross-validation
    try:
        nb_cv = cross_val_score(nb_model, X, y, cv=min(5, len(X)//10), scoring='accuracy')
        dt_cv = cross_val_score(dt_model, X, y, cv=min(5, len(X)//10), scoring='accuracy')
        print(f"Naïve Bayes CV: {nb_cv.mean():.4f} (+/- {nb_cv.std()*2:.4f})")
        print(f"Decision Tree CV: {dt_cv.mean():.4f} (+/- {dt_cv.std()*2:.4f})")
    except:
        pass

    results['Decision Tree'] = {
        'accuracy': dt_accuracy,
        'model': dt_model,
        'predictions': dt_pred,
        'y_test': y_test,
        'feature_importance': feature_importance
    }

    # Save models
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(nb_model, os.path.join(model_dir, 'naive_bayes_notebook.pkl'))
    joblib.dump(dt_model, os.path.join(model_dir, 'decision_tree_notebook.pkl'))

    return results

def run_notebook_analyses():
    print("Running all analyses from the notebook...")

    print("\n=== Clustering Analysis ===")
    clustering_result = perform_clustering()

    print("\n=== Classification Analysis ===")
    classification_result = train_notebook_classification()

    print("\n=== Association Mining ===")
    association_result = perform_association_mining()

    print("\n=== Regression Analysis ===")
    regression_result = perform_regression()

    print("\n=== Model Validation and Testing Summary ===")
    if 'Naive Bayes' in classification_result and 'Decision Tree' in classification_result:
        nb_acc = classification_result['Naive Bayes']['accuracy']
        dt_acc = classification_result['Decision Tree']['accuracy']
        print("Classification Models:")
        print(f"  Naïve Bayes Accuracy: {nb_acc:.4f}")
        print(f"  Decision Tree Accuracy: {dt_acc:.4f}")
        best_class = "Decision Tree" if dt_acc >= nb_acc else "Naïve Bayes"
        print(f"  Conclusion: The {best_class} model performed better for classification.")

    if regression_result:
        lr_model, r2, rmse, coef_df, model_path = regression_result
        print("Regression Model (Linear Regression):")
        print(f"  R-squared: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Conclusion: The Linear Regression model explains {r2*100:.1f}% of the variance in Total Amount.")

    print("\nAll analyses completed successfully!")

if __name__ == '__main__':
    run_notebook_analyses()