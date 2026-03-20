from django.shortcuts import render
from .forms import TransactionForm
import joblib
import os
import numpy as np

def index(request):
    return render(request, 'index.html')

def predict(request):
    results = None
    form = TransactionForm()

    if request.method == 'POST':
        form = TransactionForm(request.POST)
        if form.is_valid():
            age = form.cleaned_data['age']
            quantity = form.cleaned_data['quantity']
            price_per_unit = form.cleaned_data['price_per_unit']
            gender = form.cleaned_data['gender']
            product_category = form.cleaned_data['product_category']

            gender_encoded = 0 if gender == 'Male' else 1
            category_map = {'Beauty': 0, 'Clothing': 1, 'Electronics': 2}
            category_encoded = category_map[product_category]

            features = np.array([[age, quantity, price_per_unit, gender_encoded, category_encoded]])

            model_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'trained_models')
            model_files = {
                'Naive Bayes': 'naive_bayes.pkl',
                'Decision Tree': 'decision_tree.pkl',
                'Random Forest': 'random_forest.pkl',
                'Logistic Regression': 'logistic_regression.pkl',
                'SVM': 'svm.pkl'
            }

            results = {}
            for model_name, filename in model_files.items():
                model_path = os.path.join(model_dir, filename)
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None

                    results[model_name] = {
                        'prediction': 'High Spender' if prediction == 1 else 'Low Spender',
                        'confidence': f"{max(probability) * 100:.2f}%" if probability is not None else "N/A"
                    }

    context = {'form': form, 'results': results}
    return render(request, 'predict.html', context)