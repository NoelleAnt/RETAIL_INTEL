from django.shortcuts import render
from .forms import TransactionForm
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def index(request):
    return render(request, 'index.html')


def predict(request):
    results = None
    chart_data = None
    interpretation_text = None
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

            # Keep dataframe column names to match model training feature names, preventing sklearn warning
            features = pd.DataFrame(features, columns=[
                'Age', 'Quantity', 'Price per Unit', 'Gender_Encoded', 'Product Category_Encoded'
            ])

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
                    confidence_score = float(max(probability)) if probability is not None else None

                    results[model_name] = {
                        'prediction': 'High Spender' if prediction == 1 else 'Low Spender',
                        'confidence': f"{confidence_score * 100:.2f}%" if confidence_score is not None else "N/A",
                        'confidence_value': confidence_score * 100 if confidence_score is not None else 0.0
                    }

            if results:
                labels = list(results.keys())
                scores = [results[m]['confidence_value'] for m in labels]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(labels, scores, color='tab:blue')
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Model Confidence per Prediction')
                ax.set_xbound(0, 100)

                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va='center', ha='left', color='black')

                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                chart_data = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

                high_votes = sum(1 for v in results.values() if v['prediction'] == 'High Spender')
                low_votes = sum(1 for v in results.values() if v['prediction'] == 'Low Spender')
                average_confidence = sum(scores) / len(scores) if scores else 0

                if high_votes > low_votes:
                    majority = 'High Spender'
                    vote_strength = high_votes
                elif low_votes > high_votes:
                    majority = 'Low Spender'
                    vote_strength = low_votes
                else:
                    majority = 'Mixed outcomes'
                    vote_strength = high_votes

                interpretation_text = (
                    f"Ensemble trend: {majority} ({vote_strength} of {len(results)} models agree). "
                    f"Average confidence: {average_confidence:.1f}%. "
                )

                if average_confidence >= 75:
                    interpretation_text += "Strong consensus; prediction is likely reliable. "
                elif average_confidence >= 55:
                    interpretation_text += "Moderate consensus; proceed with caution and validate using real transactions. "
                else:
                    interpretation_text += "Low confidence; treat results as exploratory and gather more data. "

                interpretation_text += (
                    f"Input profile: age={age}, quantity={quantity}, price_per_unit={price_per_unit}, "
                    f"gender={gender}, category={product_category}."
                )

    context = {
        'form': form,
        'results': results,
        'chart_data': chart_data,
        'interpretation_text': interpretation_text
    }
    return render(request, 'predict.html', context)