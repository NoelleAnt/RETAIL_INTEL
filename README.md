# RETAIL_INTEL

## Description
A Django-based web application that loads and preprocesses retail sales data, trains 5 machine learning models (Naive Bayes, Decision Tree, Random Forest, Logistic Regression, SVM), and provides a user-friendly UI for predicting customer spending levels based on transaction details.

## Features
- Data preprocessing and cleaning
- Training of 5 ML models for classification
- Web UI for inputting transaction details
- Side-by-side comparison of model predictions
- Confidence score chart visualization for model outputs
- Written interpretation/explanation for ensemble and prediction confidence
- Cross-validation and grid search in training to reduce overfitting
- Train/test accuracy monitoring and overfitting risk estimate
- Bootstrap-styled interface for usability

## Prerequisites
- Python 3.8+
- Virtual environment (venv)
- Git (for cloning the repo)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/NoelleAnt/RETAIL_INTEL.git
   cd RETAIL_INTEL
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # On Windows PowerShell
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place the retail sales dataset (`retail_sales_dataset.csv`) in the `data/` folder.

## Usage
1. Train the models:
   ```
   python manage.py shell
   >>> from ml_models.model_trainer import train_all_models
   >>> train_all_models()
   >>> exit()
   ```

2. Run the Django server:
   ```
   python manage.py runserver
   ```

3. Access the app at `http://127.0.0.1:8000/`.

5. Enter transaction details in the form and view predictions from all 5 models.
   - A confidence bar chart is shown for model score comparison.
   - A written interpretation provides ensemble consensus and confidence guidance.

## Contributing
Fork the repository and submit pull requests for improvements.

## License
This project is licensed under the MIT License.