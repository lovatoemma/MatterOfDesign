import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# SOSTITUISCO LA VECCHIA FUNZIONE CON QUESTA
def _select_and_ravel_text_column(X_input):
    """
    Gestisce input di tipo DataFrame, Serie o array NumPy, 
    restituendo un array 1D di stringhe.
    """
    # Se l'input è un DataFrame o una Serie pandas
    if isinstance(X_input, (pd.DataFrame, pd.Series)):
        if isinstance(X_input, pd.DataFrame):
            series = X_input.iloc[:, 0]
        else:
            series = X_input
    # Se l'input è un array NumPy
    elif isinstance(X_input, np.ndarray):
        series = pd.Series(X_input.ravel())
    else:
        raise TypeError(f"Tipo di input non gestito per la colonna di testo: {type(X_input)}")
    
    # Restituisce i valori come stringhe, gestendo i NaN
    return series.fillna('').astype(str).values

def create_preprocessor(numerical_cols, categorical_cols, text_col_name=None, **kwargs):
    transformers_list = []
    if numerical_cols:
        transformers_list.append(('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols))
    if categorical_cols:
        transformers_list.append(('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols))
    if text_col_name:
        transformers_list.append(('text', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='')), ('selector', FunctionTransformer(_select_and_ravel_text_column, validate=False)), ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))]), [text_col_name]))
    return ColumnTransformer(transformers=transformers_list, remainder='drop')

def train_classification_model(X_train, y_train, X_test, y_test,
                               numerical_cols, categorical_cols, text_col_name=None,
                               model_type='random_forest', model_params=None, random_state=42):
    current_preprocessor = create_preprocessor(numerical_cols, categorical_cols, text_col_name)
    effective_model_params = model_params or {}

    if model_type == 'logistic_regression':
        estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', random_state=random_state, **effective_model_params)
        print_model_name = "Logistic Regression Classifier"
    elif model_type == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, **effective_model_params)
        print_model_name = "Random Forest Classifier"
    elif model_type == 'lightgbm':
        estimator = lgb.LGBMClassifier(random_state=random_state, verbose=-1, force_col_wise=True, **effective_model_params)
        print_model_name = "LightGBM Classifier"
    else:
        raise ValueError(f"Modello non supportato: {model_type}")

    pipeline = Pipeline(steps=[('preprocessor', current_preprocessor), ('classifier', estimator)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n--- {print_model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # --- CODICE PER LA MATRICE DI CONFUSIONE SPOSTATO QUI ---
    class_names = pipeline.classes_
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title(f'Matrice di Confusione - {print_model_name}', fontsize=18, pad=20)
    plt.ylabel('Classe Reale', fontsize=14)
    plt.xlabel('Classe Prevista', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()
    # --- FINE CODICE SPOSTATO ---

    return pipeline, acc, report

# La funzione di regressione rimane invariata
def train_regression_model(X_train, y_train, X_test, y_test,
                           numerical_cols, categorical_cols, text_col_name=None,
                           model_type='random_forest', model_params=None, random_state=42):
    current_preprocessor = create_preprocessor(numerical_cols, categorical_cols, text_col_name)
    effective_model_params = model_params or {}

    if model_type == 'linear_regression':
        estimator = LinearRegression(**effective_model_params)
        print_model_name = "Linear Regression"
    elif model_type == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, **effective_model_params)
        print_model_name = "Random Forest Regressor"
    elif model_type == 'lightgbm':
        estimator = lgb.LGBMRegressor(objective='regression', random_state=random_state, verbose=-1, force_col_wise=True, **effective_model_params)
        print_model_name = "LightGBM Regressor"
    else:
        raise ValueError(f"Modello non supportato: {model_type}")

    pipeline = Pipeline(steps=[('preprocessor', current_preprocessor), ('regressor', estimator)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {print_model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return pipeline, mse, r2
