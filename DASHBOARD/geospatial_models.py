# geospatial_models.py (Versione Completa)

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# --- PyTorch Imports ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Importa TUTTI i tuoi script di modellazione ---
import models as sk_base_models
import models_tuned as sk_tuned_models
import pytorch_models as pt_base_models
import pytorch_models_enhanced as pt_enhanced_models

# --- 1. PRE-PROCESSORE (invariato) ---

def _select_and_ravel_text_column(X_input):
    if isinstance(X_input, pd.DataFrame): series = X_input.iloc[:, 0]
    else: series = pd.Series(X_input[:, 0])
    return pd.Series(series).fillna('').astype(str).values

def create_geospatial_preprocessor(numerical_cols, categorical_cols, text_col):
    """Crea la pipeline di preprocessing per i modelli Scikit-Learn."""
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    text_transformer = Pipeline(steps=[('selector', FunctionTransformer(_select_and_ravel_text_column, validate=False)), ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))])
    
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('text', text_transformer, [text_col])
        ],
        remainder='drop'
    )

# --- 2. FUNZIONI DI ESECUZIONE PER SCIKIT-LEARN ---

def run_sklearn_base_analysis(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols, text_col):
    """Esegue i modelli base di Scikit-Learn."""
    print("\n" + "="*60)
    print(" ESECUZIONE MODELLI SCIKIT-LEARN BASE (con dati geo)")
    print("="*60)
    preprocessor = create_geospatial_preprocessor(numerical_cols, categorical_cols, text_col)
    
    # Regressione
    print("\n--- MODELLI DI REGRESSIONE ---")
    for model_type in ['linear_regression', 'random_forest', 'lightgbm']:
        sk_base_models.train_regression_model(X_train, y_train['reg'], X_test, y_test['reg'], numerical_cols, categorical_cols, text_col, model_type=model_type)

    # Classificazione
    print("\n--- MODELLI DI CLASSIFICAZIONE ---")
    for model_type in ['logistic_regression', 'random_forest', 'lightgbm']:
        sk_base_models.train_classification_model(X_train, y_train['class'], X_test, y_test['class'], numerical_cols, categorical_cols, text_col, model_type=model_type)

def run_sklearn_tuned_analysis(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols, text_col, param_grids):
    """Esegue i modelli con tuning di Scikit-Learn."""
    print("\n" + "="*60)
    print(" ESECUZIONE MODELLI SCIKIT-LEARN CON TUNING (con dati geo)")
    print("="*60)
    
    # Regressione
    print("\n--- MODELLI DI REGRESSIONE (TUNED) ---")
    for model_type in ['ridge', 'random_forest', 'lightgbm']:
        if model_type in param_grids['reg']:
            sk_tuned_models.train_tuned_regression_model(X_train, y_train['reg'], X_test, y_test['reg'], numerical_cols, categorical_cols, text_col, model_type, param_grids['reg'][model_type])

    # Classificazione
    print("\n--- MODELLI DI CLASSIFICAZIONE (TUNED) ---")
    for model_type in ['logistic_regression', 'random_forest', 'lightgbm']:
        if model_type in param_grids['class']:
            sk_tuned_models.train_tuned_classification_model(X_train, y_train['class'], X_test, y_test['class'], numerical_cols, categorical_cols, text_col, model_type, param_grids['class'][model_type])


# --- 3. FUNZIONE DI ESECUZIONE PER PYTORCH ---

def run_pytorch_analysis(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols, text_col, epochs=25, batch_size=64):
    """Esegue i modelli base e potenziati di PyTorch."""
    print("\n" + "="*60)
    print(" ESECUZIONE MODELLI PYTORCH (con dati geo)")
    print("="*60)

    # 1. Preparazione Dati Specifica per PyTorch
    preprocessor = create_geospatial_preprocessor(numerical_cols, categorical_cols, text_col)
    
    # Adatta il preprocessor solo sui dati di training
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Per la classificazione, dobbiamo encodare le etichette in numeri
    le = LabelEncoder()
    y_train_class_enc = le.fit_transform(y_train['class'])
    y_test_class_enc = le.transform(y_test['class'])
    class_names = le.classes_
    num_classes = len(class_names)
    
    # Crea i DataLoader
    # Regressione
    train_loader_reg = DataLoader(pt_base_models.TabularDataset(X_train_processed, y_train['reg'].values), batch_size=batch_size, shuffle=True)
    test_loader_reg = DataLoader(pt_base_models.TabularDataset(X_test_processed, y_test['reg'].values), batch_size=batch_size)
    # Classificazione
    train_loader_class = DataLoader(pt_base_models.TabularDataset(X_train_processed, y_train_class_enc, is_classification=True), batch_size=batch_size, shuffle=True)
    test_loader_class = DataLoader(pt_base_models.TabularDataset(X_test_processed, y_test_class_enc, is_classification=True), batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    input_size = X_train_processed.shape[1]

    # 2. Addestramento e Valutazione
    models_to_run = {
        'reg': {
            'RegressionNN Base': pt_base_models.RegressionNN(input_size),
            'RegressionNN Enhanced': pt_enhanced_models.RegressionNN_v2(input_size)
        },
        'class': {
            'ClassificationNN Base': pt_base_models.ClassificationNN(input_size, num_classes),
            'ClassificationNN Enhanced': pt_enhanced_models.ClassificationNN_v2(input_size, num_classes)
        }
    }

    for task, task_models in models_to_run.items():
        print(f"\n--- MODELLI PYTORCH DI {'CLASSIFICAZIONE' if task == 'class' else 'REGRESSIONE'} ---")
        for model_name, model in task_models.items():
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            if task == 'reg':
                loss_fn = nn.MSELoss()
                train_loader, test_loader = train_loader_reg, test_loader_reg
                eval_func = pt_base_models.evaluate_regression
            else: # task == 'class'
                loss_fn = nn.CrossEntropyLoss()
                train_loader, test_loader = train_loader_class, test_loader_class
                eval_func = pt_base_models.evaluate_classification

            print(f"\nTraining {model_name}...")
            for epoch in range(epochs):
                train_loss = pt_base_models.train_model(model, train_loader, loss_fn, optimizer, device)
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            
            print(f"\nEvaluating {model_name} on test data...")
            if task == 'reg':
                mse, r2 = eval_func(model, test_loader, device)
                print(f"MSE: {mse:.4f}, R2 Score: {r2:.4f}")
            else:
                # Passiamo i nomi delle classi per il report e la matrice di confusione
                eval_func(model, test_loader, device, class_names=class_names, model_name=model_name)
