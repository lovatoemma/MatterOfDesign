# models_tuned.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Importa il preprocessor dal file models.py
from models import create_preprocessor

def train_tuned_classification_model(X_train, y_train, X_test, y_test,
                                     numerical_cols, categorical_cols, text_col_name,
                                     model_type, param_grid, cv=5, random_state=42,
                                     scoring='f1_weighted'):
    """
    Esegue il tuning di un classificatore e lo valuta sul test set.
    """
    preprocessor = create_preprocessor(numerical_cols, categorical_cols, text_col_name)
    
    if model_type == 'logistic_regression':
        estimator = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)
        print_model_name = "Logistic Regression Classifier (Tuned)"
    elif model_type == 'random_forest':
        estimator = RandomForestClassifier(random_state=random_state)
        print_model_name = "Random Forest Classifier (Tuned)"
    elif model_type == 'lightgbm':
        estimator = lgb.LGBMClassifier(random_state=random_state, verbose=-1, force_col_wise=True)
        print_model_name = "LightGBM Classifier (Tuned)"
    else:
        raise ValueError(f"Modello non supportato: {model_type}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', estimator)])
    pipeline_param_grid = {f"classifier__{key}": value for key, value in param_grid.items()}

    print(f"\n--- Tuning {print_model_name} con GridSearchCV ---")
    grid_search = GridSearchCV(pipeline, pipeline_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"\nMigliori parametri trovati: {grid_search.best_params_}")
    print(f"Miglior punteggio CV ({scoring}): {grid_search.best_score_:.4f}")

    print("\n--- Valutazione del miglior modello sul Test Set ---")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy sul Test Set: {acc:.4f}")
    print("Classification Report sul Test Set:\n", report)
    
    class_names = best_model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.title(f'Matrice di Confusione - {print_model_name} (Test Set)', fontsize=18, pad=20)
    plt.ylabel('Classe Reale', fontsize=14)
    plt.xlabel('Classe Prevista', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()

    return best_model, grid_search.best_params_, grid_search.best_score_

def train_tuned_regression_model(X_train, y_train, X_test, y_test,
                                 numerical_cols, categorical_cols, text_col_name,
                                 model_type, param_grid, cv=5, random_state=42,
                                 scoring='r2'):
    """
    Esegue il tuning di un regressore e lo valuta sul test set.
    """
    preprocessor = create_preprocessor(numerical_cols, categorical_cols, text_col_name)

    if model_type == 'ridge':
        estimator = Ridge(random_state=random_state)
        print_model_name = "Ridge Regression (Tuned)"
    elif model_type == 'random_forest':
        estimator = RandomForestRegressor(random_state=random_state)
        print_model_name = "Random Forest Regressor (Tuned)"
    elif model_type == 'lightgbm':
        estimator = lgb.LGBMRegressor(random_state=random_state, objective='regression', verbose=-1, force_col_wise=True)
        print_model_name = "LightGBM Regressor (Tuned)"
    else:
        raise ValueError(f"Modello non supportato: {model_type}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', estimator)])
    pipeline_param_grid = {f"regressor__{key}": value for key, value in param_grid.items()}
    
    print(f"\n--- Tuning {print_model_name} con GridSearchCV ---")
    grid_search = GridSearchCV(pipeline, pipeline_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Migliori parametri trovati: {grid_search.best_params_}")
    print(f"Miglior punteggio CV ({scoring}): {grid_search.best_score_:.4f}")
    
    # Valutazione finale sul test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n--- Performance del miglior modello sul Test Set ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return best_model, grid_search.best_params_, grid_search.best_score_