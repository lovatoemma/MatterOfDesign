import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
#from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report, make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import time

# Caricamento dati
# df_comune = pd.read_csv('DatiOMI_Unificato_comune.csv', delimiter=';')
# Variabili COMUNE: fascia_x, zona, linkzona, cod_tip, descr_tipologia, stato, stato_prev_x,
# compr_min, compr_max, loc_min, loc_max, sup_nl_loc, semestre, 
# fascia_y, zona_descr, cod_tip_prev, descr_tip_prev, stato_prev_y, microzona


df_provincia = pd.read_csv('DatiOMI_Unificato_prov.csv', delimiter=';')
# Variabili PROVINCIA: comune_istat_x, comune_cat_x, comune_amm_x, comune_descrizione_x,
# fascia_x, zona, linkzona, cod_tip, descr_tipologia, stato, stato_prev,
# compr_min, compr_max, loc_min, loc_max, sup_nl_loc, semestre, 
# comune_istat_y, comune_cat_y, comune_amm_y, comune_descrizione_y,
# fascia_y, zona_descr, cod_tip_prev, descr_tip_prev, stato_prev_y, microzona

df = df_provincia.copy()  # Inizialmente lavora solo con df_provincia

# Ispezione e concatenazione gestendo colonne diverse
#print("Colonne df_comune:", df_comune.columns.tolist())
#print("Colonne df_provincia:", df_provincia.columns.tolist())
#common_cols = list(set(df_comune.columns) & set(df_provincia.columns))
#print("\nColonne comuni:", common_cols)
#unique_comune_cols = list(set(df_comune.columns) - set(df_provincia.columns))
#print("Colonne uniche df_comune:", unique_comune_cols)
#unique_provincia_cols = list(set(df_provincia.columns) - set(df_comune.columns))
#print("Colonne uniche df_provincia:", unique_provincia_cols)

#df_comune['livello'] = 'comune'
#df_provincia['livello'] = 'provincia'

#df = pd.concat([df_comune, df_provincia], ignore_index=True, sort=False)

print("\nInfo sul DataFrame combinato:")
df.info()
print("\nValori mancanti per colonna nel DataFrame combinato (prime 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Feature Engineering 
required_cols_range = ['compr_min', 'compr_max']
if all(col in df.columns for col in required_cols_range):
    for col in required_cols_range:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['prezzo_range'] = df['compr_max'] - df['compr_min']
    print(f"Colonna 'prezzo_range' calcolata. NaN presenti: {df['prezzo_range'].isnull().sum()}")
else:
    print(f"Attenzione: Colonne {required_cols_range} non trovate o non numeriche. Impossibile calcolare 'prezzo_range'.")

# --- Preprocessing sul DataFrame combinato ---
required_cols_price = ['compr_min', 'compr_max']
if all(col in df.columns for col in required_cols_price):
    for col in required_cols_price:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['prezzo_medio'] = df[required_cols_price].mean(axis=1)
    print(f"Colonna 'prezzo_medio' calcolata. NaN presenti: {df['prezzo_medio'].isnull().sum()}")
else:
    print(f"Attenzione: Colonne {required_cols_price} non trovate. Impossibile calcolare 'prezzo_medio'.")

feature_cols = ['zona', 'descr_tipologia', 'stato', 'microzona', 'prezzo_medio', 'livello']  # MODIFICATO: descr_tipo -> descr_tipologia
if 'prezzo_range' in df.columns:
    feature_cols.append('prezzo_range')

feature_cols = [col for col in feature_cols if col in df.columns]

if 'prezzo_medio' not in df.columns:
    raise ValueError("La colonna target 'prezzo_medio' non è presente nel DataFrame combinato.")
if 'prezzo_medio' not in feature_cols:
    feature_cols.append('prezzo_medio')

print(f"\nColonne selezionate per il modello: {feature_cols}")
df_model = df[feature_cols].copy()

print(f"\nValori mancanti in df_model prima dell'imputazione:")
print(df_model.isnull().sum())

df_model.dropna(subset=['prezzo_medio'], inplace=True)
print(f"\nRighe dopo dropna su 'prezzo_medio': {len(df_model)}")

numeric_features = df_model.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('prezzo_medio')
categorical_features = df_model.select_dtypes(exclude=np.number).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

if numeric_features:
    df_model[numeric_features] = num_imputer.fit_transform(df_model[numeric_features])
if categorical_features:
    df_model[categorical_features] = cat_imputer.fit_transform(df_model[categorical_features])

print(f"\nValori mancanti in df_model dopo l'imputazione:")
print(df_model.isnull().sum())

df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

X = df_encoded.drop('prezzo_medio', axis=1)
y = df_encoded['prezzo_medio']

if X.empty or y.empty:
    raise ValueError("DataFrame X o y vuoti dopo il preprocessing. Controlla i passaggi.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Pipeline e Hyperparameter Tuning per Regressione ---
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

param_grid_reg = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.1, 0.05],
    'regressor__max_depth': [3, 5]
}

scoring_reg = 'r2'

print("\nInizio Hyperparameter Tuning per Regressione (GridSearchCV)...")
start_time = time.time()
grid_search_reg = GridSearchCV(pipeline_reg, param_grid_reg, cv=3, scoring=scoring_reg, n_jobs=-1)
grid_search_reg.fit(X_train, y_train)
end_time = time.time()
print(f"GridSearchCV per Regressione completato in {end_time - start_time:.2f} secondi.")

print(f"Migliori parametri per Regressione ({scoring_reg}): {grid_search_reg.best_params_}")
print(f"Miglior score CV ({scoring_reg}): {grid_search_reg.best_score_:.4f}")

best_regressor_pipeline = grid_search_reg.best_estimator_
y_pred = best_regressor_pipeline.predict(X_test)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
r2_test = r2_score(y_test, y_pred)
print(f'\nPerformance Regressione su Test Set (Modello Tuned):')
print(f'RMSE: {rmse_test:.4f}')
print(f'R²: {r2_test:.4f}')

joblib.dump(best_regressor_pipeline, 'modello_regressione_pipeline_tuned.pkl')
print("Pipeline di regressione ottimizzata salvata.")

# --- Pipeline e Hyperparameter Tuning per Classificazione ---
target_class_col_name = 'descr_tipologia'
if target_class_col_name in df.columns:  # Lavora con df originale per selezionare feature per classificazione

    # 1. SELEZIONE FEATURE SPECIFICA PER CLASSIFICAZIONE
    classification_feature_candidates = ['zona', 'stato', 'microzona', 'livello', 'sup_nl_loc']  # Aggiungi altre colonne rilevanti da 'df'
    
    valid_classification_features = [col for col in classification_feature_candidates if col in df.columns]
    if not valid_classification_features:
        print(f"Nessuna feature valida trovata per la classificazione. Classificazione saltata.")
    else:
        # QUI INIZIA LA PARTE CRITICA PER IL LEAKAGE
        df_class_model = df[valid_classification_features + [target_class_col_name]].copy()
        df_class_model.dropna(subset=[target_class_col_name], inplace=True)
        
        class_numeric_features = df_class_model.select_dtypes(include=np.number).columns.tolist()
        class_categorical_features = df_class_model.select_dtypes(exclude=np.number).columns.tolist()
        
        # !!! POTENZIALE PUNTO DI LEAKAGE !!!
        # Se target_class_col_name è in class_categorical_features, viene rimosso
        # MA se target_class_col_name è numerico, NON viene rimosso da class_numeric_features
        # e quindi verrebbe imputato e scalato come una feature.
        if target_class_col_name in class_categorical_features:
            class_categorical_features.remove(target_class_col_name)
        # AGGIUNGERE CONTROLLO ANCHE PER NUMERICHE:
        elif target_class_col_name in class_numeric_features: # Aggiunto controllo
            class_numeric_features.remove(target_class_col_name) # Aggiunto remove

        num_imputer_class = SimpleImputer(strategy='median')
        cat_imputer_class = SimpleImputer(strategy='most_frequent')

        if class_numeric_features:
            df_class_model[class_numeric_features] = num_imputer_class.fit_transform(df_class_model[class_numeric_features])
        if class_categorical_features:
            df_class_model[class_categorical_features] = cat_imputer_class.fit_transform(df_class_model[class_categorical_features]) # CORRETTO

        df_class_encoded = pd.get_dummies(df_class_model, columns=class_categorical_features, drop_first=True)

        if target_class_col_name in df_class_encoded.columns:
            X_clf_final = df_class_encoded.drop(target_class_col_name, axis=1)
            y_clf_final = df_class_encoded[target_class_col_name]
        else:
            print(f"ATTENZIONE: La colonna target '{target_class_col_name}' non è stata trovata in df_class_encoded dopo il one-hot encoding. Controllare la logica.")
            cols_to_drop_from_X = [col for col in df_class_encoded.columns if col.startswith(target_class_col_name)]
            if target_class_col_name in df_class_encoded.columns:
                 cols_to_drop_from_X.append(target_class_col_name)
            
            X_clf_final = df_class_encoded.drop(columns=list(set(cols_to_drop_from_X)), errors='ignore')
            y_clf_final = df_class_model[target_class_col_name]

        if X_clf_final.empty or y_clf_final.empty:
            print("Feature o target per classificazione vuoti dopo preprocessing. Classificazione saltata.")
        else:
            print(f"\nDistribuzione delle classi per '{target_class_col_name}' (dopo preprocessing specifico):")
            print(y_clf_final.value_counts(normalize=True) * 100)

            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf_final, y_clf_final, test_size=0.2, random_state=42, stratify=y_clf_final)

            pipeline_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
            ])
            
            param_grid_clf = {
                'classifier__n_estimators': [100, 200, 300, 400],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None]
            }
            scoring_clf = 'accuracy'
            print("\nInizio Hyperparameter Tuning per Classificazione (GridSearchCV con RandomForest)...")
            start_time = time.time()
            grid_search_clf = GridSearchCV(pipeline_clf, param_grid_clf, cv=3, scoring=scoring_clf, n_jobs=-1)
            grid_search_clf.fit(X_train_c, y_train_c)
            end_time = time.time()
            print(f"GridSearchCV per Classificazione completato in {end_time - start_time:.2f} secondi.")

            print(f"Migliori parametri per Classificazione ({scoring_clf}): {grid_search_clf.best_params_}")
            print(f"Miglior score CV ({scoring_clf}): {grid_search_clf.best_score_:.4f}")

            best_classifier_pipeline = grid_search_clf.best_estimator_
            y_pred_class = best_classifier_pipeline.predict(X_test_c)

            print("\nClassification Report (Modello Tuned):")
            unique_labels = np.unique(np.concatenate((y_test_c, y_pred_class)))
            print(classification_report(y_test_c, y_pred_class, labels=unique_labels, zero_division=0))

            joblib.dump(best_classifier_pipeline, 'modello_classificazione_pipeline_tuned.pkl')
            print("Pipeline di classificazione ottimizzata salvata.")
else:
    print(f"Colonna target per classificazione '{target_class_col_name}' non trovata in df.")

# --- Clustering (KMeans) ---
cluster_features_cols = ['prezzo_medio']
microzona_dummies = [col for col in X.columns if 'microzona' in col]
cluster_features_cols.extend(microzona_dummies)
cluster_features_cols = [col for col in cluster_features_cols if col in df_encoded.columns]

if 'prezzo_medio' in df_encoded.columns and len(cluster_features_cols) > 1:
    cluster_data = df_encoded.loc[X.index, cluster_features_cols].copy()

    scaler_cluster = StandardScaler()
    cluster_data_scaled = scaler_cluster.fit_transform(cluster_data)

    n_clusters_optimal = 5
    print(f"\nUtilizzo di k={n_clusters_optimal} per KMeans.")

    kmeans = KMeans(n_clusters=n_clusters_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_scaled)

    df.loc[cluster_data.index, 'zona_cluster'] = cluster_labels

    print("\nHead con Zona Cluster:")
    cols_to_show = ['zona', 'zona_desc', 'zona_cluster', 'livello', 'prezzo_medio']
    cols_to_show = [col for col in cols_to_show if col in df.columns]
    if 'zona_cluster' in df.columns:
        print(df.loc[cluster_data.index, cols_to_show].head())
    else:
        print("Colonna 'zona_cluster' non aggiunta.")

    joblib.dump(kmeans, 'modello_cluster.pkl')
    joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
else:
    print("\nColonne insufficienti o 'prezzo_medio' mancante in df_encoded per il clustering.")

print("\nScript completato. Modelli ottimizzati e scaler salvati.")
