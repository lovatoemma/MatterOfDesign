import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Caricamento del dataset
dataset_path = "C:/Users/emmal/Desktop/STAGE/PROGETTO/DatiOMI/VERONA_prov/DatiOMI_Unificato.csv"
df = pd.read_csv(dataset_path, sep=";", engine='python', encoding='latin1', on_bad_lines='skip')

# Selezione delle colonne corrette
df_clean = df[["comune_descrizione_x", "zona", "zona_descr", "descr_tipologia", "compr_min", "compr_max", "loc_min", "loc_max", "microzona"]].copy()

# Rinomina colonne per maggiore chiarezza
df_clean.rename(columns={
    "comune_descrizione_x": "comune",
    "descr_tipologia": "tipologia",
    "zona_descr": "zona_descrizione",
}, inplace=True)

# Rimozione valori mancanti nelle colonne essenziali
df_clean.dropna(subset=["comune", "tipologia", "compr_min", "compr_max", "loc_min", "loc_max"], inplace=True)

# Calcolo prezzi medi e ROI
df_clean["prezzo_medio_acquisto"] = (df_clean["compr_min"] + df_clean["compr_max"]) / 2
df_clean["prezzo_medio_affitto"] = (df_clean["loc_min"] + df_clean["loc_max"]) / 2
df_clean["ROI (%)"] = np.where(
    df_clean["prezzo_medio_acquisto"] > 0, 
    (df_clean["prezzo_medio_affitto"] * 12) / df_clean["prezzo_medio_acquisto"] * 100, 
    0
)

# Encoding della tipologia per il modello di Machine Learning
label_encoder = LabelEncoder()
df_clean["tipologia_encoded"] = label_encoder.fit_transform(df_clean["tipologia"])

# Feature per il modello
features = ["prezzo_medio_acquisto", "prezzo_medio_affitto", "microzona"]
target = "tipologia_encoded"

# Suddivisione in training e test set
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Addestramento del modello di classificazione
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# **STREAMLIT DASHBOARD**
st.title("\U0001F3D7️ Dashboard Immobiliare - Provincia di Verona")
st.markdown("Seleziona un comune per ottenere analisi dettagliate sulla redditività e la tipologia immobiliare consigliata.")

# Selezione del comune
comune_selezionato = st.selectbox("\U0001F3E1 Seleziona il comune:", df_clean["comune"].unique())

# Funzione per predire la tipologia immobiliare consigliata
def predici_tipologia(comune):
    df_comune = df_clean[df_clean["comune"] == comune]
    
    if df_comune.empty:
        st.error(f"⚠️ Nessun dato disponibile per il comune: {comune}")
        return None
    
    features_comune = df_comune[features].values
    features_comune_scaled = scaler.transform(features_comune)
    predizione_tipologia = rf_model.predict(features_comune_scaled)
    tipologia_predetta = label_encoder.inverse_transform(predizione_tipologia)[0]
    
    df_comune = df_comune.copy()
    df_comune["Tipologia Predetta"] = tipologia_predetta

    return df_comune

# Calcolo delle metriche per il comune selezionato
risultato = predici_tipologia(comune_selezionato)

if risultato is not None:
    st.subheader("\U0001F4CA Dati del Comune Selezionato")
    st.dataframe(risultato[["comune", "prezzo_medio_acquisto", "prezzo_medio_affitto", "ROI (%)", "Tipologia Predetta"]])

    # Calcolo della redditività
    st.subheader("\U0001F4C8 Redditività stimata")
    st.write(f"**ROI (%) medio per il comune di {comune_selezionato}:**")
    st.write(f"**{round(risultato['ROI (%)'].mean(), 2)}% annuo**")

    # Top comuni per rendimento
    st.subheader("\U0001F4B0 Dove investire per massimizzare il rendimento?")
    top_comuni = df_clean.groupby("comune")["ROI (%)"].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_comuni.plot(kind="bar", ax=ax, color="purple")
    ax.set_title("Top 10 comuni per rendimento previsto")
    ax.set_ylabel("ROI (%)")
    ax.set_xticklabels(top_comuni.index, rotation=45, ha="right")
    st.pyplot(fig)

    # Confronto con altre tipologie immobiliari
    st.subheader("📊 Confronto tra le tipologie immobiliari")
    tipologie_confronto = df_clean.groupby("tipologia")["ROI (%)"].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    tipologie_confronto.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("ROI medio per diverse tipologie immobiliari")
    ax.set_xlabel("Tipologia Immobiliare")
    ax.set_ylabel("ROI (%) medio annuo")
    ax.set_xticklabels(tipologie_confronto.index, rotation=45, ha="right")
    st.pyplot(fig)

# Per avviare la dashboard: 
# streamlit run DashboardPredittiva_Provincia.py
