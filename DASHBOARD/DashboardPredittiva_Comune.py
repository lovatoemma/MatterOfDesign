import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#  Caricamento del dataset
dataset_path = "C:/Users/emmal/Desktop/STAGE/PROGETTO/DatiOMI/VERONA_comune/DatiOMI_Unificato.csv"
df_unificato = pd.read_csv(dataset_path, sep=";", engine='python', encoding='latin1', on_bad_lines='skip')

#  Selezione delle feature
features = ["compr_min", "compr_max", "loc_min", "loc_max", "microzona"]
target = "descr_tipologia"

#  Pulizia del dataset
df_unificato.dropna(subset=features + [target], inplace=True)

#  Encoding della variabile target
label_encoder = LabelEncoder()
df_unificato[target] = label_encoder.fit_transform(df_unificato[target])

#  Separazione delle feature e del target
X = df_unificato[features]
y = df_unificato[target]

#  Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Standardizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Creazione e addestramento del modello
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#  Dashboard con Streamlit
st.title("🏗️ Dashboard MOD - Predizione e Redditività Immobiliare, comune di Verona")
st.markdown("Seleziona una zona per ottenere la tipologia consigliata e analizzare la redditività dell’investimento.")

#  Spiegazione del ROI
st.subheader("📊 Cos'è il ROI?")
st.markdown("""
**ROI (Return on Investment)** è un indice che misura il rendimento di un investimento immobiliare.
Esso è calcolato come il rapporto tra il guadagno ottenuto e il costo dell'investimento.

- **Un ROI alto** significa che l'investimento è redditizio.
- **Un ROI basso** indica che l'investimento potrebbe non essere vantaggioso.
""")

#  Selezione della zona
zona_selezionata = st.selectbox("📍 Seleziona la zona:", df_unificato["zona"].unique())

#  Funzione per predire la tipologia consigliata
def predici_tipologia_zona(zona_id):
    zona_esistente = df_unificato[df_unificato["zona"] == zona_id]
    
    if zona_esistente.empty:
        st.error(f"⚠️ Nessuna zona trovata con ID: {zona_id}")
        return None
    
    zona_features = zona_esistente[features].values
    zona_features_scaled = scaler.transform(zona_features)
    predizione_tipologia = rf_model.predict(zona_features_scaled)
    tipologia_predetta = label_encoder.inverse_transform(predizione_tipologia)[0]
    
    zona_esistente = zona_esistente.copy()
    zona_esistente["Tipologia Predetta"] = tipologia_predetta

    return zona_esistente

#  Calcolo della Redditività (ROI)
df_unificato["Prezzo €/m²"] = (df_unificato["compr_min"] + df_unificato["compr_max"]) / 2
df_unificato["Affitto €/m²"] = (df_unificato["loc_min"] + df_unificato["loc_max"]) / 2
df_unificato["ROI (%) Predetto"] = (df_unificato["Affitto €/m²"] * 12) / df_unificato["Prezzo €/m²"] * 100

#  Ottieni la previsione per la zona scelta
risultato_previsione = predici_tipologia_zona(zona_selezionata)

if risultato_previsione is not None:
    st.subheader("📊 Dati della Zona Selezionata")
    st.dataframe(risultato_previsione[["zona", "zona_descr", "compr_min", "compr_max", "loc_min", "loc_max", "microzona", "Tipologia Predetta"]])

    #  Calcolo redditività
    risultato_previsione["ROI (%)"] = (risultato_previsione["loc_max"] * 12) / risultato_previsione["compr_max"] * 100

    st.subheader("📈 Redditività stimata")
    st.write(f" **ROI (%) medio per la tipologia consigliata ({risultato_previsione['Tipologia Predetta'].values[0]}):**")
    st.write(f" **{round(risultato_previsione['ROI (%)'].values[0], 2)}% annuo**")

    #  Dove Investire per Massimizzare il Rendimento
    st.subheader("💰 Dove investire per massimizzare il rendimento?")

    if "zona_descr" in df_unificato.columns:
        top_zone = df_unificato.groupby(["zona", "zona_descr"])["ROI (%) Predetto"].mean().sort_values(ascending=False).head(10)
        zone_labels = top_zone.index.get_level_values("zona_descr")
    else:
        top_zone = df_unificato.groupby("zona")["ROI (%) Predetto"].mean().sort_values(ascending=False).head(10)
        zone_labels = top_zone.index  

    fig, ax = plt.subplots(figsize=(10, 6))
    top_zone.plot(kind="bar", ax=ax, color="purple")
    ax.set_title("Top 10 zone per rendimento previsto")
    ax.set_ylabel("ROI (%)")
    ax.set_xticklabels(zone_labels, rotation=45, ha="right")
    st.pyplot(fig)

    #  Confronto con altre tipologie immobiliari
    st.subheader("📊 Confronto con altre tipologie immobiliari")

    tipologie_confronto = df_unificato.groupby(df_unificato[target])["ROI (%) Predetto"].mean().sort_values(ascending=False)
    tipologie_confronto.index = label_encoder.inverse_transform(tipologie_confronto.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    tipologie_confronto.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("ROI medio per diverse tipologie immobiliari")
    ax.set_xlabel("Tipologia Immobiliare")
    ax.set_ylabel("ROI (%) medio annuo")
    ax.set_xticklabels(tipologie_confronto.index, rotation=45, ha="right")
    st.pyplot(fig)

    #  Prezzi medi per tipologia immobiliare
    st.subheader("📊 Prezzi medi per tipologia immobiliare")

    df_unificato["descr_tipologia_decodificata"] = label_encoder.inverse_transform(df_unificato["descr_tipologia"])
    df_prezzi = df_unificato.groupby("descr_tipologia_decodificata")[["compr_min", "compr_max"]].mean()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    df_prezzi.plot(kind="bar", ax=ax3, color=["lightblue", "darkblue"])
    ax3.set_title("Prezzo minimo e massimo per tipologia immobiliare")
    ax3.set_ylabel("Prezzo (€/mq)")
    ax3.set_xticklabels(df_prezzi.index, rotation=45, ha="right")
    st.pyplot(fig3)


# streamlit run DashboardPredittiva.py