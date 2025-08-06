import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import os
import numpy as np

# ===================================================================
# CONFIGURAZIONE PAGINA
# ===================================================================
st.set_page_config(
    page_title="MOD Analisi Investimenti",
    page_icon="💡",
    layout="wide"
)

# ===================================================================
# DEFINIZIONE DEL MODELLO
# ===================================================================
class ClassificationNN_v2(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(ClassificationNN_v2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# ===================================================================
# FUNZIONI DI CARICAMENTO
# ===================================================================
@st.cache_resource
def load_model_artifacts():
    """Carica TUTTI gli artefatti del modello, incluso il contratto delle colonne."""
    try:
        input_size = 215
        num_classes = 5
        model = ClassificationNN_v2(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load('pytorch_model_state.pth', map_location=torch.device('cpu')))
        model.eval()
        preprocessor = joblib.load('preprocessor_torch.joblib')
        label_encoder = joblib.load('label_encoder_pers.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, preprocessor, label_encoder, model_columns
    except Exception as e:
        st.error(f"⚠️ Errore caricamento artefatti: {e}. Assicurati che 'model_columns.joblib' e gli altri file del modello siano presenti.")
        return None, None, None, None

@st.cache_data
def load_main_data():
    """Carica il singolo file CSV pre-processato."""
    try:
        # Assicurati che il nome del file corrisponda a quello che hai salvato dal notebook
        df = pd.read_csv('DatiCompletiPuliti.csv', sep=';', encoding='latin1')
        df.rename(columns={'comune_descrizione_val': 'comune_descrizione'}, inplace=True, errors='ignore')
        return df
    except FileNotFoundError:
        st.error("⚠️ File 'DatiCompletiPuliti.csv' non trovato. Assicurati che sia nella stessa cartella della dashboard.")
        return None

# Caricamento di tutto il necessario
model, preprocessor, label_encoder, model_columns = load_model_artifacts()
main_df = load_main_data()

# ===================================================================
# INTERFACCIA UTENTE (SIDEBAR)
# ===================================================================
st.sidebar.title("Parametri di Analisi")
st.sidebar.info("Seleziona una zona per scoprire la tipologia di immobile con il più alto potenziale di investimento.")

if main_df is not None and model is not None:
    lista_comuni = sorted(main_df['comune_descrizione'].unique())
    with st.sidebar:
        st.header("📍 Ubicazione")
        comune_selezionato = st.selectbox('Comune:', lista_comuni, index=lista_comuni.index("VERONA") if "VERONA" in lista_comuni else 0)
        
        lista_zone = sorted(main_df[main_df['comune_descrizione'] == comune_selezionato]['zona_descr'].unique())
        zona_selezionata = st.selectbox('Descrizione Zona OMI:', lista_zone)
        
        st.header("🏢 Caratteristiche Immobile Ipotetico")
        sup_tot_imm = st.slider('Superficie (mq):', 50, 1000, 120)
        
        analyze_button = st.button('Analizza Potenziale di Investimento', type="primary", use_container_width=True)
else:
    st.sidebar.error("Applicazione non inizializzata. Controllare i file mancanti.")
    analyze_button = False

# ===================================================================
# PAGINA PRINCIPALE E LOGICA DI PREVISIONE
# ===================================================================
st.title("💡 Dashboard di Analisi Strategica per Investimenti Immobiliari")
st.markdown("---")

if analyze_button:
    with st.spinner('Il motore sta simulando diversi scenari di investimento...'):
        risultati = []
        tipologie_da_testare = ['Abitazioni civili', 'Ville e Villini', 'Negozi', 'Uffici', 'Box', 'Magazzini', 'Capannoni tipici']

        # Prepara un dizionario di valori di default plausibili basato sulle colonne del modello
        default_values = {col: 0 for col in model_columns if np.issubdtype(main_df.get(col, pd.Series(0)).dtype, np.number)}
        for col in model_columns:
            if col not in default_values:
                default_values[col] = ''

        for tipologia_test in tipologie_da_testare:
            # Crea un DataFrame di input che rispetta SEMPRE il contratto delle colonne
            input_data = pd.DataFrame([default_values], columns=model_columns)
            
            # Popola il DataFrame con i valori della simulazione corrente
            input_data['descr_tipologia'] = tipologia_test
            input_data['comune_descrizione'] = comune_selezionato
            input_data['zona_descr'] = zona_selezionata
            input_data['stato'] = 'OTTIMO'
            input_data['sup_tot_imm'] = sup_tot_imm
            input_data['sup_nl_loc'] = sup_tot_imm * 0.9
            
            # Popola le altre colonne con valori di riferimento presi dalla zona, se disponibili
            template_row = main_df[(main_df['comune_descrizione'] == comune_selezionato) & (main_df['zona_descr'] == zona_selezionata)]
            if not template_row.empty:
                for col in input_data.columns:
                    if col in template_row.columns:
                        input_data[col] = template_row[col].iloc[0]

            try:
                processed_input = preprocessor.transform(input_data)
                input_tensor = torch.FloatTensor(processed_input)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1).numpy().flatten()
                risultati.append({'Tipologia Testata': tipologia_test, 'Probabilità': probabilities})
            except Exception as e:
                st.warning(f"Impossibile analizzare '{tipologia_test}'. Errore: {e}", icon="⚠️")
        
    # --- VISUALIZZAZIONE RISULTATI ---
    if risultati:
        st.header(f"📈 Risultati per: {zona_selezionata} ({comune_selezionato})")
        df_risultati = pd.DataFrame()
        for res in risultati:
            temp_df = pd.DataFrame([res['Probabilità']], columns=label_encoder.classes_, index=[res['Tipologia Testata']])
            df_risultati = pd.concat([df_risultati, temp_df])

        classi_desiderabili = ['Abitazioni civili', 'Ville e Villini']
        df_risultati['Punteggio Investimento'] = df_risultati.get(classi_desiderabili, pd.DataFrame(0, index=df_risultati.index, columns=classi_desiderabili)).sum(axis=1)
        df_risultati = df_risultati.sort_values('Punteggio Investimento', ascending=False)
        miglior_investimento = df_risultati.index[0]

        st.subheader("🏆 Raccomandazione del Modello")
        st.success(f"Per la zona selezionata, l'investimento con il più alto potenziale è: **{miglior_investimento}**.")
        st.subheader("Confronto Potenziale per Tipologia")
        st.bar_chart(df_risultati.sort_values('Punteggio Investimento', ascending=True)[list(label_encoder.classes_)])
        
        with st.expander("Dettaglio Punteggio e Probabilità"):
            st.dataframe(df_risultati[['Punteggio Investimento'] + list(label_encoder.classes_)].style.format("{:.2%}"))

else:
    if main_df is not None:
        st.info("👈 **Utilizza il pannello a sinistra** per avviare l'analisi strategica.")      