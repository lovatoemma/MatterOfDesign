import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import os # <<< MODIFICA: Importa la libreria OS per gestire i percorsi
import numpy as np

# ===================================================================
# CONFIGURAZIONE PAGINA
# ===================================================================
st.set_page_config(
    page_title="MOD Analisi Investimenti",
    page_icon="💡",
    layout="wide"
)

# <<< MODIFICA: Crea un percorso base che funziona ovunque (locale e cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    """Carica TUTTI gli artefatti del modello usando percorsi robusti."""
    try:
        input_size = 215
        num_classes = 5
        model = ClassificationNN_v2(input_size=input_size, num_classes=num_classes)
        
        # <<< MODIFICA: Aggiunge il percorso base a ogni file
        model_path = os.path.join(BASE_DIR, 'pytorch_model_state.pth')
        preprocessor_path = os.path.join(BASE_DIR, 'preprocessor_torch.joblib')
        encoder_path = os.path.join(BASE_DIR, 'label_encoder_pers.joblib')
        columns_path = os.path.join(BASE_DIR, 'model_columns.joblib')

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        preprocessor = joblib.load(preprocessor_path)
        label_encoder = joblib.load(encoder_path)
        model_columns = joblib.load(columns_path)
        
        return model, preprocessor, label_encoder, model_columns
    except Exception as e:
        st.error(f"⚠️ Errore caricamento artefatti: {e}. Controlla che i file siano nella stessa cartella dello script su GitHub.")
        return None, None, None, None

@st.cache_data
def load_main_data():
    """Carica il singolo file CSV pre-processato."""
    try:
        # <<< MODIFICA: Aggiunge il percorso base al file CSV
        csv_path = os.path.join(BASE_DIR, 'DatiCompletiPuliti.csv')
        df = pd.read_csv(csv_path, sep=';', encoding='latin1')
        df.rename(columns={'comune_descrizione_val': 'comune_descrizione'}, inplace=True, errors='ignore')
        return df
    except FileNotFoundError:
        st.error(f"⚠️ File 'DatiCompletiPuliti.csv' non trovato. Assicurati che sia stato caricato su GitHub nella cartella 'DASHBOARD'.")
        return None

# Caricamento di tutto il necessario
model, preprocessor, label_encoder, model_columns = load_model_artifacts()
main_df = load_main_data()

# ===================================================================
# IL RESTO DELLO SCRIPT RIMANE IDENTICO...
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
    st.sidebar.error("Applicazione non inizializzata. Controllare gli errori nel pannello principale.")
    analyze_button = False

st.title("💡 Dashboard di Analisi Strategica per Investimenti Immobiliari")
st.markdown("---")

if analyze_button:
    # (Il resto del codice rimane invariato, non serve copiarlo di nuovo)
    with st.spinner('Il motore sta simulando diversi scenari di investimento...'):
        risultati = []
        tipologie_da_testare = ['Abitazioni civili', 'Ville e Villini', 'Negozi', 'Uffici', 'Box', 'Magazzini', 'Capannoni tipici']

        default_values = {col: 0 for col in model_columns if np.issubdtype(main_df.get(col, pd.Series(0)).dtype, np.number)}
        for col in model_columns:
            if col not in default_values:
                default_values[col] = ''

        for tipologia_test in tipologie_da_testare:
            input_data = pd.DataFrame([default_values], columns=model_columns)
            
            input_data['descr_tipologia'] = tipologia_test
            input_data['comune_descrizione'] = comune_selezionato
            input_data['zona_descr'] = zona_selezionata
            input_data['stato'] = 'OTTIMO'
            input_data['sup_tot_imm'] = sup_tot_imm
            input_data['sup_nl_loc'] = sup_tot_imm * 0.9
            
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