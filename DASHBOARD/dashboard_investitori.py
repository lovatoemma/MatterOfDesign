import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time

# ===================================================================
# CONFIGURAZIONE PAGINA
# ===================================================================
st.set_page_config(
    page_title="MOD Valutazione Immobili",
    page_icon="🏠",
    layout="wide"
)

# ===================================================================
# DEFINIZIONE DEL MODELLO
# Questa classe deve essere identica a quella usata per l'addestramento.
# ===================================================================
class ClassificationNN_v2(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(ClassificationNN_v2, self).__init__()
        self.network = nn.Sequential(
            # --- Correzione qui: 128 -> 256 ---
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Correzione qui: 64 -> 128 ---
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Correzione qui: input da 128 ---
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ===================================================================
# FUNZIONE DI CARICAMENTO MODELLO
# ===================================================================
@st.cache_resource
def load_model_artifacts():
    """Carica il modello, il preprocessor e il label encoder."""
    input_size = 215  # Dimensione dell'input, deve corrispondere a quella usata in addestramento
    num_classes = 5   # 5 categorie aggregate
    
    model = ClassificationNN_v2(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load('pytorch_model_state.pth', map_location=torch.device('cpu')))
    model.eval()
    
    preprocessor = joblib.load('preprocessor_torch.joblib')
    label_encoder = joblib.load('label_encoder_pers.joblib')
    
    return model, preprocessor, label_encoder

# Caricamento degli artefatti
try:
    model, preprocessor, label_encoder = load_model_artifacts()
except Exception as e:
    st.error(f"Errore durante il caricamento del modello: {e}")
    st.stop()

# ===================================================================
# INTERFACCIA UTENTE (SIDEBAR)
# ===================================================================
st.sidebar.title("Parametri Immobile")
st.sidebar.info("Inserire i dati per avviare la classificazione del motore di valutazione.")

# Mapping tipologie
mapping_tipologie = {
    'Abitazioni civili': 'Abitazioni civili', 'Ville e Villini': 'Ville e Villini',
    'Uffici': 'Uffici/Negozi/Economico', 'Negozi': 'Uffici/Negozi/Economico',
    'Abitazioni di tipo economico': 'Uffici/Negozi/Economico', 'Capannoni tipici': 'Industriale',
    'Capannoni industriali': 'Industriale', 'Box': 'Box/Magazzini/Laboratori',
    'Magazzini': 'Box/Magazzini/Laboratori', 'Laboratori': 'Box/Magazzini/Laboratori'
}

with st.sidebar:
    st.header("Dati Principali")
    descr_tipologia = st.selectbox('Tipologia Immobile:', options=list(mapping_tipologie.keys()))
    sup_tot_imm = st.number_input('Superficie Totale (mq):', value=100, min_value=1)
    stato = st.selectbox('Stato di Conservazione:', ['OTTIMO', 'NORMALE', 'SCADENTE'])
    semestre = st.number_input('Semestre di Riferimento (es. 20241):', value=20241, format="%d")

    st.header("Ubicazione")
    comune_descrizione = st.text_input('Comune:', 'VERONA')
    zona_descr = st.text_input('Descrizione Zona OMI:', 'CENTRO STORICO')
    distanza_centro_m = st.number_input('Distanza dal Centro (metri):', value=1000, min_value=0)

    st.header("Identificativi e Dettagli")
    microzona = st.text_input('Codice Microzona:', '1')
    fascia = st.text_input('Fascia Territoriale:', 'C1')
    sup_nl_loc = st.number_input('Superficie Netta Locabile (mq):', value=90, min_value=1)
    zona = st.text_input('Codice Zona OMI:', 'B1')
    linkzona = st.text_input('ID Zona (linkzona):', 'E280B1')
    cod_tip = st.text_input('Codice Tipologia:', 'A/2')
    descr_tip_prev = st.text_input('Tipologia Prevalente:', 'ABITAZIONI CIVILI')
    cod_tip_prev = st.text_input('Codice Tipologia Prevalente:', 'A/2')
    comune_istat = st.text_input('Codice ISTAT Comune:', '023091')

    # Bottone per avviare la classificazione
    classify_button = st.button('Classifica Immobile', type="primary", use_container_width=True)

# ===================================================================
# PAGINA PRINCIPALE E LOGICA DI PREVISIONE
# ===================================================================
st.title("🏠 Matter of Design - Dashboard di Valutazione Immobiliare")
st.markdown("Provincia di Verona")
st.markdown("---")


if classify_button:
    with st.spinner('Il nostro motore sta analizzando i dati... attendere prego.'):
        # Creazione del DataFrame con gli input
        input_data = pd.DataFrame({
            'descr_tipologia': [descr_tipologia], 'comune_descrizione': [comune_descrizione],
            'stato': [stato], 'zona_descr': [zona_descr], 'distanza_centro_m': [distanza_centro_m],
            'sup_tot_imm': [sup_tot_imm], 'fascia': [fascia], 'zona': [zona], 'linkzona': [linkzona],
            'cod_tip': [cod_tip], 'sup_nl_loc': [sup_nl_loc], 'semestre': [semestre],
            'cod_tip_prev': [cod_tip_prev], 'descr_tip_prev': [descr_tip_prev],
            'microzona': [microzona], 'comune_istat': [comune_istat]
        })

        try:
            # Preprocessing e Predizione
            processed_input = preprocessor.transform(input_data)
            input_tensor = torch.FloatTensor(processed_input)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_index = torch.argmax(probabilities, dim=1).item()
            
            predicted_class_name = label_encoder.inverse_transform([predicted_index])[0]
            confidence = probabilities[0][predicted_index].item()
            
            time.sleep(1) # Simula un calcolo più lungo per dare enfasi

            # --- VISUALIZZAZIONE RISULTATI ---
            st.header("✅ Risultato della Classificazione")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Categoria Stimata", value=predicted_class_name)
            with col2:
                st.metric(label="Livello di Confidenza", value=f"{confidence:.2%}")

            st.markdown("---")
            st.subheader("Dettaglio Probabilità per Categoria")
            
            probs_df = pd.DataFrame(probabilities.numpy(), columns=label_encoder.classes_).T
            probs_df.reset_index(inplace=True)
            probs_df.columns = ['Categoria', 'Probabilità']
            probs_df = probs_df.sort_values('Probabilità', ascending=False)

            for index, row in probs_df.iterrows():
                st.markdown(f"**{row['Categoria']}**")
                st.progress(row['Probabilità'], text=f"{row['Probabilità']:.2%}")

            with st.expander("Mostra i dati di input utilizzati"):
                st.dataframe(input_data)

        except Exception as e:
            st.error(f"Si è verificato un errore durante la classificazione: {e}")
            st.warning("Verificare che tutti i campi siano stati compilati correttamente.")

else:
    st.info("Utilizza il pannello a sinistra per inserire i dati di un immobile e avviare la classificazione.")
    st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?q=80&w=2070&auto=format&fit=crop", 
             caption="Immagine generica di un immobile residenziale")