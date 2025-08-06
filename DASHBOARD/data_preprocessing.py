import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import joblib

# 1. Path
### COMUNE ###
# Percorso della cartella con i file
#cartella_dati = "C:/Users/emmal/Desktop/STAGE/PROGETTO/DatiOMI/VERONA_comune"

# Dizionari con i nomi dei file
#files_valori = {
#    "2023_1": os.path.join(cartella_dati, "QI_1137052_1_20231_VALORI.csv"),
#    "2023_2": os.path.join(cartella_dati, "QI_1137050_1_20232_VALORI.csv"),
#    "2024_1": os.path.join(cartella_dati, "QI_1137049_1_20241_VALORI.csv"),
#    "2024_2": os.path.join(cartella_dati, "QIP_1184580_1_20242_VALORI.csv")
#}

#files_zone = {
#    "2023_1": os.path.join(cartella_dati, "QI_1137052_1_20231_ZONE.csv"),
#    "2023_2": os.path.join(cartella_dati, "QI_1137050_1_20232_ZONE.csv"),
#    "2024_1": os.path.join(cartella_dati, "QI_1137049_1_20241_ZONE.csv"),
#    "2024_2": os.path.join(cartella_dati, "QIP_1184580_1_20242_ZONE.csv")
#}

# 1. Path
### PROVINCIA ###
# Percorso della cartella con i file
cartella_dati = "C:/Users/emmal/Desktop/STAGE/PROGETTO/DatiOMI/VERONA_prov"


# Dizionari con i nomi dei file
files_valori = {
    "2023_1": os.path.join(cartella_dati, "QI_1201235_1_20231_VALORI.csv"),
    "2023_2": os.path.join(cartella_dati, "QIP_1201216_1_20232_VALORI.csv"),
    "2024_1": os.path.join(cartella_dati, "QI_1201237_1_20241_VALORI.csv"),
    "2024_2": os.path.join(cartella_dati, "QI_1201238_1_20242_VALORI.csv")

}

files_zone = {
    "2023_1": os.path.join(cartella_dati, "QI_1201235_1_20231_ZONE.csv"),
    "2023_2": os.path.join(cartella_dati, "QIP_1201216_1_20232_ZONE.csv"),
    "2024_1": os.path.join(cartella_dati, "QI_1201237_1_20241_ZONE.csv"),
    "2024_2": os.path.join(cartella_dati, "QI_1201238_1_20242_ZONE.csv")
}


# 2. Funzione Generica per Caricare un CSV
def carica_csv(file_path, semestre):
    """
    Carica un file CSV, rimuove la prima riga di intestazione
    e aggiunge la colonna 'semestre' per identificare il periodo.
    """
    df = pd.read_csv(file_path, sep=";", encoding="latin1", skiprows=1)  # Salta la prima riga
    df.columns = df.columns.str.lower().str.strip()  # Standardizza i nomi delle colonne
    df.drop(columns=[col for col in df.columns if "unnamed" in col], inplace=True, errors="ignore")  # Rimuove colonne inutili
    df["semestre"] = semestre  # Aggiunge colonna semestre 
    return df

# Caricamento di tutti i file
df_valori_list = [carica_csv(files_valori[sem], sem) for sem in files_valori]
df_zone_list = [carica_csv(files_zone[sem], sem) for sem in files_zone]

# Creiamo due DataFrame unificati
df_valori = pd.concat(df_valori_list, ignore_index=True)
df_zone = pd.concat(df_zone_list, ignore_index=True)

# Verifichiamo i dati caricati
print("\n Prime righe del dataset VALORI:\n", df_valori.head())
print("\n Prime righe del dataset ZONE:\n", df_zone.head())

# 3. Pulizia dati
def pulisci_dati(df):
    """
    Pulisce il DataFrame:
    - Converte colonne numeriche in float
    - Sostituisce i valori mancanti:
        - Con la mediana per le colonne numeriche
        - Con la moda (valore più frequente) per le colonne categoriche
    """
    #  Identifica colonne numeriche e categoriche
    colonne_numeriche = df.select_dtypes(include=["number"]).columns.tolist()
    colonne_categoriche = df.select_dtypes(exclude=["number"]).columns.tolist()

    #  Converte colonne numeriche con la virgola in float
    for col in df.columns:
        try:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        except ValueError:
            pass  # Se la colonna non è numerica, la ignoriamo

    #  Sostituisce i valori mancanti
    for col in colonne_numeriche:
        df[col].fillna(df[col].median(), inplace=True)  # Imputazione con mediana per numeri
    
    for col in colonne_categoriche:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Imputazione con il valore più frequente per testo

    return df

def rimuovi_colonne_uniche(df):
    """
    Rimuove le colonne che contengono un solo valore unico.
    """
    colonne_da_rimuovere = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=colonne_da_rimuovere)
    print(f"\n Colonne rimosse: {colonne_da_rimuovere}")
    return df

