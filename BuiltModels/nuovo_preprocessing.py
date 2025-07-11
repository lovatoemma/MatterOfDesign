import os
import pandas as pd
import geopandas as gpd
import re

def _carica_e_concatena_csv(lista_percorsi_file):
    """Funzione helper per caricare e concatenare file CSV."""
    lista_df = []
    for percorso, semestre in lista_percorsi_file:
        try:
            semestre_formattato = f"{semestre[:4]}_{semestre[4]}"
            df = pd.read_csv(percorso, sep=";", encoding="latin1", skiprows=1)
            df.columns = df.columns.str.lower().str.strip()
            df.drop(columns=[col for col in df.columns if "unnamed" in col], inplace=True, errors="ignore")
            df["semestre"] = semestre_formattato
            lista_df.append(df)
        except Exception as e:
            print(f"Attenzione: errore nel caricamento del file CSV {percorso}. Errore: {e}")
    return pd.concat(lista_df, ignore_index=True) if lista_df else pd.DataFrame()

def _carica_e_concatena_kml(lista_percorsi_kml, zona_col_name):
    """Funzione helper per leggere i singoli file KML e concatenarli."""
    lista_gdf = []
    for percorso_kml, semestre in lista_percorsi_kml:
        try:
            semestre_formattato = f"{semestre[:4]}_{semestre[4]}"
            gdf = gpd.read_file(percorso_kml, driver='KML')
            if 'Name' not in gdf.columns:
                continue
            gdf.rename(columns={'Name': zona_col_name}, inplace=True)
            gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
            gdf['semestre'] = semestre_formattato
            lista_gdf.append(gdf)
        except Exception as e:
            print(f"Attenzione: errore nel processare il file KML {percorso_kml}. Errore: {e}")
    if not lista_gdf:
        return gpd.GeoDataFrame()
    print(f"Concatenazione di {len(lista_gdf)} geometrie KML...")
    gdf_completo = pd.concat(lista_gdf, ignore_index=True)
    return gpd.GeoDataFrame(gdf_completo, geometry='geometry', crs="EPSG:4326")

def pulisci_dati(df):
    """Pulisce il DataFrame convertendo tipi e imputando valori mancanti."""
    print(f"Inizio pulizia dati su DataFrame di shape {df.shape}...")
    for col in df.columns:
        if col != 'geometry':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='ignore')
    colonne_numeriche = df.select_dtypes(include=["number"]).columns.tolist()
    colonne_categoriche = df.select_dtypes(exclude=["number", "geometry"]).columns.tolist()
    for col in colonne_numeriche:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in colonne_categoriche:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Mancante", inplace=True)
    print("Pulizia dati completata.")
    return df

def rimuovi_colonne_uniche(df, colonne_da_proteggere=None):
    """Rimuove le colonne con un solo valore unico, escludendo quelle protette."""
    if colonne_da_proteggere is None:
        colonne_da_proteggere = []
    colonne_da_rimuovere = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1 and col not in colonne_da_proteggere
    ]
    if colonne_da_rimuovere:
        df = df.drop(columns=colonne_da_rimuovere)
        print(f"Colonne con valore unico rimosse (escluse quelle protette): {colonne_da_rimuovere}")
    return df

def carica_dati_separati(cartella_dati, zona_col_name='zona'):
    """
    Funzione principale che carica i dati dalla struttura finale:
    - CSV sfusi nella cartella principale.
    - KML sfusi in sottocartelle (es. VR20201).
    """
    print(f"--- Inizio scansione della cartella dati: {cartella_dati} ---")
    if not os.path.isdir(cartella_dati):
        raise FileNotFoundError(f"Percorso non valido: {cartella_dati}")

    percorsi_valori, percorsi_zone, percorsi_kml = [], [], []

    # 1. Scansione file CSV sfusi nella cartella principale
    for nome_file in os.listdir(cartella_dati):
        path_file = os.path.join(cartella_dati, nome_file)
        if os.path.isfile(path_file):
            match = re.search(r'_(\d{5})_', nome_file)
            if match:
                semestre = match.group(1)
                if nome_file.lower().endswith('_valori.csv'):
                    percorsi_valori.append((path_file, semestre))
                elif nome_file.lower().endswith('_zone.csv'):
                    percorsi_zone.append((path_file, semestre))

    # 2. Scansione sottocartelle per i file KML
    sottocartelle = [d for d in os.listdir(cartella_dati) if os.path.isdir(os.path.join(cartella_dati, d))]
    for nome_cartella_semestre in sottocartelle:
        # Estrai il codice del semestre dal nome della cartella (es. da 'VR20201' a '20201')
        match_cartella = re.search(r'(\d{5})', nome_cartella_semestre)
        if not match_cartella:
            continue # Salta le cartelle che non hanno un codice semestre nel nome
        
        semestre = match_cartella.group(1)
        path_cartella_semestre = os.path.join(cartella_dati, nome_cartella_semestre)
        for nome_file_kml in os.listdir(path_cartella_semestre):
            if nome_file_kml.lower().endswith('.kml'):
                path_file_kml = os.path.join(path_cartella_semestre, nome_file_kml)
                percorsi_kml.append((path_file_kml, semestre))

    print(f"Trovati {len(percorsi_valori)} file VALORI, {len(percorsi_zone)} file ZONE, e {len(percorsi_kml)} file KML.")
    
    # Il resto della funzione rimane invariato
    df_tabulare_valori = _carica_e_concatena_csv(percorsi_valori)
    df_tabulare_zone = _carica_e_concatena_csv(percorsi_zone)
    if df_tabulare_valori.empty or df_tabulare_zone.empty:
        raise FileNotFoundError("Dati VALORI o ZONE non trovati.")
    
    df_tabulare = pd.merge(df_tabulare_valori, df_tabulare_zone, on=['linkzona', 'semestre'], how='left', suffixes=('_val', '_zone'))
    
    gdf_geospaziale = _carica_e_concatena_kml(percorsi_kml, zona_col_name)
    
    colonne_chiave = ['semestre', 'linkzona', zona_col_name, 'geometry']
    df_tabulare_pulito = pulisci_dati(df_tabulare)
    df_tabulare_pulito = rimuovi_colonne_uniche(df_tabulare_pulito, colonne_da_proteggere=colonne_chiave)
    
    gdf_geospaziale_pulito = None
    if not gdf_geospaziale.empty:
        gdf_geospaziale_pulito = pulisci_dati(gdf_geospaziale)
        gdf_geospaziale_pulito = rimuovi_colonne_uniche(gdf_geospaziale_pulito, colonne_da_proteggere=colonne_chiave)
        print("\nPreprocessing completato per entrambi i DataFrame.")
    else:
        print("\nATTENZIONE: Nessun dato geospaziale caricato o trovato.")

    return df_tabulare_pulito, gdf_geospaziale_pulito