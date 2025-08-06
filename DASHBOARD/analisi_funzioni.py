# analisi_funzioni.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_distribuzione_numerica(df, colonna, titolo):
    """
    Crea un grafico che mostra l'istogramma e il boxplot di una colonna numerica.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': (0.8, 0.2)})
    sns.histplot(df[colonna], kde=True, ax=axes[0], bins=50)
    axes[0].set_title(f'Distribuzione di {titolo}', fontsize=16)
    axes[0].set_xlabel('')
    sns.boxplot(x=df[colonna], ax=axes[1])
    axes[1].set_xlabel(f'Valore di {titolo}', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Matrice di Confusione'):
    """
    Funzione per visualizzare una matrice di confusione.
    'classes' deve essere la lista dei nomi delle classi in formato stringa.
    Questa funzione si aspetta che y_true, y_pred e classes siano tutti dello stesso tipo (stringa).
    """
    # --- PROVA DI CARICAMENTO ---
    print(">>> ESEGUO LA VERSIONE DEFINITIVA DI plot_confusion_matrix <<<")
    
    # La funzione confusion_matrix di sklearn gestisce direttamente le etichette
    # in formato stringa. Le passiamo la lista 'classes' come riferimento
    # per garantire l'ordine corretto di righe e colonne.
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 9))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    
    plt.title(title, fontsize=18, pad=20)
    plt.ylabel('Classe Reale', fontsize=14)
    plt.xlabel('Classe Prevista', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()