from flask import Flask, request, session, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from flask_cors import CORS
import boto3
import os
import time
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics import renderPDF
import re
import boto3
import json
import os
from flask import Flask, request, jsonify
from fpdf import FPDF
import boto3
import os
from io import BytesIO
import matplotlib.pyplot as plt
import io
import base64
from sqlalchemy.dialects.mysql import JSON  
import math
from flask import Flask, render_template, request, redirect, url_for
import uuid

def normalizza_categoria(categoria: str) -> str:
    return categoria.strip().lower().replace(" ", "_")


# Carica le variabili di ambiente dal file .env
load_dotenv()

# Configurazione dell'app Flask
app = Flask(__name__)

app.secret_key = 'Design'  


CORS(app)  # Abilita CORS
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # URL del database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME')
app.config['AWS_ACCESS_KEY'] = os.getenv('AWS_ACCESS_KEY')
app.config['AWS_SECRET_KEY'] = os.getenv('AWS_SECRET_KEY')

#################
### DATA BASE ###
#################

# Configurazione del database
db = SQLAlchemy(app)

## PRODOTTO
class Prodotto(db.Model):
    __tablename__ = 'Prodotto'
    
    codice_articolo = db.Column(db.String(50), primary_key=True)
    brand = db.Column(db.String(100), nullable=False)
    finitura = db.Column(db.String(100), nullable=False)
    posa = db.Column(db.String(100), default=None)
    collezione = db.Column(db.String(100), nullable=False)
    dimensione = db.Column(db.String(50), nullable=False)
    prezzo_unitario = db.Column(db.Numeric(10, 2), default=None)
    prezzo_mq = db.Column(db.Numeric(10, 2), default=None)
    colore = db.Column(db.String(50), nullable=True)
    spessore = db.Column(db.Numeric(5, 2), default=None)
    tipologia_id = db.Column(db.Integer, db.ForeignKey('Tipologia.tipologia_id'), nullable=False)
    documento = db.Column(db.String(255), nullable=True)
    immagine = db.Column(db.String(255), nullable=True)
    nota = db.Column(db.String(500), default=None)
    peso = db.Column(db.Numeric(5, 2), default=None)
    categoria = db.Column(db.String(100), default=None)

    __table_args__ = (
        db.CheckConstraint(
            "(prezzo_unitario IS NOT NULL AND prezzo_mq IS NULL) OR "
            "(prezzo_unitario IS NULL AND prezzo_mq IS NOT NULL) OR "
            "(prezzo_unitario IS NULL AND prezzo_mq IS NULL)",
            name="check_prezzo_unitario_o_prezzo_mq"
        ),
    )

    def __repr__(self):
        return f"<Prodotto(codice_articolo={self.codice_articolo}, brand={self.brand}, categoria={self.categoria})>"

## ACCESSORI
class Accessori(db.Model):
    __tablename__ = 'Accessori'
    
    codice = db.Column(db.String(50), primary_key=True)  # Cambiato da accessorio_id
    prezzo = db.Column(db.Numeric(10, 2), nullable=False)
    descrizione = db.Column(db.String(500), nullable=True)  # Modifico la lunghezza della descrizione
    peso = db.Column(db.Numeric(5, 2), nullable=True)
    codice_articolo = db.Column(db.String(50), db.ForeignKey('Prodotto.codice_articolo'), nullable=False)
    
    prodotto = db.relationship('Prodotto', backref=db.backref('accessori', lazy=True))
    
    def __repr__(self):
        return f"<Accessori(codice={self.codice}, prezzo={self.prezzo}, descrizione={self.descrizione}, peso={self.peso})>"


## TIPOLOGIA        
class Tipologia(db.Model):
    __tablename__ = 'Tipologia'
    
    tipologia_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nome = db.Column(db.String(100), nullable=False)

    prodotti = db.relationship('Prodotto', backref='tipologia', lazy=True)

    def __repr__(self):
        return f"<Tipologia(tipologia_id={self.tipologia_id}, nome={self.nome})>"


## INTERVENTO IMMOBILIARE
class Intervento(db.Model):
    __tablename__ = 'Intervento'
    
    id = db.Column(db.String(50), primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    via = db.Column(db.String(200))
    cap = db.Column(db.String(10))
    citta = db.Column(db.String(100))
    cliente = db.Column(db.String(100))
    numero_unita = db.Column(db.Integer)
    numero_piani = db.Column(db.Integer)

    capitolati = db.relationship('Capitolato', backref='intervento', lazy=True)

## CAPITOLATO
class Capitolato(db.Model):
    __tablename__ = 'Capitolato'

    id = db.Column(db.String(50), primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    intervento_id = db.Column(db.String(50), db.ForeignKey('Intervento.id'), nullable=False)
    tipo = db.Column(db.String(10))
    data_creazione = db.Column(db.DateTime, default=db.func.current_timestamp())
    s3_url = db.Column(db.String(255))
    dati_json = db.Column(JSON)  


with app.app_context():
    db.create_all()

### FINE DATA BASE ###


###########
### API ###
###########


## Dashboard per visualizzare tutti gli interventi
@app.route('/interventi', methods=['GET'])
def lista_interventi():
    interventi = Intervento.query.all()
    data = [{
        "id": i.id,
        "nome": i.nome,
        "cliente": i.cliente,
        "via": i.via,
        "cap": i.cap,
        "citta": i.citta,
        "numero_unita": i.numero_unita,
        "numero_piani": i.numero_piani
    } for i in interventi]
    return jsonify(data), 200


## Visualizza i capitolati associati ad un intervento
@app.route('/interventi/<string:intervento_id>/capitolati', methods=['GET'])
def capitolati_intervento(intervento_id):
    capitolati = Capitolato.query.filter_by(intervento_id=intervento_id).all()
    data = [{
        "id": c.id,
        "nome": c.nome,
        "tipo": c.tipo,  #  "base" o "extra"
        "data_creazione": c.data_creazione,
        "s3_url": c.s3_url
    } for c in capitolati]
    return jsonify(data), 200

## Recupera un vecchio capitolato
@app.route('/carica-capitolato/<string:capitolato_id>', methods=['GET'])
def carica_capitolato(capitolato_id):
    capitolato = Capitolato.query.filter_by(id=capitolato_id).first()
    if not capitolato:
        return jsonify({"error": "Capitolato non trovato"}), 404

    # Ricarica i dati nella sessione se vuoi
    for chiave, valore in capitolato.dati_json.items():
        session[chiave] = valore

    return jsonify({
        "message": f"Capitolato '{capitolato.nome}' caricato in sessione",
        "dati": capitolato.dati_json
    }), 200

## 1. Endpoint per consentire al cliente di compilare i campi della 
# "Scheda dati intervento immobiliare".
@app.route('/intervento', methods=['POST'])
def salva_intervento():
    try:
        # Ricevi i dati JSON inviati dal frontend
        data = request.get_json()

        # Verifica che i dati obbligatori siano presenti
        campi_obbligatori = ["nome", "cliente", "via", "cap", "citta", "numero_unita", "numero_piani"]
        for campo in campi_obbligatori:
            if campo not in data or not data[campo]:
                return jsonify({"error": f"Il campo '{campo}' è obbligatorio"}), 400

        # Creo un nuovo intervento
        nuovo_intervento = Intervento(
            id=str(uuid.uuid4()),  # Genera un ID unico
            nome=data["nome"],
            cliente=data["cliente"],
            via=data["via"],
            cap=data["cap"],
            citta=data["citta"],
            numero_unita=int(data["numero_unita"]),
            numero_piani=int(data["numero_piani"])
        )

        # Salva nel database
        db.session.add(nuovo_intervento)
        db.session.commit()

        return jsonify({"message": "Intervento salvato con successo", "id": nuovo_intervento.id}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
## 2. Endpoint per visualizzare le tipologie di prodotto 
@app.route('/tipologie', methods=['GET'])
def get_tipologie():
    try:
        tipologie = Tipologia.query.all()
        tipologie_list = [{"tipologia_id": t.tipologia_id, "nome": t.nome} for t in tipologie]
        return jsonify(tipologie_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


## 3. Endpoint per inserire un nuovo prodotto
@app.route('/inserisci-prodotto', methods=['POST'])
def inserisci_prodotto():
    try:
        dati = request.get_json()

        if isinstance(dati, dict):
            dati = [dati]

        messaggi = []

        for data in dati:
            campi_obbligatori = ["codice_articolo", "brand", "collezione", "tipologia_id", "categoria"]

            for campo in campi_obbligatori:
                if campo not in data or not data[campo]:
                    return jsonify({"error": f"Il campo {campo} è obbligatorio"}), 400

            categoria_normalizzata = normalizza_categoria(data["categoria"])

            prodotto_esistente = Prodotto.query.filter_by(codice_articolo=data["codice_articolo"]).first()

            if prodotto_esistente:
                prodotto = prodotto_esistente
                messaggio = f"Prodotto '{prodotto.codice_articolo}' già esistente, aggiunto a scelte_data."
            else:
                prodotto = Prodotto(
                    codice_articolo=data["codice_articolo"],
                    brand=data["brand"],
                    finitura=data["finitura"],
                    posa=data.get("posa"),
                    collezione=data["collezione"],
                    dimensione=data["dimensione"],
                    prezzo_unitario=data.get("prezzo_unitario"),
                    prezzo_mq=data.get("prezzo_mq"),
                    colore=data.get("colore"),
                    spessore=data.get("spessore"),
                    tipologia_id=data["tipologia_id"],
                    documento=data.get("documento"),
                    immagine=data.get("immagine"),
                    nota=data.get("nota"),
                    peso=data.get("peso"),
                    categoria=categoria_normalizzata
                )

                db.session.add(prodotto)
                db.session.commit()
                messaggio = f"Nuovo prodotto '{prodotto.codice_articolo}' inserito e aggiunto a scelte_data."

            if 'scelte_data' not in session:
                session['scelte_data'] = {}

            if str(prodotto.tipologia_id) not in session['scelte_data']:
                session['scelte_data'][str(prodotto.tipologia_id)] = []

            if prodotto.codice_articolo not in session['scelte_data'][str(prodotto.tipologia_id)]:
                session['scelte_data'][str(prodotto.tipologia_id)].append(prodotto.codice_articolo)

            messaggi.append(messaggio)

        return jsonify({"message": messaggi}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

## 4. Endpoint per visualizzare che i prodotti siano inseriti correttamente 
# in base alla tipologia
@app.route('/prodotti/tipologia/<int:tipologia_id>', methods=['GET'])
def get_prodotti_by_tipologia(tipologia_id):
    try:
        prodotti = Prodotto.query.filter_by(tipologia_id=tipologia_id).all()
        result = [{
            "codice_articolo": p.codice_articolo,
            "brand": p.brand,
            "collezione": p.collezione,
            "prezzo_unitario": float(p.prezzo_unitario) if p.prezzo_unitario else None,
            "prezzo_mq": float(p.prezzo_mq) if p.prezzo_mq else None
        } for p in prodotti]

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

## Endpoint per associare gli accessori ad ogni prodotto selezionato
@app.route('/associare-accessori', methods=['POST'])
def associare_accessori():
    try:
        dati = request.get_json()

        # Se i dati sono un dizionario, li trasformiamo in una lista
        if isinstance(dati, dict):
            dati = [dati]

        messaggi = []

        # Itera su tutti gli accessori da associare
        for data in dati:
            # Verifica che il codice_articolo sia presente nei dati
            if 'codice_articolo' not in data:
                return jsonify({"error": "Il campo 'codice_articolo' è obbligatorio"}), 400

            codice_articolo = data['codice_articolo']
            accessori_data = data.get('accessori', [])

            # Se non ci sono accessori, procediamo comunque senza errore
            if not accessori_data:
                messaggi.append(f"Nessun accessorio da associare per il prodotto {codice_articolo}.")
                continue  

            # Verifica che il prodotto esista
            prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
            if not prodotto:
                return jsonify({"error": f"Prodotto con codice {codice_articolo} non trovato"}), 404

            # Gestione accessori
            for accessorio in accessori_data:
                if 'codice' not in accessorio or 'prezzo' not in accessorio or 'descrizione' not in accessorio or 'peso' not in accessorio:
                    return jsonify({"error": "Ogni accessorio deve contenere 'codice', 'prezzo', 'descrizione' e 'peso'"}), 400

                # Cerca se l'accessorio esiste già
                accessorio_esistente = Accessori.query.filter_by(codice=accessorio['codice']).first()

                if accessorio_esistente:
                    messaggio = f"Accessorio '{accessorio['codice']}' già esistente, associato al prodotto {codice_articolo}."
                else:
                    # Nuovo accessorio
                    nuovo_accessorio = Accessori(
                        codice=accessorio['codice'],
                        prezzo=accessorio['prezzo'],
                        descrizione=accessorio['descrizione'],
                        peso=accessorio['peso'],
                        codice_articolo=codice_articolo
                    )
                    db.session.add(nuovo_accessorio)
                    messaggio = f"Accessorio '{accessorio['codice']}' aggiunto al prodotto {codice_articolo}."
                messaggi.append(messaggio)

            # Commit dei cambiamenti al database
            db.session.commit()

        return jsonify({"message": messaggi}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


## 5. Endpoint per salvare i prodotti selezionati
@app.route('/salva-scelte', methods=['POST'])
def salva_scelte():
    try:
        # Recupera i dati JSON dalla richiesta
        data = request.get_json()

        # Verifica che i dati siano corretti
        if not data or 'scelte' not in data:
            return jsonify({"error": "Invalid JSON payload. The 'scelte' key is required."}), 400

        scelte = data['scelte']

        # Verifica 
        if 'scelte_data' not in session:
            session['scelte_data'] = {}

        if 'accessori_data' not in session:
            session['accessori_data'] = {}

        for scelta in scelte:
            tipologia_id = scelta.get('tipologia_id')
            codice_articolo = scelta.get('codice_articolo')
            accessori_selezionati = scelta.get('accessori', [])

            # Controllo che tipologia_id e codice_articolo siano forniti
            if not tipologia_id or not codice_articolo:
                return jsonify({"error": "Invalid scelta format. Both 'tipologia_id' and 'codice_articolo' are required."}), 400

            # Aggiungo il codice articolo alla sessione per la tipologia specifica
            if str(tipologia_id) not in session['scelte_data']:
                session['scelte_data'][str(tipologia_id)] = []

            if codice_articolo not in session['scelte_data'][str(tipologia_id)]:
                session['scelte_data'][str(tipologia_id)].append(codice_articolo)


            if accessori_selezionati:
                if codice_articolo not in session['accessori_data']:
                    session['accessori_data'][codice_articolo] = []

                for accessorio in accessori_selezionati:
                    # Verifica se l'accessorio esiste nel database
                    accessorio_codice = accessorio.get('codice')
                    accessorio_prezzo = accessorio.get('prezzo')
                    accessorio_descrizione = accessorio.get('descrizione')
                    accessorio_peso = accessorio.get('peso')

                    if not accessorio_codice or not accessorio_prezzo or not accessorio_descrizione:
                        return jsonify({"error": "Accessorio con codice mancante o dati incompleti"}), 400

                    accessorio_db = Accessori.query.filter_by(codice=accessorio_codice).first()
                    if not accessorio_db:
                        # Se l'accessorio non esiste, lo creiamo
                        nuovo_accessorio = Accessori(
                            codice=accessorio_codice,
                            prezzo=accessorio_prezzo,
                            descrizione=accessorio_descrizione,
                            peso=accessorio_peso,
                            codice_articolo=codice_articolo
                        )
                        db.session.add(nuovo_accessorio)
                        db.session.commit()

                    # Aggiungo l'accessorio alla lista degli accessori selezionati per il prodotto
                    if accessorio_codice not in session['accessori_data'][codice_articolo]:
                        session['accessori_data'][codice_articolo].append(accessorio_codice)

            # Se non ci sono accessori, salvo comunque l'articolo con lista vuota
            else:
                session['accessori_data'][codice_articolo] = []

        return jsonify({"message": "Le scelte e gli accessori sono stati salvati correttamente"}), 201

    except Exception as e:
        # Gestione degli errori in caso di eccezioni
        return jsonify({"error": str(e)}), 500




## 6. Endpoint per salvare le quantità scelte dall'utente per l'intero intervento
@app.route('/salva-quantita', methods=['POST'])
def salva_quantita():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Dati non forniti"}), 400

        if 'quantita_data' not in session:
            session['quantita_data'] = {}

        if 'accessori_data' not in session:
            session['accessori_data'] = {}

        for codice_articolo, valore in data.items():
            try:
                # Impostiamo la quantità del prodotto
                quantita = float(valore)
                if math.isnan(quantita) or quantita < 0:
                    quantita = 0
            except (ValueError, TypeError):
                quantita = 0

            # Salva la quantità del prodotto
            session['quantita_data'][codice_articolo] = quantita

            # Verifica se il prodotto ha degli accessori associati
            if codice_articolo in session['accessori_data']:
                accessori_selezionati = session['accessori_data'][codice_articolo]

                # Assegna la stessa quantità agli accessori
                for codice_accessorio in accessori_selezionati:
                    if 'quantita_data' not in session:
                        session['quantita_data'] = {}

                    # Salva la quantità per ogni accessorio
                    session['quantita_data'][codice_accessorio] = quantita

        return jsonify({"message": "Quantità salvate correttamente per i prodotti e gli accessori associati"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


## 7. Endpoint per salvare le quantità relative all'unità abitativa
@app.route('/unita-abitative', methods=['POST'])
def generate_unita_abitative():
    try:
        # Recupera il numero di unità direttamente da intervento_data
        intervento = session.get('intervento_data', {})
        numero_unita = intervento.get('numero_unita')

        if not numero_unita or not str(numero_unita).isdigit():
            return jsonify({"error": "Dati intervento non validi o numero_unita mancante"}), 400

        numero_unita = int(numero_unita)
        if numero_unita < 0 or numero_unita > 100:
            return jsonify({"error": "Il numero di unità abitative deve essere compreso tra 0 e 100"}), 400

        data = request.get_json() or {}

        # Inizializza struttura dati in sessione
        session['unita_abitative_data'] = {}

        if 'accessori_data' not in session:
            session['accessori_data'] = {}

        for i in range(1, numero_unita + 1):
            unita_name = f"A{i:02d}"
            session['unita_abitative_data'][unita_name] = {}

            # Salva le quantità personalizzate per ogni unità
            unita_data = data.get(unita_name, {})
            for codice_articolo, quantita in unita_data.items():
                try:
                    quantita = float(quantita)
                    if math.isnan(quantita):
                        quantita = 0
                except (ValueError, TypeError):
                    quantita = 0
                session['unita_abitative_data'][unita_name][codice_articolo] = quantita

                # Verifica se ci sono accessori associati al prodotto
                if codice_articolo in session['accessori_data']:
                    accessori_selezionati = session['accessori_data'][codice_articolo]

                    # Assegna la stessa quantità agli accessori
                    for codice_accessorio in accessori_selezionati:
                        if 'unita_abitative_data' not in session:
                            session['unita_abitative_data'] = {}

                        # Salva la quantità per ogni accessorio nella stessa unità
                        if unita_name not in session['unita_abitative_data']:
                            session['unita_abitative_data'][unita_name] = {}

                        session['unita_abitative_data'][unita_name][codice_accessorio] = quantita

        return jsonify({"message": "Dati delle unità abitative e degli accessori salvati nella sessione"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



## 8. Capitolato finale di tutto l'intervento immobiliare
@app.route('/capitolato-finale-completo', methods=['POST'])
def visualizza_capitolato_fine_completo():
    try:
        data = request.get_json()
        if not data or 'margini_mod' not in data or 'sconti_prodotti' not in data:
            return jsonify({"error": "Devi fornire margini MOD e condizioni di sconto per ogni prodotto"}), 400

        margini_mod = data['margini_mod']
        sconti_prodotti = data['sconti_prodotti']

        if not session.get('quantita_data') or not session.get('scelte_data'):
            return jsonify({"error": "Dati mancanti per generare il capitolato finale"}), 400

        quantita_data = session['quantita_data']
        scelte_data = session['scelte_data']

        capitolato = []
        totale_prezzo_cliente = 0
        totale_guadagno_mod = 0
        costi_tipologie = {}

        for tipologia_id, articoli in scelte_data.items():
            tipologia = Tipologia.query.filter_by(tipologia_id=tipologia_id).first()
            if not tipologia:
                continue

            prodotti_dettagli = []
            totale_tipologia = 0
            tipologia_nome = tipologia.nome.upper()

            if isinstance(articoli, dict):
                articoli = [item for sublist in articoli.values() for item in sublist]

            for codice_articolo in articoli:
                prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
                if not prodotto:
                    continue

                prezzo_listino = float(prodotto.prezzo_unitario or 0)
                prezzo_mq = float(prodotto.prezzo_mq or 0)
                prezzo_da_usare = prezzo_listino if prezzo_mq == 0 else prezzo_mq

                condizioni_sconto = sconti_prodotti.get(codice_articolo, "")
                prezzo_netto_mod = prezzo_da_usare
                if condizioni_sconto:
                    sconti = [float(s.strip().replace('%', '')) for s in condizioni_sconto.split(',')]
                    for sconto in sconti:
                        prezzo_netto_mod *= (1 - sconto / 100)

                percentuale_mod = float(margini_mod.get(codice_articolo, 5))
                prezzo_cliente = prezzo_netto_mod * (1 + percentuale_mod / 100)
                guadagno_mod = prezzo_cliente - prezzo_netto_mod

                quantita = float(quantita_data.get(codice_articolo, 0))
                if math.isnan(quantita):
                    quantita = 0

                totale_prodotto = prezzo_cliente * quantita
                guadagno_prodotto = guadagno_mod * quantita

                totale_tipologia += totale_prodotto
                totale_prezzo_cliente += totale_prodotto
                totale_guadagno_mod += guadagno_prodotto

                prodotti_dettagli.append({
                    "codice_articolo": prodotto.codice_articolo,
                    "collezione": prodotto.collezione,
                    "brand": prodotto.brand,
                    "categoria": prodotto.categoria,
                    "prezzo_listino": round(prezzo_da_usare, 2),
                    "condizioni_sconto_mod": condizioni_sconto,
                    "prezzo_netto_mod": round(prezzo_netto_mod, 2),
                    "percentuale_mod": percentuale_mod,
                    "prezzo_cliente": round(prezzo_cliente, 2),
                    "guadagno_mod": round(guadagno_mod, 2),
                    "quantita": quantita,
                    "totale_prezzo_cliente": round(totale_prodotto, 2),
                    "totale_guadagno_mod": round(guadagno_prodotto, 2)
                })

                if codice_articolo in session.get('accessori_data', {}):
                    accessori_selezionati = session['accessori_data'][codice_articolo]
                    for codice_accessorio in accessori_selezionati:
                        accessorio = Accessori.query.filter_by(codice=codice_accessorio).first()
                        if accessorio:
                            prezzo_accessorio_da_usare = float(accessorio.prezzo or 0)

                            condizioni_sconto_accessorio = sconti_prodotti.get(codice_accessorio, "")
                            prezzo_netto_mod_accessorio = prezzo_accessorio_da_usare
                            if condizioni_sconto_accessorio:
                                sconti = [float(s.strip().replace('%', '')) for s in condizioni_sconto_accessorio.split(',')]
                                for sconto in sconti:
                                    prezzo_netto_mod_accessorio *= (1 - sconto / 100)

                            percentuale_mod_accessorio = float(margini_mod.get(codice_accessorio, 5))
                            prezzo_cliente_accessorio = prezzo_netto_mod_accessorio * (1 + percentuale_mod_accessorio / 100)
                            guadagno_mod_accessorio = prezzo_cliente_accessorio - prezzo_netto_mod_accessorio

                            quantita_accessorio = quantita_data.get(codice_accessorio, 0)
                            totale_accessorio = prezzo_cliente_accessorio * quantita_accessorio
                            guadagno_accessorio = guadagno_mod_accessorio * quantita_accessorio

                            prodotti_dettagli.append({
                                "codice_articolo": accessorio.codice,
                                "collezione": accessorio.descrizione,
                                "brand": "Accessori",
                                "categoria": "Accessori",
                                "prezzo_listino": round(prezzo_accessorio_da_usare, 2),
                                "condizioni_sconto_mod": condizioni_sconto_accessorio,
                                "prezzo_netto_mod": round(prezzo_netto_mod_accessorio, 2),
                                "percentuale_mod": percentuale_mod_accessorio,
                                "prezzo_cliente": round(prezzo_cliente_accessorio, 2),
                                "guadagno_mod": round(guadagno_mod_accessorio, 2),
                                "quantita": quantita_accessorio,
                                "totale_prezzo_cliente": round(totale_accessorio, 2),
                                "totale_guadagno_mod": round(guadagno_accessorio, 2)
                            })

                            totale_tipologia += totale_accessorio
                            totale_prezzo_cliente += totale_accessorio
                            totale_guadagno_mod += guadagno_accessorio

            capitolato.append({
                "tipologia": tipologia_nome,
                "prodotti": prodotti_dettagli,
                "totale_tipologia": round(totale_tipologia, 2)
            })

            costi_tipologie[tipologia_nome] = totale_tipologia
        # Crea il grafico a torta per il capitolato
        img_base64 = ""
        labels = []
        sizes = []

        for nome, valore in costi_tipologie.items():
            try:
                valore = float(valore)
                if not math.isnan(valore) and valore > 0:
                    labels.append(nome)
                    sizes.append(valore)
            except (ValueError, TypeError):
                continue

        if sizes:
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
            plt.close(fig)


        return jsonify({
            "capitolato": capitolato,
            "totale_prezzo_cliente": round(totale_prezzo_cliente, 2),
            "totale_guadagno_mod": round(totale_guadagno_mod, 2),
            "grafico_torta_base64": img_base64
        }), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500




## Endpoint per ottenere il capitolato suddiviso per unità abitative        
@app.route('/capitolato-per-unita', methods=['POST'])
def capitolato_per_unita():
    try:
        data = request.get_json()

        if not data or 'margini_mod' not in data or 'sconti_prodotti' not in data:
            return jsonify({"error": "Devi fornire margini MOD e condizioni di sconto per ogni prodotto"}), 400

        margini_mod = data['margini_mod']
        sconti_prodotti = data['sconti_prodotti']

        required_sessions = ['intervento_data', 'scelte_data', 'unita_abitative_data', 'quantita_data']
        if not all(session.get(k) for k in required_sessions):
            return jsonify({"error": "Dati mancanti per generare il capitolato per unità"}), 400

        scelte_data = session['scelte_data']
        unita_abitative_data = session['unita_abitative_data']
        quantita_data = session['quantita_data']

        capitolato_per_unita = {}
        totale_prezzo_cliente_global = 0
        totale_guadagno_mod_global = 0

        for unita, unita_info in unita_abitative_data.items():
            unita_totale_prezzo_cliente = 0
            unita_totale_guadagno_mod = 0
            dettagli_per_unita = []

            for tipologia_id, articoli in scelte_data.items():
                tipologia = Tipologia.query.filter_by(tipologia_id=tipologia_id).first()
                if not tipologia:
                    continue

                prodotti_dettagli = []
                totale_tipologia = 0
                tipologia_nome = tipologia.nome.upper()

                # Assicurati che articoli sia una lista piatta
                if isinstance(articoli, dict):
                    articoli = [item for sublist in articoli.values() for item in sublist]

                for codice_articolo in articoli:
                    prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
                    if not prodotto:
                        continue

                    prezzo_listino = float(prodotto.prezzo_unitario or 0)
                    prezzo_mq = float(prodotto.prezzo_mq or 0)
                    prezzo_da_usare = prezzo_listino if prezzo_mq == 0 else prezzo_mq

                    condizioni_sconto = sconti_prodotti.get(codice_articolo, "")
                    prezzo_netto_mod = prezzo_da_usare
                    if condizioni_sconto:
                        sconti = [float(s.strip().replace('%', '')) for s in condizioni_sconto.split(',')]
                        for sconto in sconti:
                            prezzo_netto_mod *= (1 - sconto / 100)

                    percentuale_mod = float(margini_mod.get(codice_articolo, 5))
                    prezzo_cliente = prezzo_netto_mod * (1 + percentuale_mod / 100)
                    guadagno_mod = prezzo_cliente - prezzo_netto_mod

                    quantita = float(unita_info.get(codice_articolo, quantita_data.get(codice_articolo, 0)))
                    if math.isnan(quantita):
                        quantita = 0

                    totale_prodotto = prezzo_cliente * quantita
                    guadagno_prodotto = guadagno_mod * quantita

                    totale_tipologia += totale_prodotto
                    unita_totale_prezzo_cliente += totale_prodotto
                    unita_totale_guadagno_mod += guadagno_prodotto

                    prodotti_dettagli.append({
                        "codice_articolo": prodotto.codice_articolo,
                        "collezione": prodotto.collezione,
                        "brand": prodotto.brand,
                        "categoria": prodotto.categoria,
                        "prezzo_listino": round(prezzo_da_usare, 2),
                        "condizioni_sconto_mod": condizioni_sconto,
                        "prezzo_netto_mod": round(prezzo_netto_mod, 2),
                        "percentuale_mod": percentuale_mod,
                        "prezzo_cliente": round(prezzo_cliente, 2),
                        "guadagno_mod": round(guadagno_mod, 2),
                        "quantita": quantita,
                        "totale_prezzo_cliente": round(totale_prodotto, 2),
                        "totale_guadagno_mod": round(guadagno_prodotto, 2)
                    })

                    # Gestione accessori (senza prezzo_mq)
                    if codice_articolo in session.get('accessori_data', {}):
                        for codice_accessorio in session['accessori_data'][codice_articolo]:
                            accessorio = Accessori.query.filter_by(codice=codice_accessorio).first()
                            if accessorio:
                                prezzo_accessorio_da_usare = float(accessorio.prezzo or 0)

                                condizioni_sconto_accessorio = sconti_prodotti.get(codice_accessorio, "")
                                prezzo_netto_mod_accessorio = prezzo_accessorio_da_usare
                                if condizioni_sconto_accessorio:
                                    sconti = [float(s.strip().replace('%', '')) for s in condizioni_sconto_accessorio.split(',')]
                                    for sconto in sconti:
                                        prezzo_netto_mod_accessorio *= (1 - sconto / 100)

                                percentuale_mod_accessorio = float(margini_mod.get(codice_accessorio, 5))
                                prezzo_cliente_accessorio = prezzo_netto_mod_accessorio * (1 + percentuale_mod_accessorio / 100)
                                guadagno_mod_accessorio = prezzo_cliente_accessorio - prezzo_netto_mod_accessorio

                                quantita_accessorio = float(unita_info.get(codice_accessorio, quantita_data.get(codice_accessorio, 0)))
                                totale_accessorio = prezzo_cliente_accessorio * quantita_accessorio
                                guadagno_accessorio = guadagno_mod_accessorio * quantita_accessorio

                                totale_tipologia += totale_accessorio
                                unita_totale_prezzo_cliente += totale_accessorio
                                unita_totale_guadagno_mod += guadagno_accessorio

                                prodotti_dettagli.append({
                                    "codice_articolo": accessorio.codice,
                                    "collezione": accessorio.descrizione,
                                    "brand": "Accessori",
                                    "categoria": "Accessori",
                                    "prezzo_listino": round(prezzo_accessorio_da_usare, 2),
                                    "condizioni_sconto_mod": condizioni_sconto_accessorio,
                                    "prezzo_netto_mod": round(prezzo_netto_mod_accessorio, 2),
                                    "percentuale_mod": percentuale_mod_accessorio,
                                    "prezzo_cliente": round(prezzo_cliente_accessorio, 2),
                                    "guadagno_mod": round(guadagno_mod_accessorio, 2),
                                    "quantita": quantita_accessorio,
                                    "totale_prezzo_cliente": round(totale_accessorio, 2),
                                    "totale_guadagno_mod": round(guadagno_accessorio, 2)
                                })

                dettagli_per_unita.append({"tipologia": tipologia_nome, "prodotti": prodotti_dettagli, "totale_tipologia": round(totale_tipologia, 2)})

            capitolato_per_unita[unita] = dettagli_per_unita

        recap = {"totale_prezzo_cliente": round(totale_prezzo_cliente_global, 2), "totale_guadagno_mod": round(totale_guadagno_mod_global, 2)}
        capitolato_normale = session.get('capitolato_per_unita', {}).get('capitolato_per_unita', {})

        recap_differenze_extra = {}

        for unita in capitolato_extra_per_unita:
            totale_extra_unita = sum(
                tipologia['totale_tipologia'] for tipologia in capitolato_extra_per_unita[unita]
            )

            totale_normale_unita = sum(
                tipologia['totale_tipologia'] for tipologia in capitolato_normale.get(unita, [])
            )

            differenza_extra_unita = totale_extra_unita - totale_normale_unita

            recap_differenze_extra[unita] = round(differenza_extra_unita, 2)

        recap["differenze_extra_cliente"] = recap_differenze_extra

        session['capitolato_extra_per_unita'] = {"capitolato_extra_per_unita": capitolato_extra_per_unita, "recap": recap}

        return jsonify({"capitolato_extra_per_unita": capitolato_extra_per_unita, "recap": recap}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


##################################################
#### SALVATAGGIO DEL CAPITOLATO EXTRA SU S3 ######
##################################################
# SALVATAGGIO CAPITOLATO EXTRA DETTAGLIATO (basato sulla struttura di salva_capitolato_base)
@app.route('/salva-capitolato-extra-dettagliato', methods=['POST']) 
def salva_capitolato_extra_dettagliato(): 
    try:
        data = request.get_json()
        nome_capitolato = data.get("nome")
        intervento_id = data.get("intervento_id")

        if not all([nome_capitolato, intervento_id]):
            return jsonify({"error": "Nome capitolato e intervento_id sono obbligatori"}), 400

        # Recupera i dati del capitolato extra calcolati dalla sessione
        capitolato_extra_per_unita_data = session.get("capitolato_extra_per_unita", {}) 
        if not capitolato_extra_per_unita_data:
             return jsonify({"error": "Dati capitolato extra per unità non trovati in sessione. Eseguire prima il calcolo."}), 400

        # Crea PDF in memoria
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Intestazione
        pdf.cell(0, 10, txt=f"Capitolato Extra: {nome_capitolato}", ln=True, align='C')

        # Dettagli Capitolato extra per unità
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Capitolato Extra per Unità Abitative", ln=True)
        pdf.set_font("Arial", size=8) 
        # Converte il dizionario complesso in una stringa JSON formattata per il PDF
        try:
            pdf_content = json.dumps(capitolato_extra_per_unita_data, indent=2, ensure_ascii=False)
        except TypeError:
            pdf_content = str(capitolato_extra_per_unita_data) 
        pdf.multi_cell(0, 5, txt=pdf_content) 

        # 4. Salva il PDF in memoria
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        # 5. Carica su S3
        capitolato_id = uuid.uuid4().hex # Genera ID univoco
        # Aggiungo suffisso _extra 
        nome_file = f"{capitolato_id}_{nome_capitolato.replace(' ', '_')}_extra.pdf"
        s3_url = carica_s3(pdf_output, nome_file)

        if not s3_url:
            return jsonify({"error": "Errore nel caricamento del PDF su S3"}), 500

        # 6. Raccolgo tutti i dati della sessione relativi all'extra da salvare nel DB
        contenuto = {
            "intervento_data": session.get("intervento_data"), # Utile per contesto
            "scelte_extra_data": session.get("scelte_extra_data"),
            "accessori_extra_data": session.get("accessori_extra_data"),
            "unita_abitative_extra_data": session.get("unita_abitative_extra_data"),
            "capitolato_extra_per_unita": capitolato_extra_per_unita_data # Salva i dati calcolati usati per il PDF
            
        }

        # 7. Salva nel DB
        nuovo = Capitolato(
            id=capitolato_id,
            nome=nome_capitolato,
            tipo="extra", 
            intervento_id=intervento_id,
            s3_url=s3_url,
            dati_json=contenuto # Salva i dati extra raccolti
        )
        db.session.add(nuovo)
        db.session.commit()

        return jsonify({
            "message": "Capitolato Extra dettagliato salvato con successo", # Messaggio aggiornato
            "id": nuovo.id,
            "s3_url": s3_url  
        }), 201

    except Exception as e:
        db.session.rollback()
        # Log dell'errore lato server per debug
        app.logger.error(f"Errore in /salva-capitolato-extra-dettagliato: {str(e)}")
        return jsonify({"error": f"Errore interno del server: {str(e)}"}), 500





############################################
####           AVVIO APPLICAZIONE       ####
############################################
if __name__ == '__main__':

    app.run(debug=True) # debug=True è utile in sviluppo, disabilitalo in produzione