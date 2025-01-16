from flask import Flask, request, jsonify, render_template
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

# Carica le variabili di ambiente dal file .env
load_dotenv()

# Configurazione dell'app Flask
app = Flask(__name__)

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

## TIPOLOGIA        
class Tipologia(db.Model):
    __tablename__ = 'Tipologia'
    
    tipologia_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nome = db.Column(db.String(100), nullable=False)

    prodotti = db.relationship('Prodotto', backref='tipologia', lazy=True)

    def __repr__(self):
        return f"<Tipologia(tipologia_id={self.tipologia_id}, nome={self.nome})>"
### FINE DATA BASE ###

###########
### API ###
###########
@app.route('/')
def home():
    return render_template('InterventoImmobiliare.html')  # Serve il file HTML nella directory "templates"


## 1. Endpoint per consentire al cliente di compilare i campi della 
# "Scheda dati intervento immobiliare".
# Variabile globale per salvare i dati dell'intervento
intervento_data = {}

@app.route('/intervento', methods=['POST'])
def create_intervento():
    try:
        # Leggi i dati inviati dal form HTML
        data = request.form

        # Controlla se ci sono dati
        if not data:
            return jsonify({"error": "Nessun dato inviato"}), 400

        # Salva i dati in memoria nella variabile globale
        global intervento_data
        intervento_data = {
            "nome_intervento": data.get('nome_intervento'),
            "via": data.get('via'),
            "cap": data.get('cap'),
            "citta": data.get('citta'),
            "cliente": data.get('cliente'),
            "numero_unita": data.get('numero_unita'),
            "numero_piani": data.get('numero_piani'),
            "sup_pavimenti_legno": data.get('sup_pavimenti_legno'),
            "sup_pavimenti": data.get('sup_pavimenti'),
            "sup_rivestimenti": data.get('sup_rivestimenti'),
            "vasi_wc": data.get('vasi_wc'),
            "bidet": data.get('bidet'),
            "lavabo": data.get('lavabo'),
            "piatto_doccia": data.get('piatto_doccia'),
            "vasca_bagno": data.get('vasca_bagno'),
            "termoarredo": data.get('termoarredo'),
            "miscelatori_lavabo": data.get('miscelatori_lavabo'),
            "miscelatori_bidet": data.get('miscelatori_bidet'),
            "gruppi_doccia": data.get('gruppi_doccia'),
            "porta_interna": data.get('porta_interna'),
            "maniglie_porte_interne": data.get('maniglie_porte_interne')
        }

        return jsonify({"message": "Tutti i campi sono stati correttamente compilati"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

   
## 2. Endpoint per generare i dati dinamicamente in base al numero di unità abitative scelto.
# "Scheda dati unità abitative" 
# Variabile globale per salvare i dati delle unità abitative
unita_abitative_data = {}

@app.route('/unita-abitative', methods=['POST'])
def generate_unita_abitative():
    try:
        data = request.get_json()
        numero_unita = data.get('numero_unita')

        # Validazione del numero di unità
        if not isinstance(numero_unita, int) or numero_unita < 0 or numero_unita > 100:
            return jsonify({"error": "Il numero di unità abitative deve essere compreso tra 0 e 100"}), 400

        # Salva i dati delle unità abitative in memoria
        global unita_abitative_data
        unita_abitative_data = {}

        for i in range(1, numero_unita + 1):
            unita_name = f"A{i:02d}"
            unita_abitative_data[unita_name] = {
                "piano": data.get(f"piano_{unita_name}"),
                "superficie": data.get(f"superficie_{unita_name}"),
                "sup_pavimenti_legno": data.get(f"sup_pavimenti_legno_{unita_name}"),
                "sup_pavimenti": data.get(f"sup_pavimenti_{unita_name}"),
                "sup_rivestimenti": data.get(f"sup_rivestimenti_{unita_name}"),
                "vasi_wc": data.get(f"vasi_wc_{unita_name}"),
                "bidet": data.get(f"bidet_{unita_name}"),
                "lavabo": data.get(f"lavabo_{unita_name}"),
                "piatto_doccia": data.get(f"piatto_doccia_{unita_name}"),
                "vasca_bagno": data.get(f"vasca_bagno_{unita_name}"),
                "termoarredo": data.get(f"termoarredo_{unita_name}"),
                "miscelatori_lavabo": data.get(f"miscelatori_lavabo_{unita_name}"),
                "miscelatori_bidet": data.get(f"miscelatori_bidet_{unita_name}"),
                "gruppi_doccia": data.get(f"gruppi_doccia_{unita_name}"),
                "porta_interna": data.get(f"porta_interna_{unita_name}"),
                "maniglie_porte_interne": data.get(f"maniglie_porte_interne_{unita_name}")
            }

        return jsonify({"unita_abitative": unita_abitative_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Salvataggio dei dati: Quando il cliente inserisce i dati nei due endpoint (/intervento e /unita-abitative), 
# questi vengono salvati in memoria.

## 3. Endpoint per visualizzare i dati inseriti nelle due schede precedenti
# In-memory storage
@app.route('/visualizza-dati', methods=['GET'])
def visualizza_dati():
    try:
        global intervento_data, unita_abitative_data

        # Combina i dati salvati
        combined_data = {
            "intervento": intervento_data,
            "unita_abitative": unita_abitative_data
        }

        return jsonify(combined_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Visualizzazione dei dati: Quando il cliente richiede la scheda riassuntiva tramite l'endpoint /visualizza-dati, 
# l'applicazione combina i dati salvati in memoria e li restituisce come JSON.

## 4. Endpoint per definire il nome di un capitolato
@app.route('/configura', methods=['POST', 'GET'])
def configura():
    if request.method == 'POST':
        # Riceve il nome del capitolato dal client
        data = request.get_json()
        nome_capitolato = data.get('nome_capitolato')
        if not nome_capitolato:
            return jsonify({"error": "Il nome del capitolato è obbligatorio"}), 400
        
        # Costruisce un URL per il frontend (se necessario)
        return jsonify({"redirect_url": f"/configura?capitolato={nome_capitolato.replace(' ', '%20')}"}), 200

    elif request.method == 'GET':
        # Riceve il capitolato come parametro di query
        nome_capitolato = request.args.get('capitolato')
        if not nome_capitolato:
            return jsonify({"error": "Nessun capitolato specificato"}), 400

        # Carica la pagina con i dati del capitolato
        return render_template('configura.html', capitolato=nome_capitolato)

## 5. Endpoint per visualizzare le tipologie di prodotto 
@app.route('/tipologie', methods=['GET'])
def get_tipologie():
    try:
        tipologie = Tipologia.query.all()
        tipologie_list = [{"tipologia_id": t.tipologia_id, "nome": t.nome} for t in tipologie]
        return jsonify(tipologie_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

## 6. Endpoint per ottenere i prodotti
@app.route('/prodotti/tipologia/<int:tipologia_id>', methods=['GET'])
def get_prodotti_by_tipologia(tipologia_id):
    try:
        prodotti = Prodotto.query.filter_by(tipologia_id=tipologia_id).all()
        result = [{
            "codice_articolo": p.codice_articolo,
            "brand": p.brand,
            "finitura": p.finitura,
            "posa": p.posa,
            "collezione": p.collezione,
            "dimensione": p.dimensione,
            "prezzo_unitario": float(p.prezzo_unitario) if p.prezzo_unitario else None,
            "prezzo_mq": float(p.prezzo_mq) if p.prezzo_mq else None,
            "colore": p.colore,
            "spessore": float(p.spessore) if p.spessore else None,
            "tipologia_id": p.tipologia_id,
            "documento": p.documento,
            "immagine": p.immagine,
            "nota": p.nota,
            "peso": float(p.peso) if p.peso else None,
            "categoria": p.categoria
        } for p in prodotti]

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Questo endpoint (/prodotti/tipologia/<int:tipologia_id>) accetta un tipologia_id come parametro e 
# restituisce tutti i prodotti che appartengono a quella tipologia.

## 7. Endpoint per salvare le scelte fatte dal cliente
# In-memory storage for scelte
scelte_data = {}

@app.route('/salva-scelte', methods=['POST'])
def salva_scelte():
    try:
        data = request.get_json()
        if not data or 'scelte' not in data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Salva le scelte in memoria
        scelte = data['scelte']
        for scelta in scelte:
            tipologia_id = scelta.get('tipologia_id')
            codice_articolo = scelta.get('codice_articolo')
            if not tipologia_id or not codice_articolo:
                return jsonify({"error": "Invalid scelta format"}), 400
            if tipologia_id not in scelte_data:
                scelte_data[tipologia_id] = []
            scelte_data[tipologia_id].append(codice_articolo)

        return jsonify({"message": "Le scelte sono state salvate correttamente"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

## 8. Endpoint per visualizzare il capitolato finale
@app.route('/capitolato-finale', methods=['GET'])
def visualizza_capitolato_finale():
    try:
        risultato = calcola_capitolato()
        if "error" in risultato:
            return jsonify(risultato), 400
        return jsonify(risultato), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


## 9. Endpoint per visualizzare il capitolato finale suddiviso per unità abitative
@app.route('/capitolato-unita-abitative', methods=['GET'])
def visualizza_capitolato_unita_abitative():
    try:
        if not intervento_data or not scelte_data or not unita_abitative_data:
            return jsonify({"error": "Dati mancanti per generare il capitolato per unità abitative"}), 400

        capitolato_unita = {}
        totale_finale = 0

        tipologia_quantita_map = {
            "pavimento_legno": ["sup_pavimenti_legno"],
            "pavimento": ["sup_pavimenti"],
            "rivestimenti": ["sup_rivestimenti"],
            "sanitari": ["vasi_wc", "bidet"],
            "lavabi": ["lavabo"],
            "rubinetteria": ["miscelatori_lavabo", "miscelatori_bidet", "gruppi_doccia"],
            "vasche_bagno": ["vasca_bagno"],
            "piatti_doccia": ["piatto_doccia"],
            "porte_interne": ["porta_interna", "maniglie_porte_interne"],
            "termoarredi": ["termoarredo"]
        }

        for unita_name, unita_data in unita_abitative_data.items():
            capitolato_unita[unita_name] = []
            totale_unita = 0

            for tipologia_id, articoli in scelte_data.items():
                tipologia = Tipologia.query.filter_by(tipologia_id=tipologia_id).first()
                if not tipologia:
                    continue

                tipologia_nome = tipologia.nome.lower()
                campi_quantita = tipologia_quantita_map.get(tipologia_nome, [])
                totale_tipologia = 0
                prodotti_dettagli = []

                for codice_articolo in articoli:
                    prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
                    if not prodotto:
                        continue

                    quantita = sum(unita_data.get(campo, 0) for campo in campi_quantita)
                    costo = 0
                    if prodotto.prezzo_unitario:
                        costo = quantita * float(prodotto.prezzo_unitario)
                    elif prodotto.prezzo_mq:
                        costo = quantita * float(prodotto.prezzo_mq)

                    totale_tipologia += costo
                    prodotti_dettagli.append({
                        "codice_articolo": prodotto.codice_articolo,
                        "nome": prodotto.collezione,
                        "brand": prodotto.brand,
                        "categoria": prodotto.categoria,
                        "quantita": quantita,
                        "costo": round(costo, 2)
                    })

                capitolato_unita[unita_name].append({
                    "tipologia": tipologia_nome.upper(),
                    "prodotti": prodotti_dettagli,
                    "totale_tipologia": round(totale_tipologia, 2)
                })

                totale_unita += totale_tipologia

            capitolato_unita[unita_name].append({"totale_unita": round(totale_unita, 2)})
            totale_finale += totale_unita

        risultato = {
            "capitolato_per_unita": capitolato_unita,
            "totale_finale": round(totale_finale, 2)
        }

        return jsonify(risultato), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
## 10. Endpoint per visualizzare l'incidenza dei costi.
@app.route('/incidenza-costi', methods=['GET'])
def incidenza_costi():
    try:
        if not intervento_data or not scelte_data:
            return jsonify({"error": "Dati mancanti per calcolare l'incidenza dei costi"}), 400

        tipologia_quantita_map = {
            "pavimento_legno": ["sup_pavimenti_legno"],
            "pavimento": ["sup_pavimenti"],
            "rivestimenti": ["sup_rivestimenti"],
            "sanitari": ["vasi_wc", "bidet"],
            "lavabi": ["lavabo"],
            "rubinetteria": ["miscelatori_lavabo", "miscelatori_bidet", "gruppi_doccia"],
            "vasche_bagno": ["vasca_bagno"],
            "piatti_doccia": ["piatto_doccia"],
            "porte_interne": ["porta_interna", "maniglie_porte_interne"],
            "termoarredi": ["termoarredo"]
        }

        incidenza_percentuale = {}
        totale_finale = 0
        costi_tipologie = {}

        for tipologia_id, articoli in scelte_data.items():
            tipologia = Tipologia.query.filter_by(tipologia_id=tipologia_id).first()
            if not tipologia:
                continue

            tipologia_nome = tipologia.nome.lower()
            campi_quantita = tipologia_quantita_map.get(tipologia_nome, [])
            totale_tipologia = 0

            for codice_articolo in articoli:
                prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
                if not prodotto:
                    continue

                quantita = sum(intervento_data.get(campo, 0) for campo in campi_quantita)
                costo = 0
                if prodotto.prezzo_unitario:
                    costo = quantita * float(prodotto.prezzo_unitario)
                elif prodotto.prezzo_mq:
                    costo = quantita * float(prodotto.prezzo_mq)

                totale_tipologia += costo

            costi_tipologie[tipologia.nome] = round(totale_tipologia, 2)
            totale_finale += totale_tipologia

        # Calcola l'incidenza percentuale
        for tipologia_nome, costo in costi_tipologie.items():
            incidenza_percentuale[tipologia_nome] = round((costo / totale_finale) * 100, 2) if totale_finale > 0 else 0

        # Creazione del grafico a torta
        try:
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64

            labels = list(incidenza_percentuale.keys())
            sizes = list(incidenza_percentuale.values())

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Assicurarsi che il grafico sia un cerchio

            # Salva il grafico come immagine in memoria
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
        except Exception as e:
            graph_url = None

        risultato = {
            "incidenza_percentuale": incidenza_percentuale,
            "totale_finale": round(totale_finale, 2),
            "grafico": f"data:image/png;base64,{graph_url}" if graph_url else None
        }

        return jsonify(risultato), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500    


## 11. Salva pdf su S3
@app.route('/salva-pdf', methods=['POST'])
def salva_pdf():
    try:
        # Richiama le funzioni e accedi al primo elemento della tuple
        intervento_data = visualizza_dati()[0]  # Accedi solo ai dati
        capitolato = visualizza_capitolato_finale()[0]  # Accedi solo ai dati
        capitolato_unita = visualizza_capitolato_unita_abitative()[0]  # Accedi solo ai dati
        incidenza_data = incidenza_costi()[0]  # Accedi solo ai dati

        incidenza = incidenza_data["incidenza_percentuale"]
        totale_finale = incidenza_data["totale_finale"]

        class PDF(FPDF):
            def header(self):
                self.image('logo.png', 10, 8, 33)
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Capitolato Finale', border=False, ln=True, align='C')
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

        # Creazione del PDF
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Dati dell'intervento immobiliare
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Scheda Intervento Immobiliare', ln=True)
        pdf.set_font('Arial', '', 12)
        for key, value in intervento_data["intervento"].items():
            pdf.cell(0, 10, f'{key.capitalize()}: {value}', ln=True)

        # Capitolato finale
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Capitolato Elegance', ln=True)
        pdf.set_font('Arial', '', 12)
        for item in capitolato["capitolato"]:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'Tipologia: {item["tipologia"]}', ln=True)
            pdf.set_font('Arial', '', 12)
            for prodotto in item['prodotti']:
                pdf.cell(0, 10, f'- {prodotto["nome"]} ({prodotto["brand"]}): {prodotto["costo"]} EUR', ln=True)
            pdf.cell(0, 10, f'Totale Tipologia: {item["totale_tipologia"]} EUR', ln=True)

        # Capitolato per unità abitative
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Capitolato per Unita Abitative', ln=True)
        for unita, dettagli in capitolato_unita["capitolato_per_unita"].items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'Unita: {unita}', ln=True)
            pdf.set_font('Arial', '', 12)
            for dettaglio in dettagli:
                pdf.cell(0, 10, f'Tipologia: {dettaglio["tipologia"]}', ln=True)
                for prodotto in dettaglio['prodotti']:
                    pdf.cell(0, 10, f'- {prodotto["nome"]} ({prodotto["brand"]}): {prodotto["costo"]} EUR', ln=True)
                pdf.cell(0, 10, f'Totale Tipologia: {dettaglio["totale_tipologia"]} EUR', ln=True)

        # Incidenza costi
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Incidenza Costi', ln=True)
        pdf.set_font('Arial', '', 12)
        for tipologia, percentuale in incidenza.items():
            pdf.cell(0, 10, f'{tipologia}: {percentuale}%', ln=True)
        pdf.cell(0, 10, f'Totale Finale: {totale_finale} EUR', ln=True)

        # Salva il PDF in memoria
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        # Caricamento su Amazon S3
        s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY'), aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
        bucket_name = os.getenv('S3_BUCKET_NAME')
        file_name = f"capitolato_finale_{intervento_data['intervento'].get('nome_intervento', 'default')}.pdf"

        s3.upload_fileobj(pdf_output, bucket_name, file_name)

        return jsonify({"message": "PDF generato e caricato su S3", "file_name": file_name}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500



## 12. Endpoint per visualizzare la dashboard finale con i dati dell'intervento e i capitolati salvati
@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        # Controlla che i dati di intervento siano disponibili
        if not intervento_data:
            return jsonify({"error": "Dati intervento mancanti"}), 400

        # Connessione a S3 per ottenere i file PDF salvati
        s3 = boto3.client('s3', aws_access_key_id='YOUR_AWS_ACCESS_KEY', aws_secret_access_key='YOUR_AWS_SECRET_KEY')
        bucket_name = 'YOUR_BUCKET_NAME'
        try:
            response = s3.list_objects_v2(Bucket=bucket_name)
            files = response.get('Contents', [])
            capitolati = [
                {
                    "nome_file": file['Key'],
                    "url": f"https://{bucket_name}.s3.amazonaws.com/{file['Key']}"
                }
                for file in files
            ]
        except Exception as e:
            capitolati = []

        # Struttura dei dati per la dashboard
        dashboard_data = {
            "dati_intervento": intervento_data,
            "capitolati_salvati": capitolati
        }

        return jsonify(dashboard_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500    

##############
## FUNZIONI ##
##############

# Funzione generica per calcolare il capitolato
def calcola_capitolato():
    try:
        global intervento_data, scelte_data

        if not intervento_data or not scelte_data:
            return {"error": "Dati mancanti per generare il capitolato finale"}

        tipologia_quantita_map = {
            "pavimento_legno": ["sup_pavimenti_legno"],
            "pavimento": ["sup_pavimenti"],
            "rivestimenti": ["sup_rivestimenti"],
            "sanitari": ["vasi_wc", "bidet"],
            "lavabi": ["lavabo"],
            "rubinetteria": ["miscelatori_lavabo", "miscelatori_bidet", "gruppi_doccia"],
            "vasche_bagno": ["vasca_bagno"],
            "piatti_doccia": ["piatto_doccia"],
            "porte_interne": ["porta_interna", "maniglie_porte_interne"],
            "termoarredi": ["termoarredo"]
        }

        capitolato = []
        totale_finale = 0

        for tipologia_id, articoli in scelte_data.items():
            tipologia = Tipologia.query.filter_by(tipologia_id=tipologia_id).first()
            if not tipologia:
                continue

            tipologia_nome = tipologia.nome.lower()
            campi_quantita = tipologia_quantita_map.get(tipologia_nome, [])
            totale_tipologia = 0

            for codice_articolo in articoli:
                prodotto = Prodotto.query.filter_by(codice_articolo=codice_articolo).first()
                if not prodotto:
                    continue

                quantita = sum(intervento_data.get(campo, 0) for campo in campi_quantita)
                costo = 0
                if prodotto.prezzo_unitario:
                    costo = quantita * float(prodotto.prezzo_unitario)
                elif prodotto.prezzo_mq:
                    costo = quantita * float(prodotto.prezzo_mq)

                totale_tipologia += costo
                capitolato.append({
                    "tipologia": tipologia_nome.upper(),
                    "prodotto": {
                        "codice_articolo": prodotto.codice_articolo,
                        "nome": prodotto.collezione,
                        "categoria": prodotto.categoria,                        
                        "brand": prodotto.brand,
                        "quantita": quantita,
                        "costo": round(costo, 2)
                    }
                })

            totale_finale += totale_tipologia

        return {
            "capitolato": capitolato,
            "totale_finale": round(totale_finale, 2)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)
