from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
import boto3
import os
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Carica le variabili di ambiente dal file .env
load_dotenv()

# Configurazione dell'app Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # URL del database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME')
app.config['AWS_ACCESS_KEY'] = os.getenv('AWS_ACCESS_KEY')
app.config['AWS_SECRET_KEY'] = os.getenv('AWS_SECRET_KEY')

### DATA BASE ###
# Configurazione del database
db = SQLAlchemy(app)

## PRODOTTO
class Prodotto(db.Model):
    __tablename__ = 'Prodotto'
    
    codice_articolo = db.Column(db.String(50), primary_key=True)  # Chiave primaria
    brand = db.Column(db.String(100), nullable=False)  # Marca del prodotto
    finitura = db.Column(db.String(100), nullable=False)  # Tipo di finitura
    posa = db.Column(db.String(100), default=None)  # Tipo di posa (opzionale)
    collezione = db.Column(db.String(100), nullable=False)  # Nome della collezione
    dimensione = db.Column(db.String(50), nullable=False)  # Dimensione del prodotto
    prezzo_unitario = db.Column(db.Numeric(10, 2), default=None)  # Prezzo unitario
    prezzo_mq = db.Column(db.Numeric(10, 2), default=None)  # Prezzo per metro quadro
    colore = db.Column(db.String(50), nullable=True)  # Colore del prodotto
    spessore = db.Column(db.Numeric(5, 2), default=None)  # Spessore del prodotto (opzionale)
    tipologia_id = db.Column(db.Integer, db.ForeignKey('Tipologia.tipologia_id'), nullable=False)  # Foreign Key verso Tipologia
    documento = db.Column(db.String(255), nullable=True)  # Percorso del documento associato
    immagine = db.Column(db.String(255), nullable=True)  # Percorso dell'immagine associata
    nota = db.Column(db.String(500), default=None)  # Note aggiuntive sul prodotto
    peso = db.Column(db.Numeric(5, 2), default=None)  # Peso del prodotto (opzionale)
    categoria = db.Column(db.String(100), default=None)  # Categoria del prodotto

    # Constraint CHECK
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
    
    tipologia_id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # ID univoco per ogni tipologia
    nome = db.Column(db.String(100), nullable=False)  # Nome della tipologia, campo obbligatorio

    # Relazione con Prodotto
    prodotti = db.relationship('Prodotto', backref='tipologia', lazy=True)

    def __repr__(self):
        return f"<Tipologia(tipologia_id={self.tipologia_id}, nome={self.nome})>"
### FINE DATA BASE ###

### API ###
# 1. Questa API restituisce un JSON con tutti i prodotti.
@app.route('/prodotti', methods=['GET'])
def get_prodotti():
    prodotti = Prodotto.query.all()  # Recupera tutti i prodotti dal database
    result = []
    for p in prodotti:
        result.append({
            "codice_articolo": p.codice_articolo,
            "brand": p.brand,
            "finitura": p.finitura,
            "dimensione": p.dimensione,
            "prezzo_mq": float(p.prezzo_mq) if p.prezzo_mq else None,
            "categoria": p.categoria,
            "colore": p.colore
        })
    return jsonify(result), 200

# Questa API consente al cliente di inviare i dati compilati della 
# "Scheda dati intervento immobiliare".
@app.route('/capitolato', methods=['POST'])
def create_capitolato():
    try:
        data = request.get_json()

        # Leggi i campi della scheda
        nome_intervento = data['nome_intervento']
        via = data['via']
        cap = data['cap']
        citta = data['citta']
        cliente = data['cliente']
        numero_unita = data['numero_unita']
        numero_piani = data['numero_piani']

        superfici = {
            "pavimenti_zona_giorno": data['sup_pavimenti_zona_giorno'],
            "pavimenti_bagni": data['sup_pavimenti_bagni'],
            "rivestimenti_bagni": data['sup_rivestimenti_bagni']
        }

        componenti = {
            "vasi_wc": data['vasi_wc'],
            "bidet": data['bidet'],
            "lavabi": data['lavabi'],
            "piatti_doccia": data['piatti_doccia'],
            "vasca_bagno": data['vasca_bagno'],
            "termoarredi": data['termoarredi'],
            "miscelatori_lavabo": data['miscelatori_lavabo'],
            "miscelatori_bidet": data['miscelatori_bidet'],
            "gruppi_doccia": data['gruppi_doccia'],
            "porte_interne": data['porte_interne'],
            "maniglie_porte_interne": data['maniglie_porte_interne']
        }
        return jsonify({"message": "Capitolato creato con successo"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Questa API genera i dati dinamicamente in base al numero di unità abitative scelto.
# "Scheda dati unità abitative" 
@app.route('/unita-abitative', methods=['POST'])
def generate_unita_abitative():
    try:
        # Ricevi il numero di unità abitative dal frontend
        data = request.get_json()
        numero_unita = data['numero_unita']

        # Verifica che il numero sia valido (tra 0 e 4)
        if numero_unita < 0 or numero_unita > 4:
            return jsonify({"error": "Il numero di unità abitative deve essere compreso tra 0 e 4"}), 400

        # Genera la struttura delle unità abitative
        unita_abitative = []
        for i in range(1, numero_unita + 1):
            unita_abitative.append({
                "unita": f"A0{i}",  # Nome dell'unità (A01, A02, ecc.)
                "piano": None,
                "superficie": None,
                "sup_pavimenti_zona_giorno": None,
                "sup_pavimenti_bagni": None,
                "sup_rivestimenti_bagni": None,
                "vasi_wc": None,
                "bidet": None,
                "lavabi": None,
                "piatti_doccia": None,
                "vasca_bagno": None,
                "termoarredi": None,
                "miscelatori_lavabo": None,
                "miscelatori_bidet": None,
                "gruppi_doccia": None,
                "porte_interne": None,
                "maniglie_porte_interne": None
            })

        # Restituisci la struttura al frontend
        return jsonify({"unita_abitative": unita_abitative}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/salva-capitolato', methods=['POST'])
def salva_capitolato():
    try:
        # Ottieni i dati inviati dal frontend
        data = request.get_json()

        # Dati generali
        nome_intervento = data['nome_intervento']
        via = data['via']
        cap = data['cap']
        citta = data['citta']
        cliente = data['cliente']
        numero_unita = data['numero_unita']
        unita_abitative = data['unita_abitative']  # Lista con i dettagli delle unità

        # Genera il file PDF
        file_name = f"{nome_intervento.replace(' ', '_')}_capitolato.pdf"
        generate_pdf(data, file_name)

        # Nome del bucket S3 e percorso remoto
        bucket_name = "matterofdesign-capitolati"
        s3_file_name = f"capitolati/{file_name}"

        # Carica il PDF su S3
        if upload_to_s3(file_name, bucket_name, s3_file_name):
            # Elimina il file PDF locale per non occupare spazio sul server
            os.remove(file_name)

            # Genera un URL firmato per scaricare il PDF
            s3 = boto3.client(
                's3',
                aws_access_key_id=app.config['AWS_ACCESS_KEY'],
                aws_secret_access_key=app.config['AWS_SECRET_KEY']
            )
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_file_name},
                ExpiresIn=3600  # URL valido per 1 ora
            )

            # Restituisci il percorso remoto al frontend
            return jsonify({
                "message": "Capitolato generato e caricato su S3 con successo.",
                "s3_url": url  # URL firmato
            }), 201
        else:
            return jsonify({"error": "Errore durante il caricamento su S3"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500



### FUNZIONI ###
# 1. Funzione per Generare il PDF del capitolato
def generate_pdf(data, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Dati generali
    pdf.cell(200, 10, txt="Capitolato", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Nome Intervento: {data['nome_intervento']}", ln=True)
    pdf.cell(200, 10, txt=f"Cliente: {data['cliente']}", ln=True)
    pdf.cell(200, 10, txt=f"Indirizzo: {data['via']} - {data['cap']} - {data['citta']}", ln=True)

    # Dati unità abitative
    pdf.cell(200, 10, txt="Unità Abitative", ln=True)
    for unita in data['unita_abitative']:
        pdf.cell(200, 10, txt=f"Unità: {unita['unita']}", ln=True)
        pdf.cell(200, 10, txt=f"  Piano: {unita['piano']}, Superficie: {unita['superficie']} mq", ln=True)
        pdf.cell(200, 10, txt=f"  Sup. Pavimenti Zona Giorno: {unita['sup_pavimenti_zona_giorno']} mq", ln=True)
        pdf.cell(200, 10, txt=f"  Sup. Pavimenti Bagni: {unita['sup_pavimenti_bagni']} mq", ln=True)
        pdf.cell(200, 10, txt=f"  Sup. Rivestimenti Bagni: {unita['sup_rivestimenti_bagni']} mq", ln=True)
        pdf.cell(200, 10, txt=f"  Vasi WC: {unita['vasi_wc']}, Bidet: {unita['bidet']}, Lavabi: {unita['lavabi']}", ln=True)
        pdf.cell(200, 10, txt=f"  Piatti Doccia: {unita['piatti_doccia']}, Vasca da Bagno: {unita['vasca_bagno']}", ln=True)
        pdf.cell(200, 10, txt=f"  Termoarredi: {unita['termoarredi']}, Miscelatori Lavabo: {unita['miscelatori_lavabo']}", ln=True)
        pdf.cell(200, 10, txt=f"  Miscelatori Bidet: {unita['miscelatori_bidet']}, Gruppi Doccia: {unita['gruppi_doccia']}", ln=True)
        pdf.cell(200, 10, txt=f"  Porte Interne: {unita['porte_interne']}, Maniglie: {unita['maniglie_porte_interne']}", ln=True)

    # Salva il PDF
    pdf.output(file_path)


# 2. Funzione per caricare il file su S3
def upload_to_s3(file_path, bucket_name, s3_file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id=app.config['AWS_ACCESS_KEY'],
        aws_secret_access_key=app.config['AWS_SECRET_KEY']
    )
    try:
        s3.upload_file(file_path, bucket_name, s3_file_name)
        return True
    except Exception as e:
        print(f"Errore durante il caricamento su S3: {e}")
        return False
