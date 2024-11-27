from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from flask_cors import CORS
import boto3
import os
import time
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

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

### DATA BASE ###
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

### API ###
# 1. Endpoint per la pagina iniziale di login
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

# 3. Endpoint per ottenere i prodotti
#@app.route('/prodotti', methods=['GET'])
#def get_prodotti():
#    page = request.args.get('page', 1, type=int)
#    per_page = request.args.get('per_page', 10, type=int)
#    prodotti_paginated = Prodotto.query.paginate(page=page, per_page=per_page)
#    result = [{
#        "codice_articolo": p.codice_articolo,
#        "brand": p.brand,
#        "finitura": p.finitura,
#        "posa": p.posa,
#        "collezione": p.collezione,
#        "dimensione": p.dimensione,
#        "prezzo_unitario": float(p.prezzo_unitario) if p.prezzo_unitario else None,
#        "prezzo_mq": float(p.prezzo_mq) if p.prezzo_mq else None,
#        "colore": p.colore,
#        "spessore": float(p.spessore) if p.spessore else None,
#        "tipologia_id": p.tipologia_id,  # Posizionato esattamente come nel database
#        "documento": p.documento,
#        "immagine": p.immagine,
#        "nota": p.nota,
#        "peso": float(p.peso) if p.peso else None,
#        "categoria": p.categoria
#    } for p in prodotti_paginated.items]
#
#    return jsonify({
#        "prodotti": result,
#        "total": prodotti_paginated.total,
#        "pages": prodotti_paginated.pages,
#        "current_page": prodotti_paginated.page
#    }), 200

# 4. Endpoint per consentire al cliente di inviare i dati compilati della 
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
    
# 5. Endpoint per generare i dati dinamicamente in base al numero di unità abitative scelto.
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

# 6. Endpoint per salvare il capitolato su S3
@app.route('/salva-capitolato', methods=['POST'])
def salva_capitolato():
    try:
        data = request.get_json()

        nome_intervento = data['nome_intervento']
        via = data['via']
        cap = data['cap']
        citta = data['citta']
        cliente = data['cliente']
        numero_unita = data['numero_unita']
        unita_abitative = data['unita_abitative']

        file_name = f"{nome_intervento.replace(' ', '_')}_capitolato_{int(time.time())}.pdf"
        generate_pdf(data, file_name)

        bucket_name = app.config['S3_BUCKET_NAME']
        s3_file_name = f"capitolati/{file_name}"

        if upload_to_s3(file_name, bucket_name, s3_file_name):
            os.remove(file_name)

            s3 = boto3.client(
                's3',
                aws_access_key_id=app.config['AWS_ACCESS_KEY'],
                aws_secret_access_key=app.config['AWS_SECRET_KEY']
            )
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_file_name},
                ExpiresIn=3600
            )

            return jsonify({
                "message": "Capitolato generato e caricato su S3 con successo.",
                "s3_url": url
            }), 201
        else:
            return jsonify({"error": "Errore durante il caricamento su S3"}), 500

    except Exception as e:
        app.logger.error(f"Errore in /salva-capitolato: {str(e)}")
        return jsonify({"error": str(e)}), 500


### FUNZIONI ###
# 1. Genera il PDF del capitolato
def generate_pdf(data, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Capitolato", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Nome Intervento: {data['nome_intervento']}", ln=True)
    pdf.cell(200, 10, txt=f"Cliente: {data['cliente']}", ln=True)
    pdf.cell(200, 10, txt=f"Indirizzo: {data['via']} - {data['cap']} - {data['citta']}", ln=True)

    pdf.cell(200, 10, txt="Unità Abitative", ln=True)
    for unita in data['unita_abitative']:
        pdf.cell(200, 10, txt=f"Unità: {unita['unita']}", ln=True)

    pdf.output(file_path)

# 2. Carica il file su S3
def upload_to_s3(file_path, bucket_name, s3_file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id=app.config['AWS_ACCESS_KEY'],
        aws_secret_access_key=app.config['AWS_SECRET_KEY']
    )
    try:
        s3.upload_file(file_path, bucket_name, s3_file_name)
        return True
    except NoCredentialsError as e:
        app.logger.error("Credenziali AWS mancanti")
        return False
    except Exception as e:
        app.logger.error(f"Errore durante il caricamento su S3: {e}")
        return False


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)