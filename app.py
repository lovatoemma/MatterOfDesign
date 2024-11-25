from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
import boto3
import os
from dotenv import load_dotenv  # Importa dotenv

# Carica le variabili di ambiente dal file .env
load_dotenv()
print("Database URI:", os.getenv('SQLALCHEMY_DATABASE_URI'))
print("S3 Bucket Name:", os.getenv('S3_BUCKET_NAME'))

# Configurazione dell'app Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME')
app.config['AWS_ACCESS_KEY'] = os.getenv('AWS_ACCESS_KEY')
app.config['AWS_SECRET_KEY'] = os.getenv('AWS_SECRET_KEY')

# Connessione al database
db = SQLAlchemy(app)

# Connessione al database
db = SQLAlchemy(app)

# Modello del database
class Capitolato(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    cliente = db.Column(db.String(100), nullable=False)
    intervento = db.Column(db.String(200), nullable=False)
    s3_url = db.Column(db.String(200), nullable=False)

# Funzione per generare il PDF
def generate_pdf(data, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(200, 10, txt="Capitolato Finiture", ln=True, align="C")
    pdf.ln(10)

    # Dati nel PDF
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Salva il PDF
    pdf.output(file_path)

# Funzione per caricare un file su S3
def upload_to_s3(file_path, file_name):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=app.config['AWS_ACCESS_KEY'],
        aws_secret_access_key=app.config['AWS_SECRET_KEY'],
    )
    s3.upload_file(file_path, app.config['S3_BUCKET_NAME'], file_name)
    return f"https://{app.config['S3_BUCKET_NAME']}.s3.amazonaws.com/{file_name}"

# Endpoint per creare un capitolato
@app.route("/capitolato", methods=["POST"])
def create_capitolato():
    data = request.json

    # Verifica dati
    if not all(key in data for key in ("nome", "cliente", "intervento")):
        return jsonify({"error": "Dati mancanti"}), 400

    # Genera PDF
    file_path = f"{data['nome']}.pdf"
    generate_pdf(data, file_path)

    # Carica su S3
    file_name = f"capitolati/{data['nome']}.pdf"
    s3_url = upload_to_s3(file_path, file_name)

    # Salva nel database
    capitolato = Capitolato(
        nome=data["nome"],
        cliente=data["cliente"],
        intervento=data["intervento"],
        s3_url=s3_url,
    )
    db.session.add(capitolato)
    db.session.commit()

    # Rimuovi PDF locale
    os.remove(file_path)

    return jsonify({"message": "Capitolato creato", "s3_url": s3_url})

# Endpoint per recuperare un capitolato
@app.route("/capitolato/<int:id>", methods=["GET"])
def get_capitolato(id):
    capitolato = Capitolato.query.get_or_404(id)
    return jsonify({
        "nome": capitolato.nome,
        "cliente": capitolato.cliente,
        "intervento": capitolato.intervento,
        "s3_url": capitolato.s3_url,
    })

if __name__ == "__main__":
    app.run(debug=True)

# ENDPOINT:
# POST: http://127.0.0.1:5000/capitolato (per creare un capitolato).
# GET: http://127.0.0.1:5000/capitolato/<id> (per ottenere un capitolato esistente).