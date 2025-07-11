"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch_directml
import torch
import re
import time

app = FastAPI()

# 🔓 CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 DirectML aktivieren
device = torch_directml.device()
backend = "DirectML (Windows GPU-Beschleunigung)"
torch.set_float32_matmul_precision("medium")

print(f"💡 PyTorch verwendet: {backend}")
print(f"💻 Modell läuft auf: {device}")

# 📁 HTML-Textmodell laden
text_model_path = "./SatiremodelJelleTest"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
text_model = AutoModelForCausalLM.from_pretrained(text_model_path).to(device)
text_model.eval()
text_tokenizer.pad_token = text_tokenizer.eos_token

# 📁 MBart-Modell für Promptgenerierung laden
image_model_path = "./bildgenerator-bart"  # ⛔️ NICHT Colab-spezifisch
img_tokenizer = MBart50TokenizerFast.from_pretrained(image_model_path)
img_model = MBartForConditionalGeneration.from_pretrained(image_model_path)
img_tokenizer.src_lang = "de_DE"
forced_bos_token_id = img_tokenizer.lang_code_to_id["de_DE"]

# 🛑 HTML-Stoppkriterium
class StopOnHTML(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="</html>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded

# 🧠 Promptgenerator
def generate_image_prompt(mail_text, max_new_tokens=128):
    inputs = img_tokenizer(
        mail_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )
    output = img_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=2
    )
    return img_tokenizer.decode(output[0], skip_special_tokens=True)

# 🔁 API-Endpunkt
@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    input_ids = text_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    stop_criteria = StoppingCriteriaList([StopOnHTML(text_tokenizer)])

    start_time = time.time()

    with torch.no_grad():
        output = text_model.generate(
            input_ids=input_ids,
            max_length=1024,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            do_sample=True,
            pad_token_id=text_tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            stopping_criteria=stop_criteria,
            num_return_sequences=1
        )

    duration = time.time() - start_time
    print(f"🕒 Dauer: {duration:.2f} Sek.")
    print(f"📦 Gerät: {next(text_model.parameters()).device}")
    print(f"⚙️ Backend: {backend}")

    generated_text = text_tokenizer.decode(output[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    # 📤 Bildprompt erzeugen
    bildprompt = generate_image_prompt(generated_text)

    # 🏷️ Optional: Titel extrahieren
    title_match = re.search(r"<title>(.*?)</title>", generated_text, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else None
    print(bildprompt)
    return {
        "response": generated_text,
        "image_prompt": bildprompt,
    }
"""
"""
def dupliziere_zeilen(input_datei, output_datei):
    with open(input_datei, 'r', encoding='utf-8') as infile:
        zeilen = infile.readlines()

    with open(output_datei, 'w', encoding='utf-8') as outfile:
        for zeile in zeilen:
            outfile.write(zeile)
            outfile.write(zeile)

# Beispielaufruf
dupliziere_zeilen('input.txt', 'input2.txt')

import json

def ersetze_targets(input_datei, target_datei, output_datei):
    # Lade alle Zieltexte (Targets), getrennt durch Semikolon
    with open(target_datei, 'r', encoding='utf-8') as t_file:
        target_inhalt = t_file.read()
        targets = [t.strip() for t in target_inhalt.split(';') if t.strip()]

    # Lade alle JSON-Zeilen aus input2.txt
    with open(input_datei, 'r', encoding='utf-8') as infile:
        zeilen = infile.readlines()

    # Prüfe, ob Anzahl Targets mit Zeilenanzahl übereinstimmt
    if len(targets) != len(zeilen):
        print(f"Fehler: Anzahl Targets ({len(targets)}) stimmt nicht mit Anzahl Zeilen ({len(zeilen)}) überein.")
        return

    # Ersetze "target": "" durch passenden Text und speichere in neuer Datei
    with open(output_datei, 'w', encoding='utf-8') as outfile:
        for zeile, target_text in zip(zeilen, targets):
            try:
                json_obj = json.loads(zeile)
                json_obj["target"] = target_text
                neue_zeile = json.dumps(json_obj, ensure_ascii=False)
                outfile.write(neue_zeile + "\n")
            except json.JSONDecodeError:
                print(f"Ungültige JSON-Zeile: {zeile}")

# Beispielaufruf
ersetze_targets("input2.txt", "target.txt", "input_final.txt")
"""

"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList
)
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch_directml
import torch
import re
import base64
from io import BytesIO
from email.mime.text import MIMEText
import smtplib
from googletrans import Translator

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelle & Geräte
device = torch_directml.device()
torch.set_float32_matmul_precision("medium")

# Textmodell
text_model_path = "./SatiremodelJelleTest"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
text_model = AutoModelForCausalLM.from_pretrained(text_model_path).to(device)
text_model.eval()
text_tokenizer.pad_token = text_tokenizer.eos_token

# Bildprompt-Modell
image_model_path = "./bildgenerator-bart"
img_tokenizer = MBart50TokenizerFast.from_pretrained(image_model_path)
img_model = MBartForConditionalGeneration.from_pretrained(image_model_path)
img_tokenizer.src_lang = "de_DE"
forced_bos_token_id = img_tokenizer.lang_code_to_id["de_DE"]

# Bildgenerierung
image_pipe = StableDiffusionXLPipeline.from_pretrained(
    "./BildgeneratorKI", torch_dtype=torch.float32
)
image_pipe.safety_checker = None
image_pipe.enable_attention_slicing()

# HTML-Stoppkriterium
class StopOnHTML(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="</html>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded

# 🔹 Nur HTML generieren
@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    input_ids = text_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    stop_criteria = StoppingCriteriaList([StopOnHTML(text_tokenizer)])

    with torch.no_grad():
        output = text_model.generate(
            input_ids=input_ids,
            max_length=1024,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            do_sample=True,
            pad_token_id=text_tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            stopping_criteria=stop_criteria,
            num_return_sequences=1
        )

    generated_text = text_tokenizer.decode(output[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return {
        "response": generated_text
    }

# 🔹 Bildprompt generieren
@app.post("/generate_image_prompt")
async def generate_image_prompt_api(request: Request):
    data = await request.json()
    html_text = data.get("prompt", "")

    inputs = img_tokenizer(html_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    output = img_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=128,
        num_beams=4,
        no_repeat_ngram_size=2
    )

    german_prompt = img_tokenizer.decode(output[0], skip_special_tokens=True)
    english_prompt = Translator().translate(german_prompt, src='de', dest='en').text

    return {
        "image_prompt": german_prompt,
        "image_prompten": english_prompt
    }

# 🔹 Bild aus Prompt generieren
@app.post("/generate_image")
async def generate_image_route(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    image = image_pipe(prompt).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return {"image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}

# 🔹 Mailversand
class EmailRequest(BaseModel):
    to: str
    html: str

@app.post("/send_email")
async def send_email(data: EmailRequest):
    sender_email = "jelle.alexander.walter.pichl@gmail.com"
    sender_pass = "otpf krsi enqi nxgk"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEText(data.html, "html")
    subject_match = re.search(r"<title>(.*?)</title>", data.html, re.IGNORECASE)
    msg["Subject"] = subject_match.group(1) if subject_match else "🧅 Satirische Spam-Mail"
    msg["From"] = sender_email
    msg["To"] = data.to

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.send_message(msg)
        return {"message": f"Mail erfolgreich gesendet an {data.to}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Senden: {e}")
"""
"""
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch_directml
import torch

# 📦 Modell laden
device = torch_directml.device()
torch.set_float32_matmul_precision("medium")

text_model_path = "./SatiremodelJelleTest"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
text_model = AutoModelForCausalLM.from_pretrained(text_model_path).to(device)
text_model.eval()
text_tokenizer.pad_token = text_tokenizer.eos_token
print("✅ Modell geladen.")

# 🛑 HTML-Stoppkriterium
class StopOnHTML(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="</html>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded

# 📌 Kategorien & Prompts
prompts = {
    "fantasy": "Generiere eine satirische Spam-Mail in der Fantasywelt",
    "tiere": "Generiere eine satirische Spam-Mail in der Tierwelt",
    "buero": "Generiere eine satirische Spam-Mail im Büroalltag",
    "weltall": "Generiere eine satirische Spam-Mail im Weltall",
    "pflanzen": "Generiere eine satirische Spam-Mail im Pflanzenreich",
    "zufall": "Generiere eine satirische Spam-Mail"
}

# 📂 Zielordner vorbereiten
output_dir = Path("pre_generated")
output_dir.mkdir(parents=True, exist_ok=True)

# 🔁 Generierungsfunktion
def generate(prompt, iterations=100):
    results = []
    for i in range(iterations):
        input_ids = text_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        stop_criteria = StoppingCriteriaList([StopOnHTML(text_tokenizer)])

        with torch.no_grad():
            output = text_model.generate(
                input_ids=input_ids,
                max_length=1024,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                do_sample=True,
                pad_token_id=text_tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=5,
                stopping_criteria=stop_criteria,
                num_return_sequences=1
            )

        generated_text = text_tokenizer.decode(output[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        results.append(generated_text)
        print(f"[{prompt[:20]}...] Beispiel {i+1}/{iterations} ✅")
    return results

# 📊 Verteilung für 1200 Texte mit Gewichtung 1-1-1-1-1-3
category_weights = {
    "fantasy": 150,
    "tiere": 150,
    "buero": 150,
    "weltall": 150,
    "pflanzen": 150,
    "zufall": 450  # dreifach
}

# 🚀 Generieren & Anhängen
for key, iterations in category_weights.items():
    prompt = prompts[key]
    print(f"\n🔹 Generiere Kategorie: {key} ({iterations} Texte)")
    new_mails = generate(prompt, iterations=iterations)
    file_path = output_dir / f"{key}.json"

    # Bestehende Inhalte laden (wenn vorhanden)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_mails = json.load(f)
    else:
        existing_mails = []

    # Anhängen und speichern
    combined_mails = existing_mails + new_mails
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_mails, f, ensure_ascii=False, indent=2)

    print(f"💾 Gespeichert (angehängt): {file_path}")

print("\n✅ Vorproduktion abgeschlossen mit Gewichtung 1-1-1-1-1-3 (insg. 1200 Texte).")
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import random
import re
import smtplib
from email.mime.text import MIMEText

app = FastAPI()

# CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Statische Dateien (style.css, script.js etc.) bereitstellen
app.mount("/static", StaticFiles(directory="."), name="static")

# 🏠 Indexseite (HTML) ausliefern
@app.get("/", response_class=HTMLResponse)
def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>index.html nicht gefunden</h1>", status_code=404)

# 🔹 Text aus vorgenerierter Datei laden und Eintrag löschen

import psycopg2

# Datenbankverbindung einmal global öffnen
def get_db_connection():
    return psycopg2.connect(
        host="dpg-d1nsqu95pdvs738nvou0-a",
        port=5432,
        database="kiundkunstdb",
        user="kiundkunstdb_user",
        password="V9yLU0Nz1j31F4I3b5vJLF0w1QAQbhy7",
        sslmode="require"
    )

@app.post("/generate")
async def generate_from_db(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    prompt_map = {
        "Generiere eine satirische Spam-Mail in der Fantasywelt": "mails_fantasy",
        "Generiere eine satirische Spam-Mail in der Tierwelt": "mails_tiere",
        "Generiere eine satirische Spam-Mail im Büroalltag": "mails_buero",
        "Generiere eine satirische Spam-Mail im Weltall": "mails_weltall",
        "Generiere eine satirische Spam-Mail im Pflanzenreich": "mails_pflanzen",
        "Generiere eine satirische Spam-Mail": "mails_zufall"
    }

    matched = [key for key in prompt_map if key in prompt]
    if not matched:
        raise HTTPException(status_code=400, detail="Ungültiger Prompt")

    table_name = prompt_map[matched[0]]

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Schritt 1: Einen zufälligen Datensatz abrufen (inkl. ID)
        cur.execute(f"SELECT id, html FROM {table_name} ORDER BY RANDOM() LIMIT 1;")
        result = cur.fetchone()

        if not result:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Keine Einträge gefunden.")

        mail_id, html = result

        # Schritt 2: Jetzt diesen Eintrag löschen
        cur.execute(f"DELETE FROM {table_name} WHERE id = %s;", (mail_id,))
        conn.commit()

        cur.close()
        conn.close()

        return {"response": html}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Datenbankfehler: {e}")

# 📧 Mailversand
class EmailRequest(BaseModel):
    to: str
    html: str

@app.post("/send_email")
async def send_email(data: EmailRequest):
    sender_email = "jelle.alexander.walter.pichl@gmail.com"
    sender_pass = "otpf krsi enqi nxgk"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEText(data.html, "html")
    subject_match = re.search(r"<title>(.*?)</title>", data.html, re.IGNORECASE)
    msg["Subject"] = subject_match.group(1) if subject_match else "🧅 Satirische Spam-Mail"
    msg["From"] = sender_email
    msg["To"] = data.to

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.send_message(msg)
        return {"message": f"Mail erfolgreich gesendet an {data.to}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Senden: {e}")

