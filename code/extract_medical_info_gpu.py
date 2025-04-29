# 🛠 Imports
import os
import fitz  # PyMuPDF
import pandas as pd
import json
import re
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === CONFIG ===
load_dotenv()

docs_folder = "../docs"
output_folder = "../output"
os.makedirs(output_folder, exist_ok=True)

output_csv = os.path.join(output_folder, "patients_summary.csv")
redaction_log_file = os.path.join(output_folder, "redaction_report.txt")

gpt_columns = [
    "Date of Birth", "Stage at Diagnosis", "Primary tumor location", "Grade",
    "Date of Diagnosis", "Date of Surgery", "surgery type", "metastatic 1st line",
    "Drugs", "Best response", "Date of Death", "Canncer Family History First degree relatives",
    "Other Diseases", "Date of Metastatic Spread Outcome", "Other Malignancies"
]
final_columns = ["ID", *gpt_columns]

# === Set device (MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Load Models ===
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# === Inference ===
def infer(model, tokenizer, system_prompt, user_content, max_new_tokens=1024):
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_content}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_reply = decoded.split("<|assistant|>")[-1].strip()
    return assistant_reply

# === Load Specific Models for Tasks ===
print("🔵 Loading Mistral 7B for translation...")
model_translator, tokenizer_translator = load_model("mistralai/Mistral-7B-Instruct-v0.2")  # Best general model

print("🔵 Loading GanjinZero/biogpt-7b-medical-chat for extraction...")
model_extractor, tokenizer_extractor = load_model("GanjinZero/biogpt-7b-medical-chat")  # Medical specialist

# === Text Anonymization ===
def anonymize_text(text):
    redaction_log = []
    ids_found = re.findall(r'\b\d{9}\b', text)
    for id_number in ids_found:
        redaction_log.append(f"Redacted ID: {id_number}")
    text = re.sub(r'\b\d{9}\b', '[REDACTED_ID]', text)

    phones_found = re.findall(r'\b05\d[-\s]?\d{7}\b', text)
    for phone_number in phones_found:
        redaction_log.append(f"Redacted Phone: {phone_number}")
    text = re.sub(r'\b05\d[-\s]?\d{7}\b', '[REDACTED_PHONE]', text)

    if re.search(r'שם\s+פרטי', text):
        redaction_log.append("Redacted First Name")
    text = re.sub(r'(שם\s+פרטי\s*:?\s*)(\S+)', r'\1[REDACTED_FIRST_NAME]', text)

    if re.search(r'שם\s+משפחה', text):
        redaction_log.append("Redacted Last Name")
    text = re.sub(r'(שם\s+משפחה\s*:?\s*)(\S+)', r'\1[REDACTED_LAST_NAME]', text)

    if re.search(r'כתובת', text):
        redaction_log.append("Redacted Address")
    text = re.sub(r'(כתובת\s*:?\s*)([^\n]+)', r'\1[REDACTED_ADDRESS]', text)

    return text, redaction_log

# === Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# === Translate Hebrew to English ===
def translate_text(hebrew_text):
    system_prompt = (
        "You are a professional medical translator.\n"
        "Translate the following medical report text from Hebrew to English, preserving medical terminology and structure.\n"
        "Return only the translated text, no explanations.\n"
    )
    return infer(model_translator, tokenizer_translator, system_prompt, hebrew_text)

# === Extract structured fields from English text ===
def extract_fields(english_text):
    system_prompt = (
        "You are an expert medical assistant.\n"
        "Extract the following fields as a JSON object from the given English medical report text:\n"
        f"{gpt_columns}\n"
        "Return only a valid JSON object, no extra text.\n"
        "If a field is missing, use an empty string.\n"
    )
    content = infer(model_extractor, tokenizer_extractor, system_prompt, english_text)
    content = re.sub(r'^```json|```$', '', content, flags=re.MULTILINE).strip()
    content = re.sub(r',([\s\n]*[}\]])', r'\1', content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error! Full content was:")
        print(content)
        raise e

# === Process all PDFs ===
def process_all_pdfs(docs_folder):
    patient_records = []
    redaction_reports = []

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        print(f"📄 Processing {pdf_file}...")
        pdf_path = os.path.join(docs_folder, pdf_file)

        hebrew_text = extract_text_from_pdf(pdf_path)
        hebrew_text, redactions = anonymize_text(hebrew_text)

        english_text = translate_text(hebrew_text)
        try:
            fields = extract_fields(english_text)
        except json.JSONDecodeError:
            print(f"❌ Skipping {pdf_file} due to extraction error.")
            continue

        patient_records.append(fields)
        redaction_reports.append((pdf_file, redactions))

    return patient_records, redaction_reports

# === Assign anonymized IDs ===
def assign_patient_ids(patient_records):
    for idx, record in enumerate(patient_records, start=1):
        record["ID"] = f"PAT{idx:03d}"
    return patient_records

# === Save results ===
def save_to_csv(patient_records, output_csv):
    df = pd.DataFrame(patient_records, columns=final_columns)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

def save_redaction_report(redaction_reports, redaction_log_file):
    with open(redaction_log_file, "w", encoding="utf-8") as f:
        for filename, redactions in redaction_reports:
            f.write(f"File: {filename}\n")
            for redaction in redactions:
                f.write(f"  - {redaction}\n")
            f.write("\n")

# === Main run ===
if __name__ == "__main__":
    patient_data, redaction_reports = process_all_pdfs(docs_folder)
    patient_data = assign_patient_ids(patient_data)
    save_to_csv(patient_data, output_csv)
    save_redaction_report(redaction_reports, redaction_log_file)

    print(f"✅ Done! CSV saved to {output_csv}")
    print(f"✅ Redaction report saved to {redaction_log_file}")