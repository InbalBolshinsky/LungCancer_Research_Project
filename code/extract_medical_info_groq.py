import os
import fitz  # PyMuPDF
import pandas as pd
import groq
import json
import re
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("⚠️ GROQ_API_KEY is missing! Please check your .env file.")

client = groq.Groq(api_key=groq_api_key)

docs_folder = "../docs"            # PDFs input folder
output_folder = "../output"         # Results output folder
os.makedirs(output_folder, exist_ok=True)

output_csv = os.path.join(output_folder, "patients_summary.csv")
redaction_log_file = os.path.join(output_folder, "redaction_report.txt")

# === Define fields ===
gpt_columns = [
    "Date of Birth", "Stage at Diagnosis", "Primary tumor location", "Grade",
    "Date of Diagnosis", "Date of Surgery", "surgery type", "metastatic 1st line",
    "Drugs", "Best response", "Date of Death", "Canncer Family History First degree relatives",
    "Other Diseases", "Date of Metastatic Spread Outcome", "Other Malignancies"
]

final_columns = ["ID", *gpt_columns]

# === Text anonymization ===
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

# === Split text into chunks ===
def split_text_into_chunks(text, max_chars=2000):
    """Split text into small chunks (~2000 chars) to avoid exceeding token limits."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# === Summarize text safely ===
def summarize_text(text):
    if len(text) > 12000:
        print("⚡ Text too large. Splitting into smaller chunks for summarization...")
        chunks = split_text_into_chunks(text, max_chars=2000)

        summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"🧩 Summarizing chunk {idx + 1}/{len(chunks)}...")
            summary = _summarize_single_chunk(chunk)
            summaries.append(summary)

        combined_summary = "\n\n".join(summaries)
        return combined_summary
    else:
        return _summarize_single_chunk(text)

def _summarize_single_chunk(chunk):
    system_prompt = (
        "You are a medical summarization assistant.\n"
        "Summarize the following medical report into a concise version that preserves all medically important facts.\n"
        "Keep diagnoses, dates, treatments, family history, surgeries, and important events.\n"
        "Do not invent or add information. Return the summary as plain text.\n"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ],
        temperature=0
    )

    summary = response.choices[0].message.content.strip()
    if not summary:
        raise ValueError("⚠️ Summarization failed, empty response!")

    return summary

# === Extract structured fields from text ===
def extract_fields_from_text(text):
    if len(text) > 12000:  # Rough threshold
        print("⚡ Summarizing text because it's too long...")
        text = summarize_text(text)

    system_prompt = (
        "אתה עוזר חכם שמוציא מידע מובנה מתוך דוחות רפואיים בעברית.\n"
        f"השדות הדרושים: {gpt_columns}\n"
        "החזר JSON בלבד, בלי הסברים, ובלי סימוני ```json.\n"
        "אם אין מידע בשדה, שים מחרוזת ריקה.\n"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("⚠️ Empty response from Groq during extraction!")

    content = re.sub(r'^```json|```$', '', content, flags=re.MULTILINE).strip()
    content = re.sub(r',([\s\n]*[}\]])', r'\1', content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error! Full content was:")
        print(content)
        raise e

# === Translate structured fields to English ===
def translate_fields(hebrew_dict):
    system_prompt = (
        "You are a professional medical translator.\n"
        "Translate the field values in the following JSON from Hebrew to English without changing the JSON structure.\n"
        "Return only the translated JSON.\n"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(hebrew_dict, ensure_ascii=False)}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("⚠️ Empty response from Groq during translation!")

    content = re.sub(r'^```json|```$', '', content, flags=re.MULTILINE).strip()
    content = re.sub(r',([\s\n]*[}\]])', r'\1', content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error during translation! Full content was:")
        print(content)
        raise e

# === Process all PDFs ===
def process_all_pdfs(docs_folder):
    patient_records = []
    redaction_reports = []

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(docs_folder, pdf_file)

        hebrew_text = extract_text_from_pdf(pdf_path)
        hebrew_text, redactions = anonymize_text(hebrew_text)

        fields_hebrew = extract_fields_from_text(hebrew_text)
        fields_english = translate_fields(fields_hebrew)
        patient_records.append(fields_english)

        redaction_reports.append((pdf_file, redactions))

    return patient_records, redaction_reports

# === Assign anonymized IDs ===
def assign_patient_ids(patient_records):
    for idx, record in enumerate(patient_records, start=1):
        record["ID"] = f"PAT{idx:03d}"
    return patient_records

# === Save results to CSV ===
def save_to_csv(patient_records, output_csv):
    df = pd.DataFrame(patient_records, columns=final_columns)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

# === Save redaction report ===
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
