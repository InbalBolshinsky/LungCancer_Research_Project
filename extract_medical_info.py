import fitz  # PyMuPDF
import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI() 

docs_folder = "docs"
output_csv = "patients_summary.csv"

# === Define the medical categories (columns) ===
columns = [
    "Patient Full Name", "ID Number", "Age", "Gender", "Address", "Contact Numbers",
    "Oncological Diagnoses", "Chief Complaint", "Medical History", "Family History",
    "Medications", "Surgeries", "Body Measurements", "Imaging Results", "Blood Test Results",
    "Current Treatments", "Recommendations"
]

# === Step 1: Extract text from PDFs ===
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# === Step 2: Extract structured fields from text (in Hebrew) ===
def extract_fields_from_text(text):
    system_prompt = (
        "אתה עוזר חכם שמוציא מידע מובנה מתוך דוחות רפואיים בעברית.\n"
        f"השדות הדרושים: {columns}\n"
        "החזר JSON בלבד, בלי הסברים.\n"
        "אם אין מידע בשדה, שים מחרוזת ריקה.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    content = response.choices[0].message.content

    if not content.strip():
        raise ValueError("⚠️ Empty response from OpenAI! Maybe the PDF text is too long.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error! Full content was:")
        print(content)
    raise e


# === Step 3: Translate structured fields ===
def translate_fields(hebrew_dict):
    system_prompt = (
        "You are a professional medical translator.\n"
        "Translate the field values in the following JSON from Hebrew to English without changing the JSON structure.\n"
        "Return only the translated JSON.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(hebrew_dict, ensure_ascii=False)}
        ],
        temperature=0
    )
    content = response.choices[0].message.content

    if not content.strip():
        raise ValueError("⚠️ Empty response from OpenAI! Maybe the PDF text is too long.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error! Full content was:")
        print(content)
    raise e


# === Step 4: Process all PDFs ===
def process_all_pdfs(docs_folder):
    patient_records = []
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(docs_folder, pdf_file)
        hebrew_text = extract_text_from_pdf(pdf_path)
        fields_hebrew = extract_fields_from_text(hebrew_text)
        fields_english = translate_fields(fields_hebrew)
        patient_records.append(fields_english)

    return patient_records

# === Step 5: Save to CSV ===
def save_to_csv(patient_records, output_csv):
    df = pd.DataFrame(patient_records, columns=columns)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

# === Main run ===
if __name__ == "__main__":
    if not openai.api_key:
        raise ValueError("⚠️ OPENAI_API_KEY is missing! Please check your .env file.")
    
    patient_data = process_all_pdfs(docs_folder)
    save_to_csv(patient_data, output_csv)
    print(f"✅ Done! Output saved to {output_csv}")
