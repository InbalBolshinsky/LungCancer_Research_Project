import pandas as pd
import os

# === CONFIG ===
input_file = os.path.join("../output", "patients_summary.csv")  # Input CSV from output/
output_folder = "../output"
output_html = os.path.join(output_folder, "analysis_report.html")

os.makedirs(output_folder, exist_ok=True)

# === Load the data ===
df = pd.read_csv(input_file, encoding="utf-8-sig")

# === Prepare the pieces ===
basic_info_html = f"""
<h2>📋 Basic Information</h2>
<ul>
    <li><strong>Number of patients:</strong> {df.shape[0]}</li>
    <li><strong>Number of columns:</strong> {df.shape[1]}</li>
    <li><strong>Columns:</strong> {', '.join(df.columns)}</li>
</ul>
"""

missing_values = (df.isnull().sum() + (df == "").sum())
missing_values_html = missing_values.to_frame(name='Missing Values').to_html(border=1, index=True)

stage_distribution_html = ""
if "Stage at Diagnosis" in df.columns:
    stage_distribution_html = df["Stage at Diagnosis"].value_counts(dropna=False).to_frame(name='Count').to_html(border=1)

best_response_distribution_html = ""
if "Best response" in df.columns:
    best_response_distribution_html = df["Best response"].value_counts(dropna=False).to_frame(name='Count').to_html(border=1)

deaths_count = 0
if "Date of Death" in df.columns:
    deaths_count = df["Date of Death"].replace("", pd.NA).dropna().shape[0]

family_history_count = 0
if "Canncer Family History First degree relatives" in df.columns:
    family_history_count = df["Canncer Family History First degree relatives"].replace("", pd.NA).dropna().shape[0]

# === Build HTML ===
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Patient Data Quality Control Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
            background: white;
            padding: 30px 50px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            text-align: center;
            color: #333;
        }}
        p, ul {{
            font-size: 16px;
            color: #555;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ccc;
        }}
        th, td {{
            padding: 10px;
            text-align: center;
        }}
        th {{
            background-color: #f0f0f0;
        }}
        .summary {{
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Patient Data Quality Control Report</h1>

        {basic_info_html}

        <h2>🔍 Missing Values</h2>
        {missing_values_html}

        <h2>🎯 Stage at Diagnosis Distribution</h2>
        {stage_distribution_html if stage_distribution_html else '<p class="summary">No data available</p>'}

        <h2>🏹 Best Response Distribution</h2>
        {best_response_distribution_html if best_response_distribution_html else '<p class="summary">No data available</p>'}

        <h2>🪦 Death Summary</h2>
        <p class="summary">Patients recorded as deceased: <strong>{deaths_count}</strong> out of {df.shape[0]}</p>

        <h2>👨‍👩‍👧‍👦 Family Cancer History</h2>
        <p class="summary">Patients with family history: <strong>{family_history_count}</strong> out of {df.shape[0]}</p>
    </div>
</body>
</html>
"""

# === Save HTML report ===
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Pretty HTML report generated: {output_html}")
