import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import cv2
import tempfile
import uuid
from datetime import datetime

# PDF libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4


app = FastAPI(title="PCOS Hybrid Prediction API")

# ---------------- LOAD MODELS ----------------

ml_model = joblib.load("pcos_rf_model.pkl")

cnn_model = tf.keras.models.load_model(
    "pcos_cnn_final_auc_0.998.keras"
)

# ---------------- IMAGE PREDICTION ----------------

def predict_from_image_file(upload_file: UploadFile):

    contents = upload_file.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    img = cv2.imread(temp_path)

    if img is None:
        raise ValueError("Uploaded image could not be read")

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    raw_prob = cnn_model.predict(img, verbose=0)[0][0]

    os.remove(temp_path)

    # CNN predicts NON-PCOS probability → invert
    pcos_prob = 1 - raw_prob

    return float(pcos_prob)


# ---------------- RISK CLASSIFICATION ----------------

def classify_risk(prob):

    if prob >= 0.75:
        return "HIGH RISK"
    elif prob >= 0.50:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


# ---------------- PDF REPORT GENERATOR ----------------
def infer_symptoms(data):
    symptoms = []

    if data.get("hair growth(Y/N)") == 1:
        symptoms.append("Excess facial/body hair (hirsutism)")

    if data.get("Hair loss(Y/N)") == 1:
        symptoms.append("Hair thinning or hair loss")

    if data.get("Skin darkening (Y/N)") == 1:
        symptoms.append("Skin darkening (acanthosis nigricans)")

    if data.get("Weight gain(Y/N)") == 1:
        symptoms.append("Unexplained weight gain")

    # cycle code you used: 4 = irregular, 2 = regular
    if data.get("Cycle(R/I)") == 4:
        symptoms.append("Irregular menstrual cycles")

    return symptoms
def recommendations_by_risk(risk):
    if risk == "HIGH RISK":
        return [
            "Consult a gynecologist or endocrinologist.",
            "Consider hormonal blood tests (LH, FSH, testosterone).",
            "Pelvic ultrasound examination recommended.",
            "Maintain healthy diet and regular exercise.",
        ]
    elif risk == "MODERATE RISK":
        return [
            "Monitor menstrual cycle regularly.",
            "Maintain balanced diet and physical activity.",
            "Consult a doctor if symptoms persist.",
        ]
    else:
        return [
            "Maintain healthy lifestyle and weight.",
            "Regular health check-ups recommended.",
        ]

def generate_report(data, ml_prob, cnn_prob, final_prob, risk):

    filename = f"pcos_report_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()

    title = styles['Title']
    heading = styles['Heading2']
    normal = styles['Normal']

    elements = []

    # ---------- HEADER ----------

    elements.append(Paragraph("PCOS Diagnostic Report", title))
    elements.append(Spacer(1,10))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        normal
    ))

    elements.append(Spacer(1,20))

    # ---------- CLINICAL DATA TABLE ----------

    clinical_table_data = [["Parameter", "Value"]]

    for k, v in data.items():
        clinical_table_data.append([k, str(v)])

    clinical_table = Table(clinical_table_data, colWidths=[300,150])

    clinical_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2E86C1")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

    ]))

    elements.append(Paragraph("Patient Clinical Inputs", heading))
    elements.append(Spacer(1,10))
    elements.append(clinical_table)

    elements.append(Spacer(1,25))

    # ---------- MODEL RESULTS ----------

    results_table_data = [

        ["Model", "Probability"],

        ["Clinical Model (Random Forest)", f"{ml_prob:.3f}"],
        ["Ultrasound CNN Model", f"{cnn_prob:.3f}"],
        ["Final Hybrid Score", f"{final_prob:.3f}"]

    ]

    results_table = Table(results_table_data, colWidths=[300,150])

    results_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1ABC9C")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

    ]))

    elements.append(Paragraph("Model Prediction Results", heading))
    elements.append(Spacer(1,10))
    elements.append(results_table)

    elements.append(Spacer(1,25))

    # ---------- RISK ASSESSMENT ----------

    risk_color = colors.green

    if risk == "MODERATE RISK":
        risk_color = colors.orange

    if risk == "HIGH RISK":
        risk_color = colors.red

    risk_table = Table(
        [["PCOS Risk Level", risk]],
        colWidths=[300,150]
    )

    risk_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(0,0),colors.lightgrey),
        ("BACKGROUND",(1,0),(1,0),risk_color),

        ("TEXTCOLOR",(1,0),(1,0),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("GRID",(0,0),(-1,-1),1,colors.grey)

    ]))

    elements.append(Paragraph("Risk Assessment", heading))
    elements.append(Spacer(1,10))
    elements.append(risk_table)

    elements.append(Spacer(1,25))

    # ---------- SYMPTOMS ----------

    symptoms = infer_symptoms(data)

    elements.append(Paragraph("Possible Symptoms Observed", heading))
    elements.append(Spacer(1,10))

    if symptoms:
        for s in symptoms:
            elements.append(Paragraph(f"• {s}", normal))
    else:
        elements.append(Paragraph(
            "No major symptoms detected from provided inputs.",
            normal
        ))

    elements.append(Spacer(1,20))

    # ---------- RECOMMENDATIONS ----------

    suggestions = recommendations_by_risk(risk)

    elements.append(Paragraph("Recommendations", heading))
    elements.append(Spacer(1,10))

    for rec in suggestions:
        elements.append(Paragraph(f"• {rec}", normal))

    elements.append(Spacer(1,20))

    # ---------- DISCLAIMER ----------

    elements.append(Paragraph("Disclaimer", heading))
    elements.append(Spacer(1,10))

    elements.append(Paragraph(
        "This AI-based system provides decision support and should not replace professional medical diagnosis.",
        normal
    ))

    # ---------- BUILD PDF ----------

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4
    )

    doc.build(elements)

    return filename


# ---------------- API ENDPOINT ----------------

@app.post("/predict")

async def predict_pcos(

    follicle_l: float = Form(...),
    follicle_r: float = Form(...),
    amh: float = Form(...),
    cycle_irregular: int = Form(...),
    cycle_length: int = Form(...),
    fsh_lh: float = Form(...),
    lh: float = Form(...),
    hair_growth: int = Form(...),
    hair_loss: int = Form(...),
    skin_dark: int = Form(...),
    weight_gain: int = Form(...),
    bmi: float = Form(...),
    image: UploadFile = File(...)

):

    clinical_data = {

        "Follicle No. (L)": follicle_l,
        "Follicle No. (R)": follicle_r,
        "AMH(ng/mL)": amh,
        "Cycle(R/I)": 4 if cycle_irregular == 1 else 2,
        "Cycle length(days)": cycle_length,
        "FSH/LH": fsh_lh,
        "LH(mIU/mL)": lh,
        "hair growth(Y/N)": hair_growth,
        "Hair loss(Y/N)": hair_loss,
        "Skin darkening (Y/N)": skin_dark,
        "Weight gain(Y/N)": weight_gain,
        "BMI": bmi

    }

    input_df = pd.DataFrame([clinical_data])

    # ML prediction
    ml_prob = ml_model.predict_proba(input_df)[0][1]

    # CNN prediction
    cnn_prob = predict_from_image_file(image)

    # Hybrid score
    final_prob = 0.5 * ml_prob + 0.5 * cnn_prob

    risk = classify_risk(final_prob)

    # Generate PDF report
    report_file = generate_report(
        clinical_data,
        ml_prob,
        cnn_prob,
        final_prob,
        risk
    )

    return FileResponse(
        path=report_file,
        media_type="application/pdf",
        filename="pcos_report.pdf"
    )


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )