import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import cv2
import tempfile
import os


app = FastAPI(title="PCOS Hybrid Prediction API")

# Load models ONCE at startup
ml_model = joblib.load("pcos_rf_model.pkl")
cnn_model = tf.keras.models.load_model(
    "pcos_cnn_final_auc_0.998.keras"
)


# ---------- IMAGE PREDICTION ----------
def predict_from_image_file(upload_file: UploadFile):
    contents = upload_file.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    img = cv2.imread(temp_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    raw_prob = cnn_model.predict(img, verbose=0)[0][0]

    os.remove(temp_path)

    # CNN outputs NON-PCOS probability → invert
    pcos_prob = 1 - raw_prob
    return float(pcos_prob)

# ---------- RISK CLASS ----------
def classify_risk(prob):
    if prob >= 0.75:
        return "HIGH RISK"
    elif prob >= 0.50:
        return "MODERATE RISK"
    else:
        return "LOW RISK"

# ---------- API ENDPOINT ----------
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
    image: UploadFile = File(None)
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
    ml_prob = ml_model.predict_proba(input_df)[0][1]

    if image:
        cnn_prob = predict_from_image_file(image)
        final_prob = 0.6 * ml_prob + 0.4 * cnn_prob
        mode = "HYBRID (Clinical + Image)"
    else:
        cnn_prob = None
        final_prob = ml_prob
        mode = "ML ONLY (Clinical)"

    return {
        "mode": mode,
        "ml_probability": round(ml_prob, 3),
        "cnn_probability": None if cnn_prob is None else round(cnn_prob, 3),
        "final_probability": round(final_prob, 3),
        "risk_level": classify_risk(final_prob),
        "disclaimer": "This is a decision-support tool, not a medical diagnosis."
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

