from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid
from datetime import datetime

from predict import predict_tb
from database import (
    create_tables,
    insert_patient,
    insert_scan,
    get_patient_scans,
    verify_user
)

# Initialize DB
create_tables()

app = FastAPI(title="TB AI Cross-Hospital API")


# ---------------------------------------
# 1️⃣ LOGIN API
# ---------------------------------------
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    role = verify_user(username, password)

    if role:
        return {"status": "success", "role": role}
    else:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid credentials"}
        )


# ---------------------------------------
# 2️⃣ PREDICT API (Cross-Hospital)
# ---------------------------------------
@app.post("/predict")
async def predict_api(
    patient_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    file: UploadFile = File(...)
):

    os.makedirs("api_uploads", exist_ok=True)

    file_path = os.path.join("api_uploads", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Insert patient
    insert_patient(patient_id, name, age, gender)

    # Prediction
    label, probability, stage = predict_tb(file_path)
    severity_score = round(probability * 100, 2)

    # Generate scan ID
    scan_id = str(uuid.uuid4())[:8]

    # Insert scan record (no PDF here, API only stores metadata)
    insert_scan(
        scan_id,
        patient_id,
        file_path,
        label,
        probability,
        stage,
        severity_score,
        report_path=None
    )

    return {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "diagnosis": label,
        "probability": probability,
        "stage": stage,
        "severity_score": severity_score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ---------------------------------------
# 3️⃣ GET PATIENT RECORDS
# ---------------------------------------
@app.get("/get_patient/{patient_id}")
def get_patient(patient_id: str):

    records = get_patient_scans(patient_id)

    if not records:
        return {"status": "No records found"}

    formatted = []

    for r in records:
        scan_id, diagnosis, probability, stage, severity_score, date, report_path = r

        formatted.append({
            "scan_id": scan_id,
            "diagnosis": diagnosis,
            "probability": probability,
            "stage": stage,
            "severity_score": severity_score,
            "date": date,
            "report_path": report_path
        })

    return {"patient_id": patient_id, "records": formatted}


# ---------------------------------------
# Run Server
# ---------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


@app.get("/")
def home():
    return {"message": "TB AI Cross-Hospital API is running"}