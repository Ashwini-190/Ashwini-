import streamlit as st
import os
import uuid
from datetime import datetime
from PIL import Image

from predict import predict_tb, predict_ct
from gradcam_resnet import generate_gradcam

from database import (
    create_tables,
    insert_default_users,
    verify_user,
    insert_patient,
    insert_scan,
    get_patient_scans
)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


# -----------------------------------
# Initialize Database
# -----------------------------------
create_tables()
insert_default_users()

# -----------------------------------
# Login System (MUST BE FIRST)
# -----------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

if not st.session_state.logged_in:

    st.set_page_config(page_title="Hospital Login", layout="centered")
    st.title(" Hospital Login Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = verify_user(username, password)

        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.success(f"Logged in as {role}")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -----------------------------------
# Main App After Login
# -----------------------------------
st.set_page_config(page_title="TB Detection System", layout="centered")

st.sidebar.success(f"Logged in as: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.title(" AI-Based Tuberculosis Detection System")

# -----------------------------------
# Role Info
# -----------------------------------
if st.session_state.role == "Doctor":
    st.info("Doctor Access: View and retrieve reports only.")

elif st.session_state.role == "Radiologist":
    st.info("Radiologist Access: Upload scans and generate reports.")

elif st.session_state.role == "Admin":
    st.info("Admin Access: Full system control.")


# -----------------------------------
# Upload Section (Restricted)
# -----------------------------------
if st.session_state.role in ["Radiologist", "Admin"]:

    st.subheader(" Patient Information")

    patient_id = st.text_input("Patient ID")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    scan_type = st.selectbox( "Select Scan Type", ["Chest X-ray", "CT Scan"] )

    uploaded_file = st.file_uploader("Upload Scan Image", type=["jpg", "jpeg", "png"]) 

    #uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file and patient_id:

        os.makedirs("test_images", exist_ok=True)
        file_path = os.path.join("test_images", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(file_path)
        #st.image(image, caption="Uploaded X-ray", use_column_width=True)
        st.image(image, caption=f"Uploaded {scan_type}", use_column_width=True)
        insert_patient(patient_id, name, age, gender)

       # label, probability, stage = predict_tb(file_path)
        # severity_score = round(probability * 100, 2)


        if scan_type == "Chest X-ray":
            label, probability, stage = predict_tb(file_path)
            severity_score = round(probability * 100, 2)

        else:  # CT Scan
            label = predict_ct(file_path)
            probability = 0.90 if label == "Tuberculosis" else 0.10  # dummy confidence
            stage = "Detected (CT Scan)"
            severity_score = round(probability * 100, 2)

        st.subheader(" Diagnostic Summary")
        st.write(f"Diagnosis: {label}")
        st.write(f"Probability: {probability:.2f}")
        st.write(f"Stage: {stage}")
        st.write(f"Severity Score: {severity_score}/100")

        # GradCAM
        if scan_type == "Chest X-ray":
            st.subheader(" Grad-CAM Visualization")
            gradcam_path = generate_gradcam(file_path)
            st.image(gradcam_path, caption="Grad-CAM Heatmap", use_column_width=True)

        if label == "Normal":
            st.success("No significant pathological activation detected.")
        else:
            st.warning("Highlighted regions indicate areas influencing TB prediction.")

        # PDF Generation
        if st.button("Generate Downloadable PDF Report"):

            scan_id = str(uuid.uuid4())[:8]
            os.makedirs("reports", exist_ok=True)
            pdf_path = f"reports/{scan_id}_TB_Report.pdf"

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>TB Clinical Report</b>", styles["Title"]))
            elements.append(Spacer(1, 0.5 * inch))

            elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
            elements.append(Paragraph(f"Scan ID: {scan_id}", styles["Normal"]))
            elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
            elements.append(Spacer(1, 0.3 * inch))

            elements.append(Paragraph(f"Diagnosis: {label}", styles["Normal"]))
            elements.append(Paragraph(f"Probability: {probability:.2f}", styles["Normal"]))
            elements.append(Paragraph(f"Stage: {stage}", styles["Normal"]))
            elements.append(Paragraph(f"Severity Score: {severity_score}/100", styles["Normal"]))
            elements.append(Spacer(1, 0.3 * inch))

            elements.append(Paragraph("Disclaimer:", styles["Heading2"]))
            elements.append(Paragraph(
                "This AI-generated report is for research support only. "
                "Clinical confirmation by a certified radiologist is required.",
                styles["Normal"]
            ))

            doc.build(elements)

            insert_scan(
                scan_id,
                patient_id,
                file_path,
                label,
                probability,
                stage,
                severity_score,
                pdf_path
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )

            st.success("Report saved successfully.")

else:
    st.warning("You do not have permission to upload scans.")


# -----------------------------------
# Retrieve Section (Doctor + Admin)
# -----------------------------------
if st.session_state.role in ["Doctor", "Admin"]:

    st.subheader(" Retrieve Patient Records")
    search_id = st.text_input("Enter Patient ID")

    if st.button("Search Records"):
        records = get_patient_scans(search_id)

        if records:
            for record in records:
                scan_id, diagnosis, probability, stage, severity_score, date, report_path = record

                st.markdown(f"""
                ### Scan ID: {scan_id}
                - Diagnosis: {diagnosis}
                - Probability: {probability:.2f}
                - Stage: {stage}
                - Severity Score: {severity_score}/100
                - Date: {date}
                """)

                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download This Report",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )

                st.divider()
        else:
            st.warning("No records found.")