import sqlite3
from datetime import datetime
import hashlib

DB_NAME = "hospital_data.db"


# -----------------------------
# Utility: Hash Password
# -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# -----------------------------
# Create Tables
# -----------------------------
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        created_at TEXT
    )
    """)

    # Scans table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        scan_id TEXT PRIMARY KEY,
        patient_id TEXT,
        image_path TEXT,
        diagnosis TEXT,
        probability REAL,
        stage TEXT,
        severity_score REAL,
        report_date TEXT,
        report_path TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT
    )
    """)

    conn.commit()
    conn.close()


# -----------------------------
# Insert Default Users
# -----------------------------
def insert_default_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    users = [
        ("admin", hash_password("admin123"), "Admin"),
        ("doctor1", hash_password("doctor123"), "Doctor"),
        ("radiologist1", hash_password("radio123"), "Radiologist"),
    ]

    for user in users:
        try:
            cursor.execute("INSERT INTO users VALUES (?, ?, ?)", user)
        except:
            pass

    conn.commit()
    conn.close()


# -----------------------------
# Verify Login
# -----------------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    hashed = hash_password(password)

    cursor.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, hashed)
    )

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None


# -----------------------------
# Patient Functions
# -----------------------------
def insert_patient(patient_id, name, age, gender):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO patients VALUES (?, ?, ?, ?, ?)
    """, (patient_id, name, age, gender,
          datetime.now().strftime("%Y-%m-%d")))

    conn.commit()
    conn.close()


def insert_scan(scan_id, patient_id, image_path,
                diagnosis, probability, stage,
                severity_score, report_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id,
        patient_id,
        image_path,
        diagnosis,
        probability,
        stage,
        severity_score,
        datetime.now().strftime("%Y-%m-%d"),
        report_path
    ))

    conn.commit()
    conn.close()


def get_patient_scans(patient_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT scan_id, diagnosis, probability, stage,
           severity_score, report_date, report_path
    FROM scans WHERE patient_id=?
    """, (patient_id,))

    records = cursor.fetchall()
    conn.close()
    return records