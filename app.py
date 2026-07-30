from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle, numpy as np, pandas as pd, os, uuid
from datetime import date

app = Flask(__name__)
app.secret_key = "loan_secret_key"

# ── Database Config ───────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "instance", "loan_data.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ── Database Models ──────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    income = db.Column(db.Float, nullable=False)
    credit_score = db.Column(db.Float, nullable=False)
    loan_amount = db.Column(db.Float, nullable=False)
    employment_years = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(20))
    confidence = db.Column(db.String(20))
    model_used = db.Column(db.String(50))
    created_at = db.Column(db.Date, default=date.today)

# ── Model Loading ────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURE_LABELS = {
    "income": "Applicant Income",
    "credit_score": "Credit Score",
    "loan_amount": "Loan Amount",
    "employment_years": "Years of Employment"
}

CSV_STORE = {}

# Load multiple models into a dictionary
MODELS = {}
model_files = [
    "logistic_regression.pkl",
    "random_forest_model.pkl",
    "svm_model.pkl",
    "decision_tree_model.pkl",
    "knn_model.pkl",
    "naive_bayes_model.pkl"
]

for fname in model_files:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            MODELS[fname.split(".")[0]] = pickle.load(f)

# Load scaler and feature names
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "features.pkl"), "rb") as f:
    feature_names = [str(x) for x in pickle.load(f)]

print("Loaded models:", list(MODELS.keys()))
if not MODELS:
    raise RuntimeError("No models loaded! Please check your model directory.")


def get_model(algo):
    """Return the requested model by name, falling back to any available model."""
    return MODELS.get(algo) or next(iter(MODELS.values()))


# ── Signup ───────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        uname = request.form["username"]
        pwd = request.form["password"]
        confirm_pwd = request.form["confirm_password"]

        if User.query.filter_by(username=uname).first():
            return render_template("signup.html", error="Username already exists!")
        if pwd != confirm_pwd:
            return render_template("signup.html", error="Passwords do not match!")
        if len(pwd) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters!")

        new_user = User(username=uname, password=pwd)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

# ── Login ────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["username"]
        pwd = request.form["password"]
        user = User.query.filter_by(username=uname, password=pwd).first()
        if user:
            session["user"] = uname
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials!")
    return render_template("login.html")

# ── Logout ───────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")

# ── Dashboard ────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    loans = LoanApplication.query.all()

    # KPI calculations
    total_apps = len(loans)
    approved = sum(1 for l in loans if l.prediction == "Approved")
    rejected = total_apps - approved
    approval_rate = round((approved / total_apps) * 100, 2) if total_apps else 0
    avg_loan = round(np.mean([l.loan_amount for l in loans]), 2) if loans else 0
    default_risk = round(100 - approval_rate, 2)

    return render_template(
        "dashboard.html",
        loans=loans,
        total_apps=total_apps,
        approval_rate=approval_rate,
        avg_loan=avg_loan,
        default_risk=default_risk,
        approved=approved,
        rejected=rejected
    )

# ── Predict Form ─────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        algo = request.form.get("model", "logistic_regression")

        try:
            values = [float(request.form.get(f, 0)) for f in feature_names]
        except ValueError:
            loans = LoanApplication.query.order_by(LoanApplication.id.desc()).all()
            return render_template("predict.html",
                                   feature_names=feature_names,
                                   labels=FEATURE_LABELS,
                                   loans=loans,
                                   error="Invalid input! Please enter numeric values.")

        arr = np.array(values).reshape(1, -1)
        X = scaler.transform(arr)

        model = get_model(algo)

        # Prediction
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        result = "Approved" if pred == 1 else "Rejected"
        confidence = f"{max(prob) * 100:.1f}%"

        new_loan = LoanApplication(
            income=values[0],
            credit_score=values[1],
            loan_amount=values[2],
            employment_years=int(values[3]),
            prediction=result,
            confidence=confidence,
            model_used=algo
        )
        db.session.add(new_loan)
        db.session.commit()

        loans = LoanApplication.query.order_by(LoanApplication.id.desc()).all()
        return render_template("predict.html",
                               feature_names=feature_names,
                               labels=FEATURE_LABELS,
                               loans=loans,
                               result=result,
                               confidence=confidence,
                               algorithm=algo)

    loans = LoanApplication.query.order_by(LoanApplication.id.desc()).all()
    return render_template("predict.html", feature_names=feature_names, labels=FEATURE_LABELS, loans=loans)


# ── Upload CSV (Form) ────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            return render_template("upload.html", error="Please select a CSV file!")

        algo = request.form.get("model", "logistic_regression")
        model = get_model(algo)

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        for i, row in df.iterrows():
            values = [float(row[feature_names[0]]),
                      float(row[feature_names[1]]),
                      float(row[feature_names[2]]),
                      int(row[feature_names[3]])]
            arr = np.array(values).reshape(1, -1)
            X = scaler.transform(arr)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            result = "Approved" if pred == 1 else "Rejected"
            confidence = f"{max(prob)*100:.1f}%"

            new_loan = LoanApplication(
                income=values[0],
                credit_score=values[1],
                loan_amount=values[2],
                employment_years=values[3],
                prediction=result,
                confidence=confidence,
                model_used=algo
            )
            db.session.add(new_loan)
        db.session.commit()
        return redirect(url_for("dashboard"))

    return render_template("upload.html")

# ── CSV Upload (AJAX) ────────────────────────────
@app.route("/csv-upload", methods=["POST"])
def csv_upload():
    file = request.files.get("csv_file")
    if not file or file.filename.strip() == "":
        return jsonify({"error": "Please select a CSV file!"}), 400

    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        store_id = str(uuid.uuid4())
        CSV_STORE[store_id] = df.to_json(orient="records")

        return jsonify({
            "success": True,
            "store_id": store_id,
            "total_rows": len(df),
            "columns": list(df.columns)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500

# ── CSV Predict ──────────────────────────────────
@app.route("/csv-predict", methods=["POST"])
def csv_predict():
    store_id = request.form.get("store_id", "").strip()
    if not store_id or store_id not in CSV_STORE:
        return jsonify({"error": "CSV data not found!"}), 400

    algo = request.form.get("model", "logistic_regression")
    model = get_model(algo)

    df = pd.read_json(CSV_STORE[store_id])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    X = scaler.transform(df[feature_names].values.astype(float))
    preds = model.predict(X)
    probs = model.predict_proba(X)
    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        result = "Approved" if pred == 1 else "Rejected"
        confidence = f"{max(prob)*100:.1f}%"
        results.append({
            "row": i+1,
            "prediction": result,
            "confidence": confidence
        })
        new_loan = LoanApplication(
            income=float(df.iloc[i][feature_names[0]]),
            credit_score=float(df.iloc[i][feature_names[1]]),
            loan_amount=float(df.iloc[i][feature_names[2]]),
            employment_years=int(df.iloc[i][feature_names[3]]),
            prediction=result,
            confidence=confidence,
            model_used=algo
        )
        db.session.add(new_loan)
    db.session.commit()
    CSV_STORE.pop(store_id, None)
    return jsonify({"success": True, "results": results})

@app.route("/")
def home():
    return "Loan Approval Portal is running!"
# ── Run App ──────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
# ── Vercel Entry Point ──────────────────────────
application = app