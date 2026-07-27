import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── Sample Loan Dataset ──
data = {
    "income": [40000, 25000, 60000, 80000, 30000, 100000, 45000, 70000],
    "credit_score": [650, 580, 720, 750, 600, 800, 640, 710],
    "loan_amount": [20000, 15000, 30000, 40000, 12000, 50000, 18000, 25000],
    "employment_years": [2, 1, 5, 7, 1, 10, 3, 6],
    "approved": [1, 0, 1, 1, 0, 1, 0, 1]  # 1 = Approved, 0 = Rejected
}
df = pd.DataFrame(data)

X = df.drop("approved", axis=1)
y = df["approved"]

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scaling ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Save Scaler and Features ──
os.makedirs("model", exist_ok=True)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# ── Train and Save Models ──
models = {
    "logistic_model.pkl": LogisticRegression(),
    "random_forest_model.pkl": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm_model.pkl": SVC(probability=True, random_state=42),
    "decision_tree_model.pkl": DecisionTreeClassifier(random_state=42),
    "knn_model.pkl": KNeighborsClassifier(n_neighbors=3),
    "naive_bayes_model.pkl": GaussianNB()
}

for filename, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    print(f"{filename} trained. Accuracy: {acc:.2f}")
    with open(f"model/{filename}", "wb") as f:
        pickle.dump(model, f)

print("✅ All models saved in /model folder")
