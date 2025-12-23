import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import dagshub

# --- KONFIGURASI ---
INPUT_PATH = 'data_training_final.csv'
warnings.filterwarnings("ignore")

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan.")
    return pd.read_csv(path)

def train_model():
    # 1. Setup Tracking
    if not os.getenv("GITHUB_ACTIONS"):
        dagshub.init(repo_owner='rstiannr', repo_name='Mlops_Resti_Asah', mlflow=True)
    else:
        mlflow.set_tracking_uri("https://dagshub.com/rstiannr/Mlops_Resti_Asah.mlflow")

    # 2. Aktifkan Autolog (Wajib Kriteria 2)
    mlflow.sklearn.autolog()

    # 3. Proses Training (Pastikan semua baris ini memiliki spasi awal yang sama)
    print("🚀 Mulai Training Model Dasar (Advance Mode)...")
    
    df = load_data(INPUT_PATH)
    
    # Pilih fitur yang sesuai dengan eksperimen
    drop_cols = ['Label', 'StockCode', 'Description', 'Month_Year', 'InvoiceDate', 'Avg_Revenue', 'CV']
    X = df.drop(columns=drop_cols, errors='ignore').astype(float)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Fit (Otomatis direkam oleh autolog ke run yang dibuat mlflow run)
    model = DecisionTreeClassifier(
        criterion='gini',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Tambahkan pendaftaran model secara eksplisit
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="model_bawang" # Gunakan nama tetap
    )
    
    # 4. Log Artefak Tambahan
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Basic Model')
    plt.savefig("confusion_matrix_basic.png")
    mlflow.log_artifact("confusion_matrix_basic.png")
    plt.close()
    
    print("💾 Sukses! Training selesai dan tercatat di MLflow.")

if __name__ == "__main__":
    train_model()