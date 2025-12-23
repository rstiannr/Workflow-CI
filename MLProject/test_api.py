import requests
import pandas as pd

# --- DATA DISESUAIKAN DENGAN PERMINTAAN SERVER ---
data = {
    "dataframe_split": {
        # Server minta kolom: Avg_Sales, Std_Dev, Max_Sales, UnitPrice
        "columns": ["Avg_Sales", "Std_Dev", "Max_Sales", "UnitPrice"],
        "data": [
            # Kita isi angka dummy (asal) berbentuk desimal (double/float)
            [38.76, 129.35, 468, 0.24]
        ]
    }
}

url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=data, headers=headers)
    print("Status Code:", response.status_code)
    print("Hasil Prediksi:", response.text)
except Exception as e:
    print("Error:", e)