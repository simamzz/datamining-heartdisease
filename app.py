import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Load trained model
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

def prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    pred = classifier.predict(input_data)
    proba = classifier.predict_proba(input_data)[:, 1]
    return pred, proba

def evaluate_model(X_test, y_test):
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba

def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

def explanation_page():
    st.title("Heart Disease Prediction App - Explanation")
    st.write("""
    ### Welcome to the Explanation Page
             
    ## Anggota Kelompok
    1. Humam Razzan Herditama   | 1202220177
    2. Kressna Mukti Wibowo      | 1202223242
    3. Mochammad Aziiz Nugroho  | 1202223258
    4. Syarif Imam Muslim       | 1202220108
             
    ### Penjelasan Aplikasi
    Aplikasi ini merupakan aplikasi yang dapat memprediksi penyakit jantung berdasarkan data yang diinputkan oleh pengguna. Aplikasi ini menggunakan model Machine Learning yang sudah dilatih sebelumnya dengan menggunakan dataset penyakit jantung yang diambil dari [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
    
    ## Penjelasan Data
    1. age (usia)
    2. sex (jenis kelamin)
    3. chest pain type (4 values) (jenis nyeri dada (4 nilai))
    4. resting blood pressure (tekanan darah saat istirahat)
    5. serum cholestoral in mg/dl (kolesterol serum dalam mg/dl)
    6. fasting blood sugar > 120 mg/dl (gula darah puasa > 120 mg/dl)
    7. resting electrocardiographic results (values 0,1,2) (hasil elektrokardiografi saat istirahat (nilai 0,1,2))
    8. maximum heart rate achieved (detak jantung maksimum yang dicapai)
    9. exercise induced angina (angina yang diinduksi oleh olahraga)
    10. oldpeak = ST depression induced by exercise relative to rest (oldpeak = depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat)
    11. the slope of the peak exercise ST segment (kemiringan segmen ST puncak latihan)
    12. number of major vessels (0-3) colored by flourosopy (jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi)
    13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (thal: 0 = normal; 1 = cacat tetap; 2 = cacat yang dapat diperbaiki)
            
    ## Penjelasan Model
    Model yang digunakan adalah Logistic Regression dan Clustering
    
    ## Kesimpulan Clustering
    Melakukan analisis clustering pada dataset penyakit jantung menggunakan algoritma K-means. Menggunakan metode Elbow untuk menentukan jumlah cluster yang optimal. Kemudian menerapkan algoritma K-means dengan jumlah cluster yang telah ditentukan yaitu (3). Principal Component Analysis (PCA) untuk mereduksi dimensi data dan memvisualisasikan hasil clustering.
    Terakhir menganalisis rata-rata setiap fitur dalam masing-masing cluster untuk memahami karakteristik setiap cluster.

    Hasil dari analisis ini menunjukkan bahwa data dapat dikelompokkan menjadi beberapa cluster yang memiliki karakteristik berbeda. Visualisasi PCA membantu dalam memahami distribusi data dalam ruang dua dimensi. Analisis lebih lanjut dapat dilakukan untuk mengidentifikasi pola dan hubungan yang lebih mendalam dalam data.

    Hasil clustering menunjukkan adanya 3 segmen pasien:

    1. Cluster 0: Kemungkinan pasien dengan kondisi kesehatan lebih baik:
    - Usia lebih muda.
    - Angina yang diinduksi olahraga sangat rendah.
    - Detak jantung maksimum yang lebih tinggi.
    - Oldpeak yang rendah (indikator stres pada jantung minimal).

    2. Cluster 1: Pasien dengan faktor risiko sedang:
    - Usia lebih tua dari Cluster 0.
    - Semua memiliki gula darah tinggi, yang merupakan faktor risiko penyakit jantung.
    - Oldpeak dan thalach berada di tengah-tengah.

    3. Cluster 2: Pasien dengan risiko lebih tinggi:
    - Usia tertua.
    - Angina yang diinduksi olahraga tinggi.
    - Detak jantung maksimum terendah, yang menunjukkan jantung kurang optimal dalam merespons stres.
    - Oldpeak yang tinggi, menunjukkan stres pada jantung yang lebih besar.

    ## Kesimpulan Logistic Regression
    Model Logistic Regression bekerja dengan baik untuk memprediksi risiko penyakit jantung dengan akurasi 81% dan AUC 93%, yang menunjukkan bahwa model ini cukup efektif dalam membedakan antara pasien yang memiliki penyakit jantung dan yang tidak.

    ### Detail Evaluasi Model:
    - **Accuracy**: 81.46%
    - **ROC-AUC Score**: 92.99%
    - **Classification Report**:
        - Precision, Recall, dan F1-Score untuk masing-masing kelas (0 = No Disease, 1 = Disease) menunjukkan performa yang baik, terutama pada kelas 1 (Disease) dengan recall yang tinggi (92%).

    ### Confusion Matrix:
    - **True Positives (TP)**: 97
    - **True Negatives (TN)**: 70
    - **False Positives (FP)**: 30
    - **False Negatives (FN)**: 8

    ### ROC Curve:
    - ROC Curve menunjukkan bahwa model memiliki kemampuan yang baik dalam membedakan antara kelas positif dan negatif dengan area di bawah kurva (AUC) sebesar 92.99%.

    ### Kesimpulan Umum:
    Model Logistic Regression yang dibangun dapat digunakan sebagai alat prediktif yang efektif untuk membantu tenaga medis atau pihak rumah sakit dalam memprediksi risiko penyakit jantung pada pasien berdasarkan data kesehatan mereka. Dengan akurasi dan AUC yang tinggi, model ini dapat diandalkan untuk pengambilan keputusan klinis yang lebih baik dan lebih cepat.
    """)

def prediction_page():
    st.title("Heart Disease Prediction ML App")
    html_temp = """
    <div style="background-color:darkblue;padding:13px; border-radius:15px; margin-bottom:20px;">
    <h1 style="color:white;text-align:center;">Heart Disease Prediction ML App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    df = pd.read_csv('heart.csv')
    df.drop_duplicates(inplace=True)
    
    X = df.drop(columns=["target"])
    y = df["target"]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
    chol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    result = ""
    proba_result = ""

    if st.button("Predict"):
        result, proba = prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        proba_result = f"{proba[0]:.2f}"
        result = 'Yes' if result[0] == 1 else 'No'

    st.success(f'Prediksi Penyakit Jantung: {result}')
    st.success(f'Probabilitas Penyakit Jantung: {proba_result}')

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Explanation", "Prediction"])

    if page == "Explanation":
        explanation_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == '__main__':
    main()
