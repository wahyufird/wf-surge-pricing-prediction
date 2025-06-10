import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_model_tuned.joblib")

st.title("Prediksi Lonjakan Harga Taksi (Surge Pricing)")

st.markdown("Isi data perjalanan untuk memprediksi apakah akan terjadi lonjakan harga.")

# Mapping
cab_type_mapping = {
    "Mini (Hatchback)": 0,
    "Sedan (Ekonomi)": 1,
    "SUV (Keluarga)": 2,
    "Premium (Luxury Car)": 3,
    "MPV (7-penumpang)": 4
}
dest_type_mapping = {
    "Bandara (Airport)": 0,
    "Stasiun / Terminal": 1,
    "Kawasan Perkantoran": 2,
    "Mall / Pusat Perbelanjaan": 3,
    "Lainnya": 4
}
gender_mapping = {"Laki-laki": 0, "Perempuan": 1}

# Input Form
cab = st.selectbox("Tipe Mobil", list(cab_type_mapping.keys()))
conf = st.slider("Confidence Lifestyle Index", 0.0, 5.0, 3.0, 0.1)
rating = st.slider("Customer Rating", 0.0, 5.0, 4.0, 0.1)
distance = st.number_input("Jarak Perjalanan (km)", min_value=0.1, step=0.1)
months = st.number_input("Lama Menjadi Customer (bulan)", min_value=1, step=1)
lifestyle = st.slider("Lifestyle Index", 0.0, 10.0, 4.0, 0.1)
cancel = st.number_input("Jumlah Pembatalan dalam 1 Bulan", min_value=0, step=1)
dest = st.selectbox("Tipe Tujuan", list(dest_type_mapping.keys()))
gender = st.selectbox("Jenis Kelamin", list(gender_mapping.keys()))

# Prediksi
if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        'Trip_Distance': distance,
        'Type_of_Cab': cab_type_mapping[cab],
        'Customer_Since_Months': months,
        'Life_Style_Index': lifestyle,
        'Confidence_Life_Style_Index': conf,
        'Destination_Type': dest_type_mapping[dest],
        'Customer_Rating': rating,
        'Cancellation_Last_1Month': cancel,
        'Gender': gender_mapping[gender]
    }])

    # Susun ulang agar sesuai urutan fitur model
    model_features = ['Trip_Distance', 'Type_of_Cab', 'Customer_Since_Months',
                      'Life_Style_Index', 'Confidence_Life_Style_Index',
                      'Destination_Type', 'Customer_Rating',
                      'Cancellation_Last_1Month', 'Var2', 'Var3', 'Gender']
    
    # Tambah dummy Var2 dan Var3 (karena model tetap butuh input ini meski tak diinput user)
    input_data['Var2'] = 2.5  # nilai tengah atau default
    input_data['Var3'] = 2.5

    input_data = input_data[model_features]  # pastikan urutan sesuai

    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error(f"🚨 Diprediksi *TERJADI* lonjakan harga. (Probabilitas: {probas[1]:.2%})")
    else:
        st.success(f"✅ Diprediksi *TIDAK TERJADI* lonjakan harga. (Probabilitas: {probas[0]:.2%})")
