# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Definisikan kurs Euro ke Rupiah
# (Kurs per akhir Juni 2025, bisa disesuaikan)
KURS_EURO_TO_IDR = 17800

# ----------------------------------
# Fungsi untuk Memuat dan Membersihkan Data
@st.cache_data
def load_and_clean_data(filepath):
    # Membaca dataset
    df = pd.read_csv(filepath, encoding='latin-1')

    # 1. Membersihkan Kolom 'Ram' dan 'Weight'
    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

    # 2. Feature Engineering dari 'ScreenResolution'
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

    # 3. Feature Engineering dari 'Cpu'
    df['Cpu Brand'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

    # 4. Feature Engineering dari 'Memory'
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    new = df["Memory"].str.split("+", n = 1, expand = True)
    df["first"]= new[0]
    df["first"]=df["first"].str.strip()
    df["second"]= new[1]
    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df['first'] = df['first'].str.extract('(\d+)')
    df['second'].fillna("0", inplace = True)
    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df['second'] = df['second'].str.extract('(\d+)')
    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)
    df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer2HDD', 'Layer2SSD', 'Memory'], inplace=True)

    # 5. Membersihkan kolom lain
    df.drop(columns=['ScreenResolution', 'Cpu', 'Product', 'laptop_ID'], inplace=True, errors='ignore')

    return df

# ----------------------------------
# Fungsi untuk Melatih Model
def train_model(df):
    categorical_cols = ['Company', 'TypeName', 'OpSys', 'Cpu Brand']
    numerical_cols = ['Inches', 'Ram', 'Weight', 'Touchscreen', 'HDD', 'SSD']

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    features = categorical_cols + numerical_cols
    X = df[features]

    ### PERUBAHAN DI SINI: Menggunakan 'Price_euros' sebagai target ###
    y = np.log1p(df['Price_euros'])

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
    model.fit(X, y)

    return model, label_encoders, features

# --- Mulai Aplikasi Streamlit ---

st.set_page_config(page_title="Prediksi Harga Laptop", layout="wide")
st.title(" Sistem Prediksi Harga Laptop")
st.write("Sistem ini memprediksi harga laptop berdasarkan spesifikasi yang Anda pilih. Dibuat sebagai proyek  Data Mining Menggunakan Algoritma Random Forest.")

# Memuat data
try:
    df_raw = pd.read_csv('laptop_price.csv', encoding='latin-1')
    df_clean = load_and_clean_data('laptop_price.csv')
    model, label_encoders, feature_list = train_model(df_clean.copy())
except FileNotFoundError:
    st.error("File 'laptop_price.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan aplikasi ini.")
    st.stop()
except KeyError as e:
    st.error(f"Terjadi kesalahan nama kolom: {e}. Pastikan kolom di file CSV sesuai dengan yang diharapkan oleh kode.")
    st.stop()


# Membuat kolom untuk input pengguna
col1, col2 = st.columns(2)

with col1:
    st.header("Masukkan Spesifikasi Laptop")
    company = st.selectbox('Merek', sorted(df_raw['Company'].unique()))
    type_name = st.selectbox('Tipe', sorted(df_raw['TypeName'].unique()))
    ram = st.select_slider('RAM (dalam GB)', options=sorted(df_raw['Ram'].str.replace('GB','').astype(int).unique()))
    weight = st.slider('Berat (dalam Kg)', min_value=0.5, max_value=4.5, step=0.1, value=1.5)
    touchscreen = st.selectbox('Layar Sentuh (Touchscreen)', ['Tidak', 'Ya'])

with col2:
    st.header(" ") # Spacer
    st.write(" ")
    inches = st.slider('Ukuran Layar (dalam Inci)', min_value=13.0, max_value=18.0, step=0.1, value=15.6)
    cpu_brand_list = sorted(df_raw['Cpu'].apply(lambda x: " ".join(x.split()[0:3])).unique())
    cpu_brand = st.selectbox('Merek CPU', cpu_brand_list)
    hdd = st.select_slider('Kapasitas HDD (dalam GB)', options=[0, 128, 256, 512, 1024, 2048])
    ssd = st.select_slider('Kapasitas SSD (dalam GB)', options=[0, 128, 256, 512, 1024])
    opsys = st.selectbox('Sistem Operasi (OS)', sorted(df_raw['OpSys'].unique()))

# Tombol untuk memprediksi
if st.button('Prediksi Harga', use_container_width=True, type="primary"):
    input_data = pd.DataFrame({
        'Company': [company], 'TypeName': [type_name], 'Inches': [inches], 'Ram': [ram],
        'Weight': [weight], 'Touchscreen': [1 if touchscreen == 'Ya' else 0], 'OpSys': [opsys],
        'Cpu Brand': [cpu_brand], 'HDD': [hdd], 'SSD': [ssd]
    })

    for col, le in label_encoders.items():
        known_classes = list(le.classes_)
        if input_data[col].iloc[0] not in known_classes:
             st.warning(f"Kategori '{input_data[col].iloc[0]}' pada '{col}' tidak dikenali. Hasil prediksi mungkin tidak akurat.")
             input_data[col] = -1
        else:
            input_data[col] = le.transform(input_data[col])

    input_data = input_data[feature_list]
    log_price_prediction = model.predict(input_data)
    price_prediction_euros = np.expm1(log_price_prediction)

    ### PERUBAHAN DI SINI: Konversi dari Euro ke Rupiah ###
    harga_rupiah = price_prediction_euros[0] * KURS_EURO_TO_IDR

    st.markdown("---")
    st.subheader(f"Berdasarkan spesifikasi yang Anda pilih, estimasi harga laptop adalah:")
    st.markdown(f"<h2 style='text-align: center; color: #28a745;'>Rp {harga_rupiah:,.0f}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: grey;'>(Prediksi harga dalam Euro: €{price_prediction_euros[0]:,.2f})</p>", unsafe_allow_html=True)


st.sidebar.info("Dibuat oleh Adam Webdev - [312210242]")
st.sidebar.markdown("---")
st.sidebar.header("Tentang Proyek")
st.sidebar.write(" Data Mining 2025")
st.sidebar.write("Dataset: Laptop Price  Kaggle.")
st.sidebar.write("Algoritma: Random Forest Regressor.")
st.sidebar.write(f"Kurs: 1 Euro = Rp {KURS_EURO_TO_IDR:,.0f}")