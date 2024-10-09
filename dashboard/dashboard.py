import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind
import requests
import numpy as np

# Fungsi untuk mendownload file dari Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Fungsi untuk mengakses dataset dari Google Drive
def load_data_from_gdrive():
    try:
        # ID file dari Google Drive
        day_file_id = '1QE1baqT-k2dt4mzrLi6WHou6YuYigQB9'
        hour_file_id = '1BQWbPKqHUY12IA-o7R-hTSkknFpohWlM'
        
        # Download file ke lokal
        download_file_from_google_drive(day_file_id, 'day.csv')
        download_file_from_google_drive(hour_file_id, 'hour.csv')
        
        # Baca file CSV dari lokal
        day_df = pd.read_csv('day.csv')
        hour_df = pd.read_csv('hour.csv')
        return day_df, hour_df
    except Exception as e:
        st.warning("Tidak bisa mengakses data dari Google Drive. Beralih ke data offline.")
        return load_data_offline()

# Fungsi untuk mengakses dataset secara offline
def load_data_offline():
    try:
        day_df = pd.read_csv('day.csv')
        hour_df = pd.read_csv('hour.csv')
        return day_df, hour_df
    except Exception as e:
        st.error("Gagal memuat dataset lokal. Pastikan file lokal tersedia.")

# Mengimpor dataset
@st.cache_data
def load_data():
    return load_data_from_gdrive()

# Menampilkan informasi proyek
st.title("Proyek Analisis Data: Bike Sharing Dataset 🚴")
st.markdown("""
- **Nama:** Ria Amelia Shinta Putricia Hendra  
- **Email:** riaptrcia@gmail.com  
- **ID Dicoding:** riaamelia  
""")

# Menambahkan filter data
st.sidebar.header('Filter Data')
day_selection = st.sidebar.selectbox("Pilih Hari", ["Semua", "Weekday", "Weekend"])

day_df, hour_df = load_data()

# Menampilkan data wrangling (tampilan beberapa baris pertama)
st.header("Data Wrangling")
st.subheader("Beberapa Baris Pertama dari Dataset 'day.csv'")
st.write(day_df.head())

st.subheader("Beberapa Baris Pertama dari Dataset 'hour.csv'")
st.write(hour_df.head())

# Mengecek nilai yang hilang dalam dataset
st.subheader("Jumlah Nilai yang Hilang di Dataset 'day.csv'")
missing_day = day_df.isnull().sum()
st.write(missing_day)

st.subheader("Jumlah Nilai yang Hilang di Dataset 'hour.csv'")
missing_hour = hour_df.isnull().sum()
st.write(missing_hour)

# ----------------------------------------------------
# Visualisasi Distribusi Jumlah Total Pengguna (cnt) dalam dataset 'day.csv'
# ----------------------------------------------------
st.header("Visualisasi Distribusi Jumlah Total Pengguna (cnt) per Hari")
plt.figure(figsize=(12, 6))
counts, bin_edges = np.histogram(day_df['cnt'], bins=30)

for i in range(len(counts)):
    plt.bar(bin_edges[i], counts[i], width=bin_edges[i + 1] - bin_edges[i], 
            color='lightblue', alpha=0.5, edgecolor='black', linewidth=1.5)

max_count = counts.max()
max_bin_index = np.argmax(counts)

plt.bar(bin_edges[max_bin_index], 
         counts[max_bin_index], 
         width=bin_edges[max_bin_index + 1] - bin_edges[max_bin_index], 
         color='darkblue', alpha=0.7, edgecolor='black', linewidth=1.5, label='Balok Tertinggi')

plt.title('Distribusi Jumlah Total Pengguna (cnt) per Hari')
plt.xlabel('Jumlah Total Pengguna')
plt.ylabel('Frekuensi')
plt.axvline(max_count, color='red', linestyle='--', linewidth=2, label='Jumlah Pengguna Tertinggi')
plt.legend()
plt.grid()
st.pyplot(plt)

# ----------------------------------------------------
# Visualisasi Sebelum Pertanyaan: Variabel Cuaca dan Kondisi
# ----------------------------------------------------

# Distribusi Suhu
st.header("Distribusi Suhu (Temperature)")
plt.figure(figsize=(10, 6))
sns.histplot(day_df['temp'], kde=True, color='orange')
plt.title("Distribusi Suhu (Temperature)")
plt.xlabel('Suhu (Normalisasi)')
plt.ylabel('Frekuensi')
plt.grid(True)
st.pyplot(plt)

# Distribusi Kelembapan
st.header("Distribusi Kelembapan (Humidity)")
plt.figure(figsize=(10, 6))
sns.histplot(day_df['hum'], kde=True, color='green')
plt.title("Distribusi Kelembapan (Humidity)")
plt.xlabel('Kelembapan')
plt.ylabel('Frekuensi')
plt.grid(True)
st.pyplot(plt)

# Distribusi Kecepatan Angin
st.header("Distribusi Kecepatan Angin (Windspeed)")
plt.figure(figsize=(10, 6))
sns.histplot(day_df['windspeed'], kde=True, color='blue')
plt.title("Distribusi Kecepatan Angin (Windspeed)")
plt.xlabel('Kecepatan Angin (Normalisasi)')
plt.ylabel('Frekuensi')
plt.grid(True)
st.pyplot(plt)

# Distribusi Kondisi Cuaca
st.header("Distribusi Kondisi Cuaca (Weathersit)")
plt.figure(figsize=(8, 5))
sns.countplot(x='weathersit', data=day_df, palette='coolwarm')
plt.title('Distribusi Kondisi Cuaca (Weathersit)')
plt.xlabel('Kondisi Cuaca')
plt.ylabel('Frekuensi')
plt.xticks([0, 1, 2, 3], ['Cerah', 'Mendung', 'Hujan', 'Salju'])
plt.grid(True)
st.pyplot(plt)

# ----------------------------------------------------
# Visualisasi Variasi Jumlah Penyewa Berdasarkan Musim
# ----------------------------------------------------
st.header("Variasi Jumlah Penyewa Berdasarkan Musim")
plt.figure(figsize=(12, 6))
sns.boxplot(data=day_df, x='season', y='cnt', palette='Blues')
plt.title('Variasi Jumlah Penyewa Berdasarkan Musim')
plt.xlabel('Musim')
plt.ylabel('Jumlah Penyewa')
plt.grid(True)
plt.xticks([0, 1, 2, 3], ['Musim Dingin', 'Musim Semi', 'Musim Panas', 'Musim Gugur'])
st.pyplot(plt)

# ----------------------------------------------------
# Pertanyaan 1: Faktor cuaca dan waktu mana yang paling signifikan mempengaruhi jumlah penyewaan sepeda?
# ----------------------------------------------------

st.header("Pertanyaan 1: Faktor cuaca dan waktu mana yang paling signifikan mempengaruhi jumlah penyewaan sepeda?")
st.markdown("""
### Langkah-langkah:
1. Data Wrangling: Memastikan tidak ada missing values dalam data yang dianalisis.
2. Analisis Korelasi: Menilai hubungan antara cuaca (weathersit), suhu (temp), kelembaban (hum), dan kecepatan angin (windspeed) dengan jumlah pengguna sepeda (cnt).
3. Analisis Regresi Linear: Menentukan faktor mana yang paling signifikan mempengaruhi jumlah pengguna sepeda dalam sehari.
4. Visualisasi: Menampilkan visualisasi hubungan antar variabel.
""")

# 1. Korelasi antar variabel
st.subheader("Heatmap Korelasi antara Variabel")
st.markdown("""
**Penjelasan Visualisasi:** Heatmap ini menunjukkan korelasi antar variabel dalam dataset. Warna yang lebih gelap menunjukkan korelasi positif yang lebih kuat, sedangkan warna yang lebih terang menunjukkan korelasi negatif atau korelasi yang lebih lemah. Ini membantu kita memahami variabel mana yang memiliki pengaruh signifikan terhadap jumlah pengguna sepeda (`cnt`).
""")
corr_matrix = day_df[['cnt', 'weathersit', 'temp', 'hum', 'windspeed']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.markdown("""
**Jawaban:** Dari heatmap korelasi, terlihat bahwa suhu (`temp`) memiliki korelasi positif paling kuat dengan jumlah penyewa sepeda (`cnt`). Artinya, semakin tinggi suhu, semakin banyak pengguna sepeda. Di sisi lain, kelembaban (`hum`) dan kecepatan angin (`windspeed`) memiliki korelasi negatif, yang menunjukkan bahwa kondisi cuaca ini cenderung menurunkan jumlah penyewa sepeda. 

Visualisasi ini mendukung bahwa suhu adalah salah satu faktor yang paling mempengaruhi jumlah penyewaan.
""")

# 2. Regresi Linear Sederhana
st.subheader("Analisis Regresi Linear untuk Menentukan Faktor yang Paling Mempengaruhi")
st.markdown("""
**Penjelasan Hasil:** Koefisien dari model regresi linear menunjukkan seberapa besar pengaruh setiap variabel terhadap jumlah pengguna sepeda. Variabel dengan koefisien tertinggi adalah yang paling signifikan mempengaruhi jumlah pengguna. Nilai R-Squared menunjukkan seberapa baik model ini menjelaskan variasi data.
""")
X = day_df[['weathersit', 'temp', 'hum', 'windspeed']]
y = day_df['cnt']

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Menampilkan koefisien
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
st.write(coefficients)

# Prediksi hasil regresi
y_pred = model.predict(X_test)

# Menampilkan hasil model regresi
st.write(f"R-Squared: {r2_score(y_test, y_pred)}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

st.markdown("""
**Jawaban:** Hasil regresi menunjukkan bahwa suhu (`temp`) memiliki koefisien paling tinggi, artinya suhu memiliki dampak terbesar terhadap jumlah penyewaan sepeda. Diikuti oleh faktor cuaca (`weathersit`) dan kelembaban (`hum`). Kecepatan angin (`windspeed`) memiliki pengaruh yang lebih kecil terhadap jumlah pengguna sepeda.

Nilai R-Squared menunjukkan bahwa model ini mampu menjelaskan variasi data dengan cukup baik.
""")

# Visualisasi scatter plot antara suhu dan jumlah pengguna
st.subheader("Hubungan antara Suhu dan Jumlah Pengguna Sepeda")
st.markdown("""
**Penjelasan Visualisasi:** Scatter plot ini menunjukkan hubungan antara suhu (`temp`) dan jumlah pengguna sepeda. Peningkatan suhu cenderung diikuti oleh peningkatan jumlah pengguna sepeda, seperti yang ditunjukkan oleh tren garis regresi. Ini mendukung hasil dari regresi linear yang menunjukkan bahwa suhu adalah salah satu faktor yang paling signifikan mempengaruhi jumlah pengguna sepeda.
""")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='temp', y='cnt', data=day_df, ax=ax2)
plt.title("Hubungan antara Suhu dan Jumlah Pengguna Sepeda")
plt.axhline(np.mean(day_df['cnt']), color='red', linestyle='--', linewidth=1.5, label='Rata-rata Pengguna')
plt.legend()
st.pyplot(fig2)

st.markdown("""
**Jawaban:** Visualisasi ini memperkuat hasil dari analisis sebelumnya bahwa suhu memiliki pengaruh besar terhadap jumlah penyewa sepeda. Terlihat bahwa semakin tinggi suhu, semakin tinggi pula jumlah pengguna sepeda.
""")

# ----------------------------------------------------
# Pertanyaan 2: Apakah ada perbedaan pola penyewaan sepeda berdasarkan musim dan hari (hari kerja vs hari libur)?
# ----------------------------------------------------

st.header("Pertanyaan 2: Apakah ada perbedaan pola penyewaan sepeda berdasarkan musim dan hari (hari kerja vs hari libur)?")
st.markdown("""
### Langkah-langkah:
1. Pembagian Data: Pisahkan data menjadi hari kerja dan hari libur, serta musim yang berbeda.
2. Statistik Deskriptif: Melakukan perhitungan statistik deskriptif untuk melihat ringkasan data pada hari kerja dan hari libur serta musim yang berbeda.
3. Uji Statistik: Menggunakan T-Test untuk melihat apakah ada perbedaan signifikan dalam jumlah pengguna sepeda berdasarkan hari dan musim.
4. Visualisasi: Membuat visualisasi perbandingan antara musim dan hari kerja/hari libur.
""")

# Membagi data berdasarkan workingday dan holiday
workingday_data = day_df[day_df['workingday'] == 1]['cnt']
holiday_data = day_df[day_df['holiday'] == 1]['cnt']

# Statistik Deskriptif
st.subheader("Statistik Deskriptif Jumlah Pengguna Sepeda pada Hari Kerja dan Hari Libur")
st.markdown("""
**Penjelasan Hasil:** Statistik deskriptif memberikan ringkasan distribusi data untuk hari kerja dan hari libur. Kita dapat melihat bahwa jumlah pengguna sepeda lebih tinggi pada hari kerja dibandingkan hari libur, namun kita perlu melakukan uji statistik untuk mengetahui apakah perbedaan ini signifikan.
""")
st.write("Hari Kerja:", workingday_data.describe())
st.write("Hari Libur:", holiday_data.describe())

# Uji Statistik (T-Test) untuk Mengetahui Perbedaan Signifikan
st.subheader("Uji Statistik (T-Test) untuk Mengetahui Perbedaan Signifikan")
st.markdown("""
**Penjelasan Hasil:** Uji T-Test membantu kita menentukan apakah perbedaan jumlah pengguna sepeda antara hari kerja dan hari libur signifikan secara statistik. Jika nilai p-value lebih kecil dari 0.05, kita bisa menyimpulkan bahwa perbedaannya signifikan.
""")
t_stat, p_value = ttest_ind(workingday_data, holiday_data)
st.write(f"T-Test: t-statistic = {t_stat}, p-value = {p_value}")

st.markdown("""
**Jawaban:** Hasil uji T-Test menunjukkan bahwa ada perbedaan yang signifikan secara statistik antara jumlah pengguna sepeda pada hari kerja dan hari libur (p-value < 0.05). Hal ini menunjukkan bahwa pengguna sepeda cenderung lebih banyak pada hari kerja dibandingkan hari libur.
""")

# Visualisasi Perbandingan Pengguna antara Hari Kerja dan Hari Libur
st.subheader("Visualisasi Perbandingan Pengguna Sepeda antara Hari Kerja dan Hari Libur")
st.markdown("""
**Penjelasan Visualisasi:** Boxplot ini membantu kita memahami perbedaan distribusi jumlah pengguna sepeda pada hari kerja dan hari libur. Jika median pada hari kerja lebih tinggi daripada hari libur, hal ini menunjukkan bahwa pengguna sepeda cenderung lebih banyak pada hari kerja.
""")
fig3, ax3 = plt.subplots()
sns.boxplot(data=[workingday_data, holiday_data], ax=ax3, palette=["lightblue", "lightcoral"])
plt.xticks([0, 1], ['Hari Kerja', 'Hari Libur'])
plt.title("Perbandingan Jumlah Pengguna Sepeda")
plt.axhline(np.median(workingday_data), color='blue', linestyle='--', linewidth=1.5, label='Median Hari Kerja')
plt.axhline(np.median(holiday_data), color='red', linestyle='--', linewidth=1.5, label='Median Hari Libur')
plt.legend()
st.pyplot(fig3)

st.markdown("""
**Jawaban:** Dari boxplot di atas, terlihat bahwa median jumlah pengguna sepeda lebih tinggi pada hari kerja dibandingkan hari libur. Ini mendukung hasil dari uji statistik bahwa pengguna sepeda lebih banyak pada hari kerja daripada hari libur.
""")

# ----------------------------------------------------
# Analisis Lanjutan: Pola Penggunaan Sepeda Sepanjang Hari dalam Seminggu
# ----------------------------------------------------
st.header("Analisis Lanjutan: Pola Penggunaan Sepeda Sepanjang Hari dalam Seminggu")

st.markdown("""
Pada bagian ini, kita akan mengeksplorasi bagaimana pola penggunaan sepeda bervariasi berdasarkan waktu dalam sehari dan hari dalam seminggu. 
Ini akan memberikan gambaran kapan waktu penggunaan sepeda tertinggi dan hari mana yang memiliki jumlah pengguna tertinggi.
""")

# Mengelompokkan data berdasarkan hari dalam seminggu dan jam dalam sehari
hour_df['weekday'] = hour_df['weekday'].replace({
    0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'
})

# Membuat heatmap untuk melihat pola penggunaan sepeda berdasarkan jam dan hari
st.subheader("Heatmap Penggunaan Sepeda Berdasarkan Hari dan Jam")
plt.figure(figsize=(12, 6))
pivot_table = hour_df.pivot_table(values='cnt', index='hr', columns='weekday', aggfunc='mean')

sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.1f', linewidths=.5)
plt.title("Rata-rata Jumlah Pengguna Sepeda Berdasarkan Jam dan Hari")
plt.xlabel("Hari")
plt.ylabel("Jam")
st.pyplot(plt)

st.markdown("""
**Hasil Analisis:** Dari heatmap di atas, kita dapat melihat bahwa penggunaan sepeda mencapai puncaknya pada jam sibuk, yaitu antara pukul 8 pagi hingga 9 pagi dan sore hari antara pukul 5 sore hingga 7 malam. Terlihat bahwa pada hari kerja (Senin-Jumat), penggunaan sepeda lebih tinggi dibandingkan akhir pekan (Sabtu-Minggu), terutama pada jam-jam sibuk di pagi dan sore hari.
""")

# ----------------------------------------------------
# Analisis Lanjutan: Pola Musiman Penggunaan Sepeda
# ----------------------------------------------------
st.header("Analisis Lanjutan: Pola Musiman Penggunaan Sepeda")

st.markdown("""
Pada bagian ini, kita akan melihat bagaimana pola penggunaan sepeda bervariasi berdasarkan musim. Ini akan membantu mengidentifikasi apakah ada musim tertentu yang memiliki jumlah pengguna sepeda lebih banyak daripada yang lain.
""")

# Rata-rata pengguna sepeda berdasarkan musim
season_avg = day_df.groupby('season')['cnt'].mean().reset_index()

# Mengubah kode musim menjadi label yang lebih informatif
season_avg['season'] = season_avg['season'].replace({
    1: 'Musim Dingin', 2: 'Musim Semi', 3: 'Musim Panas', 4: 'Musim Gugur'
})

# Membuat visualisasi barplot
st.subheader("Rata-rata Jumlah Pengguna Sepeda Berdasarkan Musim")
plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='cnt', data=season_avg, palette='Blues')
plt.title("Rata-rata Jumlah Pengguna Sepeda Berdasarkan Musim")
plt.xlabel("Musim")
plt.ylabel("Rata-rata Jumlah Pengguna")
st.pyplot(plt)

st.markdown("""
**Hasil Analisis:** Visualisasi di atas menunjukkan bahwa penggunaan sepeda paling tinggi terjadi pada **Musim Panas** dan **Musim Semi**, sementara penggunaan sepeda paling rendah terjadi pada **Musim Dingin**. Hal ini mungkin disebabkan oleh cuaca yang lebih baik pada musim panas dan semi yang mendorong orang untuk lebih sering bersepeda.
""")

# ----------------------------------------------------
# Analisis Lanjutan: Pengaruh Cuaca Ekstrem terhadap Penggunaan Sepeda
# ----------------------------------------------------
st.header("Analisis Lanjutan: Pengaruh Cuaca Ekstrem terhadap Penggunaan Sepeda")

st.markdown("""
Pada bagian ini, kita akan melihat bagaimana kondisi cuaca ekstrem (misalnya hujan lebat atau badai) memengaruhi jumlah pengguna sepeda. 
Apakah pengguna sepeda berkurang secara signifikan saat cuaca ekstrem terjadi?
""")

# Membuat kategori cuaca ekstrem untuk analisis
extreme_weather = day_df[day_df['weathersit'] == 3]  # 3 = kondisi cuaca ekstrem (hujan lebat atau badai)

# Menghitung rata-rata jumlah pengguna pada hari-hari dengan cuaca ekstrem
extreme_weather_avg = extreme_weather['cnt'].mean()

# Menghitung rata-rata jumlah pengguna pada semua hari
overall_avg = day_df['cnt'].mean()

# Membandingkan cuaca ekstrem dengan rata-rata keseluruhan
st.subheader("Pengaruh Cuaca Ekstrem terhadap Penggunaan Sepeda")
st.write(f"Rata-rata jumlah pengguna sepeda pada cuaca ekstrem: {extreme_weather_avg:.2f}")
st.write(f"Rata-rata jumlah pengguna sepeda keseluruhan: {overall_avg:.2f}")

# Visualisasi perbandingan
st.subheader("Perbandingan Penggunaan Sepeda pada Cuaca Ekstrem vs Rata-rata Keseluruhan")
plt.figure(figsize=(8, 6))
categories = ['Rata-rata Keseluruhan', 'Cuaca Ekstrem']
values = [overall_avg, extreme_weather_avg]

sns.barplot(x=categories, y=values, palette='coolwarm')
plt.title("Perbandingan Penggunaan Sepeda pada Cuaca Ekstrem vs Rata-rata")
plt.ylabel("Rata-rata Jumlah Pengguna Sepeda")
st.pyplot(plt)

st.markdown("""
**Hasil Analisis:** Analisis ini menunjukkan bahwa jumlah pengguna sepeda secara signifikan berkurang saat terjadi cuaca ekstrem seperti hujan lebat atau badai. Rata-rata pengguna sepeda pada hari-hari dengan cuaca ekstrem jauh lebih rendah daripada rata-rata keseluruhan. Hal ini wajar karena cuaca buruk membuat orang enggan untuk bersepeda.
""")

# ----------------------------------------------------
# Kesimpulan Akhir dari Analisis
# ----------------------------------------------------
st.header("Kesimpulan Akhir dari Analisis")
st.markdown("""
### Kesimpulan dari Pertanyaan 1:
- Faktor yang paling mempengaruhi jumlah pengguna sepeda adalah suhu (`temp`). Berdasarkan analisis korelasi dan regresi, suhu memiliki pengaruh paling besar terhadap jumlah total pengguna sepeda dalam sehari. Kelembaban (`hum`) dan kecepatan angin (`windspeed`) juga berpengaruh, tetapi dalam skala yang lebih kecil.
- Korelasi dan scatter plot mendukung bahwa semakin tinggi suhu, semakin tinggi pula jumlah pengguna sepeda.

### Kesimpulan dari Pertanyaan 2:
- Berdasarkan hasil uji statistik, terdapat perbedaan yang signifikan dalam jumlah pengguna sepeda antara hari kerja dan hari libur. Hari kerja cenderung memiliki lebih banyak pengguna sepeda dibandingkan hari libur, yang dapat dilihat dari statistik deskriptif dan visualisasi boxplot.
- P-value dari uji T-Test kurang dari 0.05, yang berarti perbedaan tersebut signifikan secara statistik.

### Kesimpulan dari Analisis Lanjutan:
1. **Pola Penggunaan Sepeda Sepanjang Hari dalam Seminggu:** 
   - Penggunaan sepeda paling tinggi terjadi pada jam sibuk (pagi dan sore hari) pada hari kerja.
   - Penggunaan sepeda menurun pada akhir pekan dan tidak ada puncak penggunaan yang signifikan.

2. **Pola Musiman Penggunaan Sepeda:** 
   - Penggunaan sepeda paling tinggi pada Musim Panas dan Musim Semi.
   - Penggunaan sepeda paling rendah terjadi pada Musim Dingin.

3. **Pengaruh Cuaca Ekstrem:** 
   - Cuaca ekstrem secara signifikan mengurangi jumlah pengguna sepeda.
   - Rata-rata pengguna sepeda jauh lebih rendah pada hari-hari dengan hujan lebat atau badai dibandingkan dengan rata-rata keseluruhan.
""")

# ----------------------------------------------------
# Akhir Program
# ----------------------------------------------------
