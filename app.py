import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Amazon")
st.markdown("---")

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')  # sesuaikan nama file kamu
    df = df[['Score', 'Time', 'Summary', 'Text']].dropna(subset=['Text'])
    df = df.drop_duplicates(subset=['Text'])
    df['Date'] = pd.to_datetime(df['Time'], unit='s')
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)

    def label_sentimen(score):
        if score >= 4: return 'Positif'
        elif score == 3: return 'Netral'
        else: return 'Negatif'

    def analisis_nlp(teks):
        try:
            p = TextBlob(str(teks)).sentiment.polarity
            if p > 0: return 'Positif'
            elif p == 0: return 'Netral'
            else: return 'Negatif'
        except:
            return 'Netral'

    df['Sentimen'] = df['Score'].apply(label_sentimen)
    df['Sentimen_NLP'] = df['Text'].apply(analisis_nlp)
    return df

with st.spinner('Memuat data... (1-2 menit pertama kali)'):
    df = load_data()

st.success(f"Data berhasil dimuat! Total: {len(df):,} ulasan")

# =====================
# SIDEBAR FILTER
# =====================
st.sidebar.header("🔧 Filter Data")
sentimen_filter = st.sidebar.multiselect(
    "Pilih Sentimen:",
    options=['Positif', 'Netral', 'Negatif'],
    default=['Positif', 'Netral', 'Negatif']
)
score_filter = st.sidebar.multiselect(
    "Pilih Rating (Score):",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5]
)

df_filtered = df[
    df['Sentimen_NLP'].isin(sentimen_filter) &
    df['Score'].isin(score_filter)
]

# =====================
# METRIC CARDS
# =====================
st.subheader("📈 Ringkasan")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Ulasan", f"{len(df_filtered):,}")
col2.metric("Positif", f"{(df_filtered['Sentimen_NLP']=='Positif').sum():,}")
col3.metric("Netral", f"{(df_filtered['Sentimen_NLP']=='Netral').sum():,}")
col4.metric("Negatif", f"{(df_filtered['Sentimen_NLP']=='Negatif').sum():,}")

st.markdown("---")

# =====================
# CHART DISTRIBUSI
# =====================
st.subheader("📊 Distribusi Sentimen")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = {'Positif': '#2ecc71', 'Netral': '#f39c12', 'Negatif': '#e74c3c'}
    counts = df_filtered['Sentimen_NLP'].value_counts()
    ax.bar(counts.index, counts.values,
           color=[colors.get(s, 'gray') for s in counts.index],
           edgecolor='white')
    ax.set_title('Jumlah per Sentimen')
    ax.set_ylabel('Jumlah Ulasan')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    counts.plot(kind='pie', ax=ax,
                colors=[colors.get(s, 'gray') for s in counts.index],
                autopct='%1.1f%%', startangle=90)
    ax.set_ylabel('')
    ax.set_title('Proporsi Sentimen')
    st.pyplot(fig)

st.markdown("---")

# =====================
# CHART TREN
# =====================
st.subheader("📅 Tren Sentimen per Bulan")
tren = df_filtered.groupby(['YearMonth', 'Sentimen_NLP']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 4))
for sentimen, warna in colors.items():
    if sentimen in tren.columns:
        ax.plot(tren.index, tren[sentimen],
                label=sentimen, color=warna, linewidth=2, marker='o', markersize=3)
ax.set_title('Tren Sentimen dari Waktu ke Waktu')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Ulasan')
ax.legend()
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# =====================
# PREDIKSI REAL-TIME
# =====================
st.subheader("🤖 Coba Prediksi Sentimen")
teks_input = st.text_area("Masukkan teks ulasan:", "This product is absolutely amazing!")

if st.button("Analisis Sentimen ↗"):
    polarity = TextBlob(teks_input).sentiment.polarity
    if polarity > 0:
        hasil = "Positif 😊"
        warna = "green"
    elif polarity == 0:
        hasil = "Netral 😐"
        warna = "orange"
    else:
        hasil = "Negatif 😞"
        warna = "red"
    
    st.markdown(f"**Hasil:** :{warna}[{hasil}]")
    st.markdown(f"**Polarity score:** `{polarity:.3f}`")