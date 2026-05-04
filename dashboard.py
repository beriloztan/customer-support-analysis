import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="Customer Support Analytics", layout="wide")

# Modelleri yükle
with open('data/kategori_model.pkl', 'rb') as f:
    kategori_model = pickle.load(f)
with open('data/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('data/sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)
with open('data/tfidf_sentiment.pkl', 'rb') as f:
    tfidf_sentiment = pickle.load(f)

# Veriyi yükle
df = pd.read_csv('data/musteri_tweetleri.csv', nrows=200000)

# Başlık
st.title("Customer Support Analytics Dashboard")
st.markdown("Twitter üzerinden gelen müşteri şikayetlerinin analizi — 200.000 tweet.")

# Üst metrikler
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Tweet", f"{len(df):,}")
col2.metric("Şikayet Kategorisi", "7")
col3.metric("Şirket Sayısı", "108")

# Grafikler
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        df['kategori'].value_counts().reset_index(),
        x='kategori', y='count',
        title='Şikayet Kategorileri',
        color='kategori'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    kategori_dagilim = df['kategori'].value_counts().reset_index()
    kategori_dagilim.columns = ['Kategori', 'Tweet Sayısı']
    fig2 = px.pie(kategori_dagilim, names='Kategori', values='Tweet Sayısı',
                  title='Kategori Dağılımı',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig2, use_container_width=True)

# Canlı Tweet Analizi
st.subheader("Canlı Tweet Analizi")
st.markdown("Bir tweet gir, kategori ve sentiment tahmin edelim.")

tweet = st.text_input("Tweet:")

if tweet:
    # Temizle
    import re
    clean = re.sub(r'@\w+', '', tweet)
    clean = re.sub(r'http\S+', '', clean).lower().strip()

    # Kategori tahmini
    tweet_tfidf = tfidf.transform([clean])
    kategori = kategori_model.predict(tweet_tfidf)[0]

    # Sentiment tahmini
    tweet_tfidf_s = tfidf_sentiment.transform([clean])
    sentiment = sentiment_model.predict(tweet_tfidf_s)[0]

    col1, col2 = st.columns(2)
    col1.metric("Kategori", kategori)
    if sentiment == 'Pozitif':
        col2.success("Sentiment: Pozitif 😊")
    else:
        col2.error("Sentiment: Negatif 😞")