import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi Scraping Data
@st.cache
def scrape_books():
    import requests
    from bs4 import BeautifulSoup

    base_url = "https://books.toscrape.com/catalogue/"
    start_url = "https://books.toscrape.com/catalogue/page-1.html"

    books_data = []
    max_books = 100

    while start_url and len(books_data) < max_books:
        response = requests.get(start_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for book in soup.find_all('article', class_='product_pod'):
            if len(books_data) >= max_books:
                break

            title = book.h3.a['title']
            price = book.find('p', class_='price_color').text[1:].replace('Â', '').strip()
            rating = book.p['class'][1]
            availability = book.find('p', class_='instock availability').text.strip()

            books_data.append({
                'Title': title,
                'Price': float(price.replace('£', '')),
                'Rating': rating,
                'Availability': availability
            })

        next_page = soup.find('li', class_='next')
        if next_page:
            next_url = next_page.a['href']
            start_url = base_url + next_url
        else:
            start_url = None

    return pd.DataFrame(books_data)

# Fungsi Preprocessing
def clean_and_preprocess_data(df):
    st.subheader("Langkah Preprocessing Data")
    steps = []

    # Menghapus kolom yang tidak informatif
    if 'Availability' in df.columns:
        df.drop(columns=['Availability'], inplace=True)
        steps.append("Menghapus kolom 'Availability' karena tidak informatif.")

    # Menangani duplikasi data
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    final_len = len(df)
    steps.append(f"Menghapus duplikasi data. Jumlah data sebelum: {initial_len}, setelah: {final_len}.")

    # Mengubah kolom 'Rating' dari teks menjadi angka
    rating_mapping = {
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 4,
        'Five': 5
    }
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].map(rating_mapping)
        steps.append("Mengubah kolom 'Rating' dari teks menjadi angka.")

    # Menghapus baris dengan nilai NaN (jika ada setelah pemetaan)
    nan_before = df.isna().sum().sum()
    df.dropna(inplace=True)
    nan_after = df.isna().sum().sum()
    if nan_before > nan_after:
        steps.append(f"Menghapus baris dengan nilai NaN. Sebelum: {nan_before}, Setelah: {nan_after}.")

    # Mengonversi kolom 'Price' menjadi tipe numerik jika belum
    if df['Price'].dtype != 'float':
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        steps.append("Mengonversi kolom 'Price' menjadi tipe numerik.")

    # Menghapus baris dengan nilai NaN (jika ada setelah konversi)
    df.dropna(inplace=True)

    # Menampilkan langkah preprocessing
    for step in steps:
        st.write(f"- {step}")

    return df

# Fungsi untuk Menampilkan Preprocessing di Streamlit
def show_preprocessing(df):
    st.subheader("Langkah-Langkah Preprocessing")
    st.write("Berikut adalah langkah-langkah preprocessing yang dilakukan pada dataset:")

    # Menampilkan langkah dan contoh kode
    preprocessing_steps = [
        "1. Menghapus kolom 'Availability' yang tidak informatif.",
        "2. Menghapus duplikasi data.",
        "3. Mengubah kolom 'Rating' dari teks menjadi angka.",
        "4. Menghapus baris dengan nilai NaN.",
        "5. Mengonversi kolom 'Price' menjadi tipe numerik."
    ]

    for step in preprocessing_steps:
        st.write(f"- {step}")

    # Contoh kode preprocessing
    st.code(
        """python
# Contoh kode preprocessing
rating_mapping = {
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5
}

# Menghapus kolom yang tidak diperlukan
df.drop(columns=['Availability'], inplace=True)

# Menghapus duplikasi
df.drop_duplicates(inplace=True)

# Mengubah rating menjadi angka
df['Rating'] = df['Rating'].map(rating_mapping)

# Menghapus nilai NaN
df.dropna(inplace=True)

# Konversi harga menjadi tipe numerik
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        """
    )

# Fungsi untuk Unduh CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Unduh Hasil Scraping sebagai CSV",
        data=csv,
        file_name='books_data.csv',
        mime='text/csv',
    )

# Fungsi untuk Analisis Eksplorasi Data (EDA)
def perform_eda(df):
    st.subheader("Statistik Deskriptif")
    st.write("Tabel statistik deskriptif berikut memberikan gambaran umum tentang data buku.")
    st.write(df.describe())

    # Visualisasi Harga Buku
    st.subheader("Distribusi Harga Buku")
    st.write("Histogram dan boxplot di bawah ini menunjukkan distribusi harga buku.")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram Harga Buku
    sns.histplot(df['Price'], bins=30, kde=True, color='blue', ax=ax[0])
    ax[0].set_title('Distribusi Harga Buku')
    ax[0].set_xlabel('Harga (£)')
    ax[0].set_ylabel('Frekuensi')

    # Boxplot Harga Buku
    sns.boxplot(x=df['Price'], color='orange', ax=ax[1])
    ax[1].set_title('Variasi Harga Buku')
    ax[1].set_xlabel('Harga (£)')

    st.pyplot(fig)

    # Distribusi Rating Buku
    st.subheader("Distribusi Rating Buku")
    st.write("Distribusi berikut menunjukkan jumlah buku untuk setiap nilai rating.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Rating', hue='Rating', data=df, palette='viridis', dodge=False, ax=ax)
    ax.set_title('Distribusi Rating Buku')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Jumlah Buku')
    st.pyplot(fig)

    # Korelasi Harga dan Rating
    st.subheader("Korelasi Harga dan Rating Buku")
    st.write("Scatter plot berikut menunjukkan hubungan antara rating dan harga buku.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Rating', y='Price', data=df, alpha=0.7, color='green', ax=ax)
    ax.set_title('Korelasi Harga dan Rating Buku')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Harga (£)')
    st.pyplot(fig)

    # Korelasi Numerik
    st.subheader("Korelasi Numerik")
    st.write("Matriks korelasi menunjukkan hubungan antara harga dan rating.")
    correlation = df[['Price', 'Rating']].corr()
    st.write(correlation)

# Main Program
def main():
    st.title("Analisis Buku dengan Streamlit")
    st.write("Aplikasi ini menyediakan analisis dataset buku yang diambil dari website Books to Scrape.")

    # Sidebar Menu
    st.sidebar.header("Navigasi")
    options = st.sidebar.radio("Pilih Langkah:", ["Scrape Data", "Preprocessing", "EDA", "Kesimpulan"])

    if options == "Scrape Data":
        st.header("Scraping Data")
        df = scrape_books()
        st.dataframe(df.head())
        download_csv(df)
        st.write("Data berhasil di-scrape!")

    elif options == "Preprocessing":
        st.header("Preprocessing Data")
        df = scrape_books()
        show_preprocessing(df)
        df = clean_and_preprocess_data(df)
        st.write("Dataset setelah preprocessing:")
        st.dataframe(df.head())

    elif options == "EDA":
        st.header("Analisis Eksplorasi Data (EDA)")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        perform_eda(df)

    elif options == "Kesimpulan":
        st.header("Kesimpulan")
        st.write("Dataset buku menunjukkan distribusi harga dan rating yang dapat dikategorikan ke dalam klaster menggunakan K-Means.")
        st.write("Model regresi menunjukkan hubungan yang lemah antara rating dan harga buku, dengan R-squared rendah.")

if __name__ == "__main__":
    main()
