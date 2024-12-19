import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Fungsi Scraping Data
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

# Fungsi untuk Menjalankan Skenario
def scenario_prediction(cleaned_books_data):
    scenario_option = st.selectbox("Pilih Skenario:", ["Skenario 1: Prediksi Harga Buku", "Skenario 2: Prediksi Ketersediaan Stok"])

    if scenario_option == "Skenario 1: Prediksi Harga Buku":
        st.subheader("Skenario 1: Prediksi Harga Buku berdasarkan Rating")

        # Membagi fitur dan target
        X = cleaned_books_data[['Rating']]
        y = cleaned_books_data['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Dropdown untuk memilih model
        model_option = st.selectbox("Pilih Model:", ["Linear Regression", "Decision Tree Regression"])

        if model_option == "Linear Regression":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(random_state=42)

        # Melatih model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared:** {r2:.2f}")

        # Visualisasi Prediksi
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_title(f"Hasil Prediksi dengan {model_option}")
        ax.set_xlabel("Harga Aktual")
        ax.set_ylabel("Harga Prediksi")
        st.pyplot(fig)

    elif scenario_option == "Skenario 2: Prediksi Ketersediaan Stok":
        st.subheader("Skenario 2: Prediksi Ketersediaan Stok")

        # Membuat label ketersediaan berdasarkan harga buku
        cleaned_books_data['Availability'] = cleaned_books_data['Price'].apply(lambda x: 'In stock' if x > 50 else 'Out of stock')

        # Mengambil fitur dan target
        X_class = cleaned_books_data[['Price', 'Rating']]
        y_class = cleaned_books_data['Availability'].map({'In stock': 1, 'Out of stock': 0})
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

        # Dropdown untuk memilih model
        model_option_class = st.selectbox("Pilih Model Klasifikasi:", ["Random Forest Classifier", "Logistic Regression"])

        if model_option_class == "Random Forest Classifier":
            model_class = RandomForestClassifier(random_state=42)
        else:
            model_class = LogisticRegression()

        # Melatih model
        model_class.fit(X_train_class, y_train_class)
        y_pred_class = model_class.predict(X_test_class)

        # Evaluasi Model
        accuracy = accuracy_score(y_test_class, y_pred_class)
        report = classification_report(y_test_class, y_pred_class, output_dict=True)

        st.write(f"**Akurasi:** {accuracy:.2f}")
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())

        # Visualisasi Confusion Matrix
        cm = confusion_matrix(y_test_class, y_pred_class)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
