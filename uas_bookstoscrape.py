# -*- coding: utf-8 -*-
"""UAS bookstoscrape.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mimcDUf1vn9XvBonfJa1pKwsvBFRY7ep

#Import Library
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

"""#Scraping data"""

def scrape_books():
    base_url = "https://books.toscrape.com/catalogue/"
    start_url = "https://books.toscrape.com/catalogue/page-1.html"

    books_data = []
    max_books = 400

    while start_url and len(books_data) < max_books:
        print(f"Scraping: {start_url}")
        response = requests.get(start_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract book data from the current page
        for book in soup.find_all('article', class_='product_pod'):
            if len(books_data) >= max_books:
                break

            title = book.h3.a['title']
            price = book.find('p', class_='price_color').text[1:].replace('Â', '').strip()  # Remove the currency symbol and clean text
            rating = book.p['class'][1]  # Rating is in the second class attribute
            availability = book.find('p', class_='instock availability').text.strip()

            # Append book data to the list
            books_data.append({
                'Title': title,
                'Price': float(price.replace('£', '')),
                'Rating': rating,
                'Availability': availability
            })

        # Find the next page link
        next_page = soup.find('li', class_='next')
        if next_page:
            next_url = next_page.a['href']
            start_url = base_url + next_url
        else:
            start_url = None

    # Convert to DataFrame for easier manipulation
    books_df = pd.DataFrame(books_data)
    return books_df

# Scrape data and save to a CSV file
books_df = scrape_books()
books_df.to_csv('books_data.csv', index=False)

df = pd.read_csv('books_data.csv')
df.head()

"""#Preprocessing dan Pembersihan Data"""

def clean_and_preprocess_data(df):
    # Menghapus kolom yang tidak informatif
    if 'Availability' in df.columns:
        df.drop(columns=['Availability'], inplace=True)
        # Menangani duplikasi data
    df.drop_duplicates(inplace=True)
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

    # Menghapus baris dengan nilai NaN (jika ada setelah pemetaan)
    df.dropna(inplace=True)

    # Mengonversi kolom 'Price' menjadi tipe numerik jika belum
    if df['Price'].dtype != 'float':
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Menghapus baris dengan nilai NaN (jika ada setelah konversi)
    df.dropna(inplace=True)

    return df

# Membersihkan data
cleaned_books_data = clean_and_preprocess_data(df)
cleaned_books_data.info(), cleaned_books_data.head()

"""Exploratory Data Analysis (EDA)"""

import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    # Statistik Deskriptif
    print("Statistik Deskriptif:")
    print(df.describe())

    #Visualisasi harga buku
    plt.figure(figsize=(12, 6))
    #histogram harga buku
    plt.subplot(1, 2, 1)
    sns.histplot(df['Price'], bins=30, kde=True, color='blue')
    plt.title('Distribusi Harga Buku')
    plt.xlabel('Harga (£)')
    plt.ylabel('Frekuensi')

    #Boxplot harga buku
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['Price'], color='orange')
    plt.title('Variasi Harga Buku')
    plt.xlabel('Harga (£)')
    plt.tight_layout()
    plt.show()

    #Analisis Distribusi Rating
    plt.figure(figsize=(8, 6))
    #sns.countplot(x=df['Rating'], palette='viridis')
    sns.countplot(x='Rating', hue='Rating', data=df, palette='viridis', dodge=False)
    plt.title('Distribusi Rating Buku')
    plt.xlabel('Rating')
    plt.ylabel('Jumlah Buku')
    plt.show()

    #Korelasi Harga dan Rating
    # Korelasi Harga dan Rating
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Rating', y='Price', data=df, alpha=0.7, color='green')
    plt.title('Korelasi Harga dan Rating Buku')
    plt.xlabel('Rating')
    plt.ylabel('Harga (£)')
    plt.show()


    #Korelasi numerik
    correlation = df[['Price', 'Rating']].corr()
    print("Korelasi Harga dan Rating:")
    print(correlation)

# Jalankan EDA
perform_eda(cleaned_books_data)

"""##Skenario 1: Prediksi Harga Buku berdasarkan Rating (Regresi)
- Algoritma 1: Linear Regression
- Algoritma 2: Decision Tree Regression
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Mengambil fitur dan target
X = cleaned_books_data[['Rating']]  # Fitur: Rating
y = cleaned_books_data['Price']    # Target: Harga

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Linear Regression
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)
y_pred_reg = model_reg.predict(X_test)

# Evaluasi model Linear Regression
mse_reg = mean_squared_error(y_test, y_pred_reg)
r2_reg = r2_score(y_test, y_pred_reg)
print(f"Mean Squared Error (Linear Regression): {mse_reg}")
print(f"R-squared (Linear Regression): {r2_reg}")

# Model Decision Tree Regression
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Evaluasi model Decision Tree Regression
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Mean Squared Error (Decision Tree Regression): {mse_dt}")
print(f"R-squared (Decision Tree Regression): {r2_dt}")

"""Visualisasi Perbandingan Model Linear Regression dan Decision Tree Regression"""

# Membandingkan Linear Regression dan Decision Tree Regression
model_tree = DecisionTreeRegressor(random_state=42)
model_tree.fit(X_train, y_train)

# Prediksi dengan Decision Tree Regression
y_pred_tree = model_tree.predict(X_test)

# Evaluasi model Decision Tree Regression
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

# Metrik Evaluasi
models = ['Linear Regression', 'Decision Tree Regression']
mse_scores = [mse_reg, mse_tree]
r2_scores = [r2_reg, r2_tree]

# Visualisasi Perbandingan MSE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'orange'])
plt.title('Perbandingan Mean Squared Error (MSE)')
plt.ylabel('MSE')
for i, v in enumerate(mse_scores):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center')

# Visualisasi Perbandingan R-squared
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'orange'])
plt.title('Perbandingan R-squared')
plt.ylabel('R-squared')
plt.ylim(-0.03, 0.0)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.0, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.show()

# Menentukan model yang lebih optimal
if mse_reg < mse_tree:
    print("Model Linear Regression lebih optimal berdasarkan MSE.")
else:
    print("Model Decision Tree Regression lebih optimal berdasarkan MSE.")

if r2_reg > r2_tree:
    print("Model Linear Regression lebih optimal berdasarkan R-squared.")
else:
    print("Model Decision Tree Regression lebih optimal berdasarkan R-squared.")

"""Hasil menunjukan perbandingan kinerja antara Linear Regression dan Decision Tree Regression dalam memprediksi harga buku berdasarkan rating. Berdasarkan evaluasi menggunakan Mean Squared Error (MSE) dan R-squared (R²), model Linear Regression memberikan hasil yang lebih optimal dengan MSE yang lebih rendah dan R² yang lebih tinggi dibandingkan Decision Tree Regression. Hal ini menunjukkan bahwa Linear Regression lebih mampu menangkap pola linear antara rating dan harga buku, sementara Decision Tree Regression cenderung menghasilkan model yang lebih kompleks dan kurang stabil pada data yang lebih sederhana. Oleh karena itu, model Linear Regression lebih disarankan untuk digunakan dalam kasus ini, karena lebih efektif dalam menghasilkan prediksi harga yang akurat dengan kesalahan yang lebih kecil.

##Skenario 2: Prediksi Ketersediaan Stok (In Stock vs Out of Stock) (Klasifikasi)
- Algoritma 1: Random Forest Classifier
- Algoritma 2: Logistic Regression
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Membuat Availability berdasar harga buku
cleaned_books_data['Availability'] = cleaned_books_data['Price'].apply(lambda x: 'In stock' if x > 50 else 'Out of stock')

# Mengambil fitur dan target
X_class = cleaned_books_data[['Price', 'Rating']]  # Fitur: Harga dan Rating
y_class = cleaned_books_data['Availability']  # Target: Ketersediaan

# Encode target (In stock / Out of stock) menjadi angka
y_class = y_class.map({'In stock': 1, 'Out of stock': 0})

# Membagi data menjadi data latih dan data uji
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Model Random Forest Classifier
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_class, y_train_class)
y_pred_rf = model_rf.predict(X_test_class)

# Evaluasi model Random Forest
accuracy_rf = accuracy_score(y_test_class, y_pred_rf)
print(f"Accuracy (Random Forest Classifier): {accuracy_rf}")
print("Classification Report (Random Forest):\n", classification_report(y_test_class, y_pred_rf))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test_class, y_pred_rf))

# Model Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train_class, y_train_class)
y_pred_lr = model_lr.predict(X_test_class)

# Evaluasi model Logistic Regression
accuracy_lr = accuracy_score(y_test_class, y_pred_lr)
print(f"Accuracy (Logistic Regression): {accuracy_lr}")
print("Classification Report (Logistic Regression):\n", classification_report(y_test_class, y_pred_lr))
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test_class, y_pred_lr))

"""Visualisasi Perbandingan Model Random Forest Classifier dan Logistic Regression"""

conf_matrix_log_reg = confusion_matrix(y_test_class, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test_class, y_pred_rf)

# Visualisasi Perbandingan Accuracy
plt.figure(figsize=(12, 6))
plt.bar(['Logistic Regression', 'Random Forest'], [accuracy_lr, accuracy_rf], color=['blue', 'green'])
plt.title('Perbandingan Accuracy')
plt.ylabel('Accuracy')
for i, v in enumerate([accuracy_lr, accuracy_rf]):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.show()

# Visualisasi Perbandingan Confusion Matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Confusion Matrix Logistic Regression
sns.heatmap(conf_matrix_log_reg, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title('Confusion Matrix - Logistic Regression')
axs[0].set_xlabel('Predicted')
axs[0].set_ylabel('True')

# Confusion Matrix Random Forest
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens", ax=axs[1])
axs[1].set_title('Confusion Matrix - Random Forest')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')

plt.tight_layout()
plt.show()

# Menentukan model yang lebih optimal
if accuracy_lr > accuracy_rf:
    print("Model Logistic Regression lebih optimal berdasarkan accuracy.")
else:
    print("Model Random Forest lebih optimal berdasarkan accuracy.")

"""Hasil menunjukkan perbandingan antara model Logistic Regression dan Random Forest Classifier dalam memprediksi ketersediaan buku (In stock vs. Out of stock) berdasarkan harga dan rating. Berdasarkan evaluasi menggunakan accuracy dan confusion matrix kedua model mencapai hasil yang sangat baik dengan accuracy 1.0, yang berarti keduanya berhasil memprediksi semua data uji dengan sempurna. Classification report juga menunjukkan hasil yang sangat baik dengan precision, recall, dan f1-score masing-masing bernilai 1.0 untuk kedua model, mengindikasikan bahwa keduanya mampu mengklasifikasikan buku dengan benar tanpa kesalahan.

Namun, meskipun hasilnya serupa dalam hal accuracy, Random Forest dapat dianggap lebih optimal dalam beberapa konteks karena keunggulannya dalam menangani data yang lebih kompleks dan non-linear. Sebagai model ensemble yang menggabungkan banyak pohon keputusan, Random Forest lebih robust dan mampu mengidentifikasi pola yang lebih rumit atau hubungan non-linear antara fitur-fitur seperti harga dan rating buku.

##Skenario 3: Segmentasi Buku berdasarkan Harga dan Rating (Clustering)
- Algoritma 1: K-Means Clustering
- Algoritma 2: DBSCAN
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Mengambil fitur untuk clustering
X_clust = cleaned_books_data[['Price', 'Rating']]

# Model K-Means
model_kmeans = KMeans(n_clusters=3, random_state=42)
model_kmeans.fit(X_clust)
labels_kmeans = model_kmeans.labels_

# Evaluasi K-Means dengan Silhouette Score
sil_score_kmeans = silhouette_score(X_clust, labels_kmeans)
print(f"Silhouette Score (K-Means): {sil_score_kmeans}")

# Model DBSCAN
model_dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = model_dbscan.fit_predict(X_clust)

# Evaluasi DBSCAN dengan Silhouette Score
sil_score_dbscan = silhouette_score(X_clust, labels_dbscan)
print(f"Silhouette Score (DBSCAN): {sil_score_dbscan}")

"""Visualisasi Perbandingan Model K-Means Clustering dan DBSCAN"""

# 1. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_clust)

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_clust)

# Visualisasi K-Means
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_clust['Price'], X_clust['Rating'], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Price')
plt.ylabel('Rating')

# Visualisasi DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(X_clust['Price'], X_clust['Rating'], c=dbscan_labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.xlabel('Price')
plt.ylabel('Rating')

plt.tight_layout()
plt.show()

# Evaluasi dengan Silhouette Score
sil_score_kmeans = silhouette_score(X_clust, kmeans_labels)
sil_score_dbscan = silhouette_score(X_clust, dbscan_labels)

print(f"Silhouette Score (K-Means): {sil_score_kmeans}")
print(f"Silhouette Score (DBSCAN): {sil_score_dbscan}")

"""Hasil menunjukkan perbandingan antara dua algoritma clustering, yaitu K-Means dan DBSCAN, dalam mengelompokkan data berdasarkan harga dan rating buku. Berdasarkan hasil visualisasi dan evaluasi menggunakan Silhouette Score, dapat dilihat bahwa K-Means memiliki Silhouette Score yang lebih tinggi, berarti model ini lebih efektif dalam membentuk klaster yang baik dengan lebih sedikit noise. Namun, DBSCAN memiliki keunggulan dalam menangani data dengan distribusi yang tidak teratur, serta mampu mengidentifikasi titik-titik noise yang tidak termasuk dalam klaster mana pun. Meskipun kedua algoritma memiliki kinerja yang baik, K-Means lebih optimal

##Skenario 4: Prediksi Harga Buku dengan KNN (K-Nearest Neighbors)
- Algoritma 1: K-Nearest Neighbors (KNN)
- Algoritma 2: Support Vector Machine (SVM)
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Mengambil fitur dan target
X_knn = cleaned_books_data[['Rating']]  # Fitur: Rating
y_knn = cleaned_books_data['Price']    # Target: Harga

# Membagi data menjadi data latih dan data uji
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

# Model K-Nearest Neighbors (KNN)
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train_knn, y_train_knn)
y_pred_knn = model_knn.predict(X_test_knn)

# Evaluasi model KNN
mse_knn = mean_squared_error(y_test_knn, y_pred_knn)
r2_knn = r2_score(y_test_knn, y_pred_knn)
print(f"Mean Squared Error (KNN): {mse_knn}")
print(f"R-squared (KNN): {r2_knn}")

# Model Support Vector Machine (SVM) for regression
model_svm = SVR(kernel='linear')
model_svm.fit(X_train_knn, y_train_knn)
y_pred_svm = model_svm.predict(X_test_knn)

# Evaluasi model SVM
mse_svm = mean_squared_error(y_test_knn, y_pred_svm)
r2_svm = r2_score(y_test_knn, y_pred_svm)
print(f"Mean Squared Error (SVM): {mse_svm}")
print(f"R-squared (SVM): {r2_svm}")

"""Visualisasi Perbandingan Model K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM)"""

import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi perbandingan MSE dan R2 Score
plt.figure(figsize=(12, 6))

# Visualisasi MSE
plt.subplot(1, 2, 1)
bar_width = 0.35
index = range(2)
mse_values = [mse_knn, mse_svm]
plt.bar(index, mse_values, bar_width, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.xticks(index, ['KNN', 'SVM'])
plt.title('Perbandingan Mean Squared Error')

# Visualisasi R2 Score
plt.subplot(1, 2, 2)
r2_values = [r2_knn, r2_svm]
plt.bar(index, r2_values, bar_width, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.xticks(index, ['KNN', 'SVM'])
plt.title('Perbandingan R2 Score')

plt.tight_layout()
plt.show()

"""Hasil menunjukkan perbandingan antara model K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) dalam memprediksi harga buku berdasarkan rating. Berdasarkan evaluasi yang menggunakan metrik Mean Squared Error (MSE) dan R² Score, dapat dilihat bahwa meskipun kedua model menunjukkan kinerja yang baik, KNN lebih unggul dalam hal R² Score, yang menunjukkan bahwa model ini memiliki kemampuan yang lebih baik dalam menjelaskan variansi data harga. Sementara itu, SVM juga memberikan hasil yang kompetitif, namun dengan MSE yang sedikit lebih tinggi, yang menunjukkan bahwa model ini mungkin kurang akurat dalam prediksi harga dibandingkan dengan KNN."""

# Fungsi Clustering
def perform_clustering(df):
    st.subheader("Clustering Buku Berdasarkan Harga dan Rating")
    st.write("Clustering ini membagi buku ke dalam 3 kelompok berdasarkan harga dan rating.")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Price', 'Rating']])
    fig, ax = plt.subplots()
    sns.scatterplot(x='Price', y='Rating', hue='Cluster', palette='viridis', data=df, ax=ax)
    st.pyplot(fig)

# Fungsi Regresi
def perform_regression(df):
    st.subheader("Regresi untuk Prediksi Harga Berdasarkan Rating")
    st.write("Model regresi digunakan untuk memprediksi harga buku berdasarkan rating.")
    X = df[['Rating']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    st.pyplot(fig)

"""# Streamlit"""

# Main Program
def main():
    st.title("Final Project Data Mining Kelompok 8")
    st.write("Scraping dan pengolahan data pada Studi Kasus website bookstoscrape.com")

    # Sidebar Menu
    st.sidebar.header("Navigasi")
    options = st.sidebar.radio("Pilih Langkah:", ["Scrape Data", "Visualisasi", "Algoritma 1: Clustering", "Algoritma 2: Regresi", "Kesimpulan"])

    if options == "Scrape Data":
        st.header("Scraping Data")
        df = scrape_books()
        st.dataframe(df.head())
        st.write("Contoh tampilan data yang didapat setelah melakukan scraping")

    elif options == "Visualisasi":
        st.header("Visualisasi Data")
        df = scrape_books()
        df = clean_and_preprocess_data(df)

    elif options == "Algoritma 1: Clustering":
        st.header("Algoritma 1: K-Means Clustering")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        perform_clustering(df)

    elif options == "Algoritma 2: Regresi":
        st.header("Algoritma 2: Regresi Linear")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        perform_regression(df)

    elif options == "Kesimpulan":
        st.header("Kesimpulan")
        st.write("Dataset buku menunjukkan distribusi harga dan rating yang dapat dikategorikan ke dalam klaster menggunakan K-Means.")
        st.write("Model regresi menunjukkan hubungan yang lemah antara rating dan harga buku, dengan R-squared rendah.")
