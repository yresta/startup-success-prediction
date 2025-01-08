import streamlit as st
from joblib import load
import numpy as np
import random

# Kelas DecisionTree (untuk Decision Tree klasik)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Jika hanya ada satu kelas atau mencapai kedalaman maksimum, buat simpul daun
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return unique_classes[0]

        # Pemilihan fitur terbaik untuk pemisahan
        best_split = self._best_split(X, y, n_features)
        left_tree = self._build_tree(*best_split['left'], depth + 1)
        right_tree = self._build_tree(*best_split['right'], depth + 1)

        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y, n_features):
        best_split = {'gini': float('inf')}
        best_left = best_right = None
        best_feature = best_value = None

        features = random.sample(range(n_features), n_features)  # Pilih fitur secara acak
        for feature in features:
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]

                # Hitung impuritas Gini
                gini_left = self._gini_impurity(left_y)
                gini_right = self._gini_impurity(right_y)
                gini = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)

                if gini < best_split['gini']:
                    best_split['gini'] = gini
                    best_left = (X[left_mask], left_y)
                    best_right = (X[right_mask], right_y)
                    best_feature = feature
                    best_value = value

        return {'gini': best_split['gini'], 'left': best_left, 'right': best_right, 'feature': best_feature, 'value': best_value}

    def _gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def predict(self, X):
        return [self._predict_row(row, self.tree) for row in X]

    def _predict_row(self, row, tree):
        if isinstance(tree, dict):
            if row[tree['feature']] <= tree['value']:
                return self._predict_row(row, tree['left'])
            else:
                return self._predict_row(row, tree['right'])
        return tree

# Kelas RandomForest
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrapping (ambil sampel acak dengan pengembalian)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # Prediksi dengan mayoritas suara dari semua pohon
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [self._majority_vote(tree_preds[:, i]) for i in range(len(X))]

    def _majority_vote(self, predictions):
        return np.bincount(predictions).argmax()

# Sidebar untuk navigasi halaman
st.sidebar.write("Pilih halaman untuk mengeksplorasi lebih lanjut tentang ThriveSight🚀")
page = st.sidebar.selectbox("Navigasi Halaman", ("Home Page", "Penjelasan Project", "Prediksi", "Evaluasi Model"))


if page == "Home Page":
    st.title("Selamat Datang di ThriveSight🚀")

    st.video("video.mp4")  

    st.write("Ingin mengetahui peluang keberhasilan startup Anda? **ThriveSight** membantu Anda memanfaatkan data penting untuk mendapatkan gambaran sederhana namun bermakna tentang potensi perjalanan bisnis Anda, apakah berada di jalur **sukses** atau mungkin menghadapi risiko tertentu.")
    
    st.markdown(""" 
    ✨ **Faktor yang Kami Pertimbangkan dalam Prediksi:**
    - **Usia & Pencapaian:** Usia startup & pencapaian yang telah diraih.
    - **Jejak Pendanaan:** Jumlah putaran pendanaan & partisipasi dalam aktivitas startup.
    - **Prestasi & Reputasi:** Apakah termasuk dalam Top 500?
    - **Relasi & Wilayah:** Koneksi bisnis dan lokasi operasional.
    """)

    st.write(" ")

    st.markdown("""
    🔮 **Hasil Prediksi:**  
    Dapatkan gambaran sederhana tentang potensi sukses atau tantangan yang mungkin dihadapi, disertai dengan insight berbasis data untuk membantu Anda membuat keputusan yang lebih terinformasi.
    """)

    st.write(" ")
    st.markdown("""
    💡 **Kenapa Memilih Kami?**
    - **Pendekatan Data:** Analisis berbasis informasi yang relevan.
    - **Mudah Dimengerti:** Hasil prediksi yang sederhana dan jelas.
    - **Membantu Rencana:** Memberi wawasan untuk langkah ke depan.
    """)

    st.write(" ")
    st.write("Ambil langkah untuk memahami potensi startup Anda bersama **ThriveSight** 🚀")



elif page == "Penjelasan Project":
    st.title("Penjelasan Project")

    st.write("""
    **ThriveSight** bertujuan untuk membantu pengusaha dan investor menilai potensi keberhasilan sebuah startup 
    berdasarkan data terkait seperti usia, pencapaian, pendanaan, hubungan bisnis, dan berbagai faktor lainnya. 
    Dengan menggunakan algoritma **Random Forest**, sistem ini memberikan prediksi yang berbasis data untuk mendukung pengambilan keputusan yang lebih terinformasi.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Latar Belakang", "Tujuan", "Manfaat", "Metode"])

    with tab1:
        st.write("### Latar Belakang")
        st.write("""
        Dalam dunia startup, pengambilan keputusan yang tepat berdasarkan data menjadi kunci untuk kesuksesan 🚀. 
        Namun, banyak pengusaha dan investor yang masih kesulitan dalam menilai potensi sebuah startup untuk sukses atau gagal. 
        Faktor-faktor seperti usia, pencapaian, pendanaan, dan hubungan bisnis dapat memberikan gambaran penting mengenai masa depan sebuah startup. 
        Oleh karena itu, dibutuhkan sebuah sistem yang dapat memprediksi kemungkinan keberhasilan atau kegagalan startup dengan menganalisis data yang tersedia.
        """)

    with tab2:
        st.write("### Tujuan")
        st.write("""
        Proyek ini bertujuan untuk mengembangkan sistem prediksi yang dapat memprediksi kemungkinan keberhasilan atau kegagalan sebuah startup menggunakan algoritma Random Forest 🌲. 
        Dengan memanfaatkan dataset yang berisi data-data terkait usia startup, jumlah pendanaan, pencapaian, dan faktor lainnya, sistem ini akan memberikan prediksi tentang potensi kesuksesan startup.
        """)

    with tab3:
        st.write("### Manfaat")
        st.write("""
        - **Bagi Pengusaha:** Memberikan wawasan yang lebih terperinci berdasarkan data untuk merencanakan strategi bisnis dan mengurangi risiko kegagalan 💼.
        - **Bagi Investor:** Menyediakan alat untuk menilai risiko dan potensi investasi yang lebih objektif berdasarkan analisis data yang komprehensif 💰.
        - **Bagi Ekosistem Startup:** Membantu menciptakan ekosistem yang lebih terinformasi dengan pendekatan berbasis data, meningkatkan peluang kesuksesan startup 🌱.
        """)

    with tab4:
        st.write("### Metode")
        st.write("""
        Untuk proyek ini, kami menggunakan algoritma Random Forest 🌳 dengan dataset berisi informasi terkait startup.
        Langkah-langkah Metode:
        1. **Pengumpulan Data (Dataset):** Mengumpulkan dataset yang berisi informasi terkait startup, seperti usia, pendanaan, pencapaian, dan faktor-faktor lainnya 📋.
        2. **Preprocessing Data:** Membersihkan dan mempersiapkan data untuk diproses oleh model, termasuk menangani nilai yang hilang 🔧.
        3. **Pelatihan Model:** Menggunakan algoritma Random Forest untuk melatih model berdasarkan dataset yang sudah diproses, dengan tujuan memprediksi apakah startup akan sukses atau gagal 🎯.
        4. **Evaluasi Model:** Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score untuk menilai kinerja model 🧮.
        5. **Prediksi:** Menggunakan model yang telah dilatih untuk memprediksi kemungkinan sukses atau gagal sebuah startup berdasarkan input data baru yang diberikan oleh pengguna 💡.
        """)

        st.write("""
        Dengan menggunakan Algoritma Random Forest, sistem ini akan memberikan hasil prediksi yang dapat membantu pengusaha dan investor dalam membuat keputusan mengenai startup mereka yang lebih terinformasi 📈.
        """)

elif page == "Evaluasi Model":
    st.title("Evaluasi Model")

    st.write("""
        Model **Random Forest** pada project kami berhasil mencapai akurasi sebesar **0.98** dalam mengklasifikasikan keberhasilan startup. 
        Hasil ini menunjukkan bahwa model memiliki kemampuan yang baik dalam memprediksi potensi sukses ✅ atau gagal ❌ sebuah startup berdasarkan data yang diberikan. 
    """)

    st.image("gambar.png", caption="Performa Model 📊")

    st.write("""
        Dengan tingkat akurasi tersebut, diharapkan model ini dapat memberikan gambaran mengenai peluang keberhasilan sebuah startup 🌱, 
        serta membantu pengusaha dan investor dalam membuat keputusan yang lebih terinformasi 💡 dan strategis 📈. 
    """)



elif page == "Prediksi":
    st.title("Halaman Prediksi")

    st.write("""
        Di sini, Anda dapat mengukur potensi keberhasilan startup Anda menggunakan **ThriveSight**! 🌟  
        Cukup masukkan beberapa data terkait startup Anda, seperti usia, pencapaian, pendanaan, dan faktor penting lainnya, 
        dan model kami akan memberikan prediksi apakah startup Anda berpotensi sukses 💼🚀 atau menghadapi risiko gagal ⚠️.
        
        **Langkah-langkah Prediksi:**
        1. Masukkan data startup Anda di form yang disediakan 📋.
        2. Klik tombol **Prediksi** untuk menghitung kemungkinan keberhasilan startup Anda 🎯.
        3. Dapatkan hasil prediksi berupa status "Sukses" ✅ atau "Gagal" ❌ berdasarkan data yang Anda masukkan.
        
        **Ayo prediksi sekarang!** 🚀 Gunakan data untuk memahami potensi kesuksesan startup Anda dan buat keputusan yang lebih terinformasi untuk masa depan yang lebih cerah! 💡
    """)

    # Input fitur dari pengguna dalam dua kolom
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia", min_value=0, help="Usia startup dalam tahun.")
        relationships = st.number_input("Hubungan Kerjasama", min_value=0, help="Jumlah hubungan atau koneksi yang dimiliki startup.")
        avg_participants = st.number_input("Rata-rata Partisipan", min_value=0, help="Rata-rata jumlah partisipan dalam aktivitas startup.")
        funding_rounds = st.number_input("Putaran Pendanaann", min_value=0, help="Jumlah total putaran pendanaan yang telah dilakukan startup.")        
        milestones = st.number_input("Jumlah Pencapaian", min_value=0, help="Jumlah total pencapaian yang telah diraih startup.")

    with col2:
        age_first_milestone_year = st.number_input("Usia Saat Pencapaian Pertama", min_value=0, help="Usia startup saat mencapai pencapaian pertama.")
        age_last_milestone_year = st.number_input("Usia Saat Pencapaian Terakhir", min_value=0, help="Usia startup saat mencapai pencapaian terakhir.")        
        has_RoundABCD = st.radio("Memiliki Pendanaan?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", help="Apakah startup telah memperoleh pendanaan di Round A, B, C, atau D?")                
        is_otherstate = st.radio("Beroperasi di Other State?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", help="Apakah startup beroperasi di negara bagian selain negara bagian utama (California, New York, Massachusetts, Texas)?")
        is_top500 = st.radio("Termasuk dalam Top 500?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak", help="Apakah startup termasuk dalam Top 500?")

    # Membuat array fitur
    selected_features = np.array([age, relationships, age_last_milestone_year, milestones, is_top500,
                                  has_RoundABCD, age_first_milestone_year, funding_rounds, avg_participants, is_otherstate]).reshape(1, -1)

    # Tambahkan tombol prediksi
    predict_button = st.button("Prediksi")

    # Prediksi berdasarkan algoritma yang dipilih hanya jika tombol ditekan
    if predict_button:
        model = load("random_forest_model.joblib")
        if model:
            # Menggunakan .predict() untuk model Random Forest
            prediction = model.predict(selected_features)

            if prediction[0] == 1:
                st.markdown("""
                    <div style="background-color: rgba(0, 255, 0, 0.2); padding: 10px; border: 2px solid green; border-radius: 5px; color: green; font-size: 18px; text-align: center;">
                        <strong>Prediksi: Sukses</strong>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: rgba(255, 0, 0, 0.2); padding: 10px; border: 2px solid red; border-radius: 5px; color: red; font-size: 18px; text-align: center;">
                        <strong>Prediksi: Gagal</strong>
                    </div>
                """, unsafe_allow_html=True)

