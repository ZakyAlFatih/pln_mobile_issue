import streamlit as st
import joblib

# Load pipeline dan LabelEncoder
pipeline, le = joblib.load("svm_pipeline.pkl")

# Title
st.title("Klasifikasi Ulasan Pelanggan (SVM + TF-IDF)")

# Input teks ulasan
user_input = st.text_area("Masukkan Ulasan Pelanggan:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks ulasan terlebih dahulu.")
    else:
        # Prediksi
        pred_idx = pipeline.predict([user_input])[0]
        label = le.inverse_transform([pred_idx])[0]

        # Decision score (opsional)
        if hasattr(pipeline.named_steps['svm'], "decision_function"):
            decision_scores = pipeline.named_steps['svm'].decision_function(
                pipeline.named_steps['tfidf'].transform([user_input])
            )[0]
        else:
            decision_scores = None

        # Output
        st.success(f"Prediksi Label: {label}")
        
        if decision_scores is not None:
            st.subheader("Skor Keputusan:")
            for lbl, score in zip(le.classes_, decision_scores):
                st.write(f"{lbl}: {score:.4f}")
