import streamlit as st
import joblib
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the Hugging Face model, tokenizer, and label encoder
@st.cache_resource 
def load_models():
    # Replace with your actual Hugging Face username and model name
    hf_model_path = "MightyHazEl/CyberBullyDetection" 
    
    # Load model and tokenizer directly from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
    
    # The label_encoder.pkl still loads from your local GitHub repository root
    label_encoder = joblib.load('label_encoder.pkl')
    
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_models()

# 2. Fungsi Pembersihan Teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# ... (The rest of your Streamlit UI code stays exactly the same) ...

# ==========================================
# USER INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")

# UI disesuaikan dengan fokus penelitian
st.title("🛡️ Analisis Dampak Cyberbullying dalam Diskusi Online")
st.write("Aplikasi ini menggunakan arsitektur Transformer untuk mendeteksi teks toksik dan mengkategorikannya guna menganalisis dampak cyberbullying.")

user_input = st.text_area("Masukkan teks di sini:", height=150)

if st.button("Analisis Teks"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner('Menganalisis konteks teks dengan Deep Learning...'):
            # A. Proses input
            cleaned_input = clean_text(user_input)
            
            # Tokenisasi input untuk model Transformer
            inputs = tokenizer(cleaned_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # B. Prediksi Model
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mendapatkan probabilitas dan indeks kelas
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            prediction_index = torch.argmax(logits, dim=1).item()
            confidence_score = probabilities[prediction_index].item() * 100
            
            # Decode indeks kembali ke label awal menggunakan label encoder
            prediction = label_encoder.inverse_transform([prediction_index])[0]
            
            # C. Tampilkan Hasil
            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            # Ubah kondisi if ini untuk menangkap label "bukan_cyberbullying"
            if "bukan" in prediction.lower() or "not" in prediction.lower():
                st.success(f"✅ **Aman!** Teks ini terdeteksi TIDAK mengandung unsur cyberbullying.")
                st.info(f"**Confidence Score:** {confidence_score:.2f}%")
            else:
                st.error(f"⚠️ **TERDETEKSI CYBERBULLYING!**")
                st.warning(f"**Kategori:** {prediction.replace('_', ' ').title()}")
                st.info(f"**Tingkat Keyakinan Model (Confidence Score):** {confidence_score:.2f}%")
                
                st.markdown("---")
                st.markdown("### Catatan Analisis Model:")
                st.write("Berbeda dengan pendekatan klasik, model ini mengukur tingkat toksisitas berdasarkan **konteks kalimat secara keseluruhan**, sehingga kata-kata netral yang dirangkai menjadi kalimat bermuatan negatif akan tetap terdeteksi.")
