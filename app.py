import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# --- Configuración Inicial ---

try:
    stopwords.words('english')
except LookupError:
    
    try:
        nltk.download('stopwords', quiet=True) 
    except Exception as e:
        st.error(f"Error descargando stopwords de NLTK: {e}")
        


# --- Cargar Modelo y Vectorizador ---

model_path = 'logistic_regression_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

# Usar un bloque try-except para manejar errores si los archivos no se encuentran
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error(f"Error: No se encontraron los archivos del modelo ('{model_path}') o vectorizador ('{vectorizer_path}'). Asegúrate de que estén en el directorio correcto.")
    # Detener la ejecución si los archivos no se cargan
    st.stop()
except Exception as e:
    st.error(f"Error al cargar modelo o vectorizador: {e}")
    st.stop()



stop_words_en = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text) 
    text = text.lower() 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#','', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    # Quitar stopwords
    text_tokens = text.split()
    filtered_text = [word for word in text_tokens if word not in stop_words_en]
    return " ".join(filtered_text)


# --- Interfaz de Usuario de Streamlit ---
st.title("🤖 Predicción de Viralidad de Tuits 🐦")
st.write("Escribe el texto de un tuit y te diremos si es probable que sea 'viral' (según nuestro modelo basado en >25k likes o >13k shares).")

# Área de texto para que el usuario escriba el tuit
user_text = st.text_area("Texto del Tuit:", height=100, placeholder="Escribe tu tuit aquí...")


if st.button("¡Predecir Viralidad!"):
    if user_text:
        # 1. Preprocesar el texto del usuario
        cleaned_text = preprocess_text(user_text)
        st.write(f"Texto preprocesado: '{cleaned_text}'") 

        # 2. Vectorizar el texto preprocesado
        
        text_vector = vectorizer.transform([cleaned_text])

        # 3. Hacer la predicción con el modelo cargado
        prediction = model.predict(text_vector)
        

        # 4. Mostrar el resultado
        st.subheader("Resultado de la Predicción:")
        if prediction[0] == 1:
            st.success("🚀 ¡Este tuit tiene potencial para ser VIRAL! 👍")
            
        else:
            st.info("📉 Este tuit parece que tendrá una interacción normal.")
            
    else:
        st.warning("⚠️ Por favor, escribe el texto de un tuit para predecir.")

st.sidebar.info("Modelo: Regresión Logística\nDefinición de Viral: >25k Likes o >13k Shares")