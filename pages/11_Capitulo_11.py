import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import cv2
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
import threading

# --- Configuración de la Página ---
st.set_page_config(page_title="Capítulo 11", page_icon="⏸️", layout="wide")

# --- Funciones de Configuración WebRTC ---
def get_ice_servers():
    """Obtiene credenciales TURN de Twilio"""
    try:
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
        response = requests.post(url, auth=(account_sid, auth_token))
        
        if response.status_code == 201:
            token_data = response.json()
            return token_data['ice_servers']
        else:
            st.warning("No se pudieron obtener servidores TURN, usando STUN público")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
    except Exception as e:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- Clase para Extracción de Características ---
class FeatureExtractor:
    """Extrae características usando SIFT y Bag of Words"""
    
    def __init__(self):
        # Usar SIFT con más características para mejor detección
        self.sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.03)
    
    def extract_sift_features(self, img):
        """Extrae descriptores SIFT de una imagen con preprocesamiento"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Mejorar contraste
        gray = cv2.equalizeHist(gray)
        
        # Detectar keypoints y descriptores
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return descriptors
    
    def create_bow_codebook(self, images, n_clusters=50):
        """Crea un codebook usando K-means sobre todos los descriptores"""
        all_descriptors = []
        
        for img in images:
            descriptors = self.extract_sift_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            return None
        
        all_descriptors = np.vstack(all_descriptors)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        kmeans.fit(all_descriptors)
        
        return kmeans
    
    def get_feature_vector(self, img, kmeans):
        """Obtiene el vector de características usando BOW"""
        descriptors = self.extract_sift_features(img)
        
        if descriptors is None:
            return np.zeros(kmeans.n_clusters)
        
        labels = kmeans.predict(descriptors)
        histogram = np.zeros(kmeans.n_clusters)
        for label in labels:
            histogram[label] += 1
        
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        return histogram


# --- Clase del Clasificador ANN ---
class ClassifierANN:
    """Clasificador usando Red Neuronal Artificial Multi-Capa Perceptrón"""
    
    def __init__(self, feature_vector_size, label_words):
        self.ann = cv2.ml.ANN_MLP_create()
        self.label_words = label_words
        
        input_size = feature_vector_size
        output_size = len(label_words)
        hidden_size = int((input_size * 2/3) + output_size)
        
        nn_config = np.array([input_size, hidden_size, output_size], dtype=np.uint8)
        self.ann.setLayerSizes(nn_config)
        self.ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
        self.ann.setTermCriteria(criteria)
        self.ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001)
        
        self.le = preprocessing.LabelBinarizer()
        self.le.fit(label_words)
    
    def train(self, training_data):
        """Entrena la red neuronal"""
        labels = [item['label'] for item in training_data]
        features = np.array([item['feature_vector'] for item in training_data], dtype=np.float32)
        encoded_labels = np.array(self.le.transform(labels), dtype=np.float32)
        self.ann.train(features, cv2.ml.ROW_SAMPLE, encoded_labels)
    
    def predict(self, feature_vector):
        """Predice la clase de un vector de características"""
        feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        retval, output = self.ann.predict(feature_vector)
        predicted_label = self.le.inverse_transform(output, threshold=0.5)
        return predicted_label[0] if len(predicted_label) > 0 else None
    
    def predict_proba(self, feature_vector):
        """Predice probabilidades para cada clase"""
        feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        retval, output = self.ann.predict(feature_vector)
        
        # Normalizar salidas a probabilidades
        probs = {}
        output = output[0]
        for i, label in enumerate(self.label_words):
            probs[label] = max(0, min(1, (output[i] + 1) / 2))  # De [-1,1] a [0,1]
        
        return probs
    
    def get_confusion_matrix(self, testing_data):
        """Calcula la matriz de confusión"""
        confusion_matrix = OrderedDict()
        for label in self.label_words:
            confusion_matrix[label] = OrderedDict()
            for label2 in self.label_words:
                confusion_matrix[label][label2] = 0
        
        for item in testing_data:
            true_label = item['label']
            predicted_label = self.predict(item['feature_vector'])
            if predicted_label:
                confusion_matrix[true_label][predicted_label] += 1
        
        return confusion_matrix


# --- Procesador de Video para Clasificación en Tiempo Real ---
class ANNVideoProcessor(VideoProcessorBase):
    """Procesa video en tiempo real con clasificación ANN mejorada"""
    
    def __init__(self):
        self.classifier = None
        self.feature_extractor = None
        self.kmeans = None
        self.frame_count = 0
        self.skip_frames = 8  # Más rápido: cada 8 frames
        self.last_prediction = "Esperando..."
        self.last_probabilities = {}
        self.prediction_history = []  # Historial para suavizado
        self.history_size = 5  # Últimas 5 predicciones
        self.lock = threading.Lock()
    
    def preprocess_frame(self, img):
        """Preprocesa el frame para mejor detección"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ecualizar histograma para mejor contraste
        gray = cv2.equalizeHist(gray)
        
        # Aplicar filtro bilateral para reducir ruido manteniendo bordes
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Aplicar threshold adaptativo para realzar formas
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Invertir si el fondo es oscuro
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Convertir de vuelta a BGR para SIFT
        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def smooth_predictions(self, current_probs):
        """Suaviza las predicciones usando historial"""
        self.prediction_history.append(current_probs)
        
        # Mantener solo las últimas N predicciones
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Promediar probabilidades
        if len(self.prediction_history) > 0:
            smoothed_probs = {}
            for label in current_probs.keys():
                probs = [p[label] for p in self.prediction_history if label in p]
                smoothed_probs[label] = np.mean(probs) if probs else 0.0
            
            return smoothed_probs
        
        return current_probs
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Verificar si el modelo está cargado
        if self.classifier is None or self.feature_extractor is None or self.kmeans is None:
            cv2.putText(img, "Modelo no cargado - Entrena primero", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Clasificar cada N frames
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            try:
                # Preprocesar frame
                processed = self.preprocess_frame(img)
                
                # Redimensionar a tamaño óptimo
                height, width = processed.shape[:2]
                target_size = 200  # Aumentado para mejor detección
                scale = target_size / max(height, width)
                new_w, new_h = int(width * scale), int(height * scale)
                img_small = cv2.resize(processed, (new_w, new_h))
                
                # Extraer características con el frame preprocesado
                feature_vector = self.feature_extractor.get_feature_vector(img_small, self.kmeans)
                
                # Predecir
                predicted_label = self.classifier.predict(feature_vector)
                probabilities = self.classifier.predict_proba(feature_vector)
                
                # Suavizar predicciones
                smoothed_probs = self.smooth_predictions(probabilities)
                
                # Encontrar la predicción con mayor probabilidad suavizada
                if smoothed_probs:
                    best_label = max(smoothed_probs, key=smoothed_probs.get)
                    best_prob = smoothed_probs[best_label]
                    
                    # Solo actualizar si la confianza es alta
                    if best_prob > 0.4:  # Threshold de confianza
                        with self.lock:
                            self.last_prediction = best_label
                            self.last_probabilities = smoothed_probs
                    else:
                        with self.lock:
                            self.last_prediction = "Baja confianza"
                            self.last_probabilities = smoothed_probs
                
            except Exception as e:
                with self.lock:
                    self.last_prediction = "Error de procesamiento"
        
        # Dibujar información en el frame
        with self.lock:
            # Fondo semitransparente para el panel
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (450, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Título
            cv2.putText(img, "Clasificacion ANN - Mejorado", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Predicción actual con icono
            icons = {'circulo': 'O', 'cuadrado': '[]', 'triangulo': '^'}
            icon = icons.get(self.last_prediction, '?')
            
            # Color según confianza
            if self.last_probabilities and self.last_prediction in self.last_probabilities:
                confidence = self.last_probabilities[self.last_prediction]
                if confidence > 0.7:
                    color = (0, 255, 0)  # Verde: alta confianza
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Amarillo: media confianza
                else:
                    color = (0, 165, 255)  # Naranja: baja confianza
            else:
                color = (200, 200, 200)
            
            cv2.putText(img, f"{icon} {self.last_prediction.upper()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Mostrar probabilidades suavizadas con barras
            y_pos = 95
            for label, prob in sorted(self.last_probabilities.items(), 
                                     key=lambda x: x[1], reverse=True):
                # Texto de probabilidad
                text = f"{label}: {prob*100:.1f}%"
                bar_color = (0, 255, 0) if prob > 0.6 else (100, 200, 200)
                cv2.putText(img, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Barra de probabilidad
                bar_width = int(prob * 200)
                cv2.rectangle(img, (180, y_pos - 12), (180 + bar_width, y_pos - 2), 
                            bar_color, -1)
                cv2.rectangle(img, (180, y_pos - 12), (380, y_pos - 2), 
                            (100, 100, 100), 1)
                
                y_pos += 25
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Funciones Auxiliares ---
def resize_image(img, max_size=300):
    """Redimensiona la imagen manteniendo el aspect ratio"""
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_w, new_h = int(width * scale), int(height * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img


def calculate_accuracy(confusion_matrix):
    """Calcula la precisión por clase"""
    acc_models = OrderedDict()
    
    for model in confusion_matrix.keys():
        acc_models[model] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    for expected_model, predicted_models in confusion_matrix.items():
        for predicted_model, value in predicted_models.items():
            if predicted_model == expected_model:
                acc_models[expected_model]['TP'] += value
                acc_models[predicted_model]['TN'] += value
            else:
                acc_models[expected_model]['FN'] += value
                acc_models[predicted_model]['FP'] += value
    
    accuracies = {}
    for model, rep in acc_models.items():
        total = rep['TP'] + rep['TN'] + rep['FN'] + rep['FP']
        if total > 0:
            acc = (rep['TP'] + rep['TN']) / total
            accuracies[model] = acc
        else:
            accuracies[model] = 0.0
    
    return accuracies


def rotate_image(img, angle):
    """Rota una imagen por un ángulo dado"""
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(255, 255, 255))
    return rotated


def add_noise(img, noise_level=10):
    """Añade ruido gaussiano a la imagen"""
    noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def get_default_images():
    """Genera imágenes de ejemplo con data augmentation para demostración"""
    images = {}
    
    # Parámetros para mayor variabilidad
    num_samples = 40  # Aumentado a 40 muestras
    img_size = 150  # Imágenes más grandes
    
    # CÍRCULOS - Con variaciones
    circles = []
    for i in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Parámetros aleatorios
        radius = np.random.randint(20, 60)
        center_x = np.random.randint(radius + 5, img_size - radius - 5)
        center_y = np.random.randint(radius + 5, img_size - radius - 5)
        thickness = np.random.choice([-1, -1, -1, 4, 5])  # Más rellenos
        
        # Colores más variados
        if np.random.random() > 0.5:
            color = tuple(np.random.randint(50, 256, 3).tolist())
        else:
            color = tuple(np.random.randint(0, 100, 3).tolist())
        
        cv2.circle(img, (center_x, center_y), radius, color, thickness)
        
        # Agregar círculos adicionales pequeños a veces
        if np.random.random() > 0.7:
            cv2.circle(img, (center_x, center_y), radius//2, 
                      tuple(np.random.randint(0, 256, 3).tolist()), 2)
        
        # Ruido más fuerte
        if np.random.random() > 0.3:
            img = add_noise(img, noise_level=20)
        
        circles.append(img)
    
    images['circulo'] = circles
    
    # CUADRADOS - Con rotaciones y variaciones
    squares = []
    for i in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Parámetros aleatorios
        size = np.random.randint(40, 100)
        center_x = img_size // 2
        center_y = img_size // 2
        
        # Dibujar cuadrado centrado
        pt1 = (center_x - size//2, center_y - size//2)
        pt2 = (center_x + size//2, center_y + size//2)
        thickness = np.random.choice([-1, -1, -1, 4, 5])
        
        if np.random.random() > 0.5:
            color = tuple(np.random.randint(50, 256, 3).tolist())
        else:
            color = tuple(np.random.randint(0, 100, 3).tolist())
        
        cv2.rectangle(img, pt1, pt2, color, thickness)
        
        # Agregar cuadrado interno a veces
        if np.random.random() > 0.7:
            inner_size = size // 2
            pt1_inner = (center_x - inner_size//2, center_y - inner_size//2)
            pt2_inner = (center_x + inner_size//2, center_y + inner_size//2)
            cv2.rectangle(img, pt1_inner, pt2_inner, 
                         tuple(np.random.randint(0, 256, 3).tolist()), 3)
        
        # CRÍTICO: Rotar en múltiplos de 15° para mejor cobertura
        angle = np.random.choice([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 
                                 150, 165, 180, 195, 210, 225, 240, 255, 
                                 270, 285, 300, 315, 330, 345])
        img = rotate_image(img, angle)
        
        # Ruido
        if np.random.random() > 0.3:
            img = add_noise(img, noise_level=20)
        
        squares.append(img)
    
    images['cuadrado'] = squares
    
    # TRIÁNGULOS - Con rotaciones y variaciones
    triangles = []
    for i in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Parámetros aleatorios para el triángulo
        center_x = img_size // 2
        center_y = img_size // 2
        size = np.random.randint(40, 90)
        
        # Triángulo equilátero centrado
        height = int(size * np.sqrt(3) / 2)
        pts = np.array([
            [center_x, center_y - 2*height//3],
            [center_x - size//2, center_y + height//3],
            [center_x + size//2, center_y + height//3]
        ], np.int32)
        
        if np.random.random() > 0.5:
            color = tuple(np.random.randint(50, 256, 3).tolist())
        else:
            color = tuple(np.random.randint(0, 100, 3).tolist())
        
        # Más probabilidad de relleno
        if np.random.random() > 0.2:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, 4)
        
        # Agregar líneas internas a veces
        if np.random.random() > 0.7:
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), 
                    tuple(np.random.randint(0, 256, 3).tolist()), 2)
        
        # CRÍTICO: Rotar en múltiplos de 15°
        angle = np.random.choice([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 
                                 150, 165, 180, 195, 210, 225, 240, 255, 
                                 270, 285, 300, 315, 330, 345])
        img = rotate_image(img, angle)
        
        # Ruido
        if np.random.random() > 0.3:
            img = add_noise(img, noise_level=20)
        
        triangles.append(img)
    
    images['triangulo'] = triangles
    
    return images


# --- Función Principal ---
def run_capitulo11():
    
    # ------------------ Encabezado ------------------
    st.title("CAPÍTULO 11: Aprendizaje automático mediante una red neuronal artificial")
    st.markdown("##### *Machine Learning (ML) by an Artificial Neural Network (ANN)*")
    st.markdown("---")
    
    # ------------------ Introducción ------------------
    st.header("📚 Introducción a las Redes Neuronales Artificiales")
    
    with st.expander("ℹ️ ¿Qué es una ANN-MLP?", expanded=False):
        st.markdown("""
        ### Red Neuronal Artificial Multi-Capa Perceptrón
        
        Una **ANN-MLP** es un tipo de red neuronal inspirada en el cerebro humano que aprende patrones complejos.
        
        #### 🏗️ Arquitectura:
        1. **Capa de Entrada**: Recibe características (vector BOW)
        2. **Capa Oculta**: Procesa y transforma datos
        3. **Capa de Salida**: Produce predicciones (probabilidades)
        
        #### 🎯 Ventajas sobre SVM:
        - ✅ **Multi-clase nativa**: Detecta múltiples objetos
        - ✅ **Probabilístico**: Retorna probabilidades
        - ✅ **Flexible**: Funciona con cualquier estructura
        - ✅ **Escalable**: Más capas = más complejidad
        
        #### 🔬 Proceso:
        1. Extracción de características (SIFT + BOW)
        2. Codificación de etiquetas
        3. Entrenamiento (Backpropagation)
        4. Evaluación (Matriz de confusión)
        """)
    
    st.info("📌 **Aprenderás:** Entrenar ANN, clasificar imágenes y webcam en tiempo real")
    
    # ------------------ Selector de Modo ------------------
    st.markdown("---")
    st.header("⚙️ Modo de Operación")
    
    mode = st.radio(
        "Selecciona cómo usar el clasificador:",
        ["🎓 Entrenar Modelo", "🔍 Clasificar Imagen", "📹 Clasificación en Vivo (Webcam)"],
        horizontal=True
    )
    
    # Inicializar estados
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = None
    if 'kmeans' not in st.session_state:
        st.session_state.kmeans = None
    if 'label_words' not in st.session_state:
        st.session_state.label_words = None
    
    # ------------------ MODO: ENTRENAMIENTO ------------------
    if mode == "🎓 Entrenar Modelo":
        st.markdown("---")
        st.header("📊 Entrenamiento del Modelo")
        
        st.info("""
        **Modo Demostración**: Entrenaremos con formas sintéticas ultra-mejoradas.
        
        🎯 **Mejoras V2 aplicadas:**
        - ✅ **40 muestras** por clase (120 total)
        - ✅ **Rotaciones discretas** cada 15° para mejor cobertura
        - ✅ **Imágenes 150x150px** (más grandes = mejor SIFT)
        - ✅ **Preprocesamiento mejorado**: Ecualización de histograma
        - ✅ **100 clusters BOW** (óptimo para estas figuras)
        - ✅ **Colores variados** (oscuros y claros)
        - ✅ **Formas anidadas** (círculos dentro de círculos, etc.)
        - ✅ **Ruido adaptativo** en 70% de las muestras
        - ✅ **Threshold de confianza** en video (filtro de predicciones débiles)
        - ✅ **Suavizado temporal** (promedia últimas 5 predicciones)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Clusters BOW", 50, 200, 100, 10,
                                   help="Más clusters = mejor detección de patrones (recomendado: 100)")
        
        with col2:
            train_ratio = st.slider("% Entrenamiento", 0.6, 0.9, 0.80, 0.05,
                                   help="Mayor ratio = más datos de entrenamiento")
        
        if st.button("🚀 Entrenar Red Neuronal", type="primary", use_container_width=True):
            
            with st.spinner("⏳ Generando imágenes..."):
                demo_images = get_default_images()
                st.session_state.label_words = list(demo_images.keys())
            
            st.success(f"✅ {sum(len(imgs) for imgs in demo_images.values())} imágenes generadas (con rotaciones y variaciones)")
            
            # Mostrar ejemplos variados
            st.markdown("**📷 Ejemplos de Imágenes Generadas (con data augmentation):**")
            st.caption("⚡ Cada clase incluye: rotaciones 0-360°, diferentes tamaños, posiciones y colores variados")
            
            cols = st.columns(len(demo_images))
            for idx, (label, images) in enumerate(demo_images.items()):
                with cols[idx]:
                    # Mostrar 3 ejemplos por clase para ver la variabilidad
                    st.markdown(f"**{label.upper()}**")
                    for i in range(min(3, len(images))):
                        st.image(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), 
                                use_container_width=True)
                        if i < 2:
                            st.markdown("")  # Espacio
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Extrayendo SIFT...")
            progress_bar.progress(0.25)
            
            feature_extractor = FeatureExtractor()
            all_images = []
            for images in demo_images.values():
                all_images.extend(images)
            
            status_text.text("📚 Creando BOW...")
            progress_bar.progress(0.5)
            
            kmeans = feature_extractor.create_bow_codebook(all_images, n_clusters=n_clusters)
            
            if kmeans is None:
                st.error("❌ Error al crear codebook")
                return
            
            status_text.text("🔢 Generando vectores...")
            progress_bar.progress(0.75)
            
            training_data = []
            for label, images in demo_images.items():
                for img in images:
                    feature_vector = feature_extractor.get_feature_vector(img, kmeans)
                    training_data.append({'label': label, 'feature_vector': feature_vector})
            
            np.random.shuffle(training_data)
            split_idx = int(len(training_data) * train_ratio)
            train_set = training_data[:split_idx]
            test_set = training_data[split_idx:]
            
            status_text.text("🧠 Entrenando ANN...")
            progress_bar.progress(1.0)
            
            classifier = ClassifierANN(n_clusters, st.session_state.label_words)
            classifier.train(train_set)
            
            st.session_state.trained_model = classifier
            st.session_state.feature_extractor = feature_extractor
            st.session_state.kmeans = kmeans
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ ¡Red entrenada!")
            
            # Evaluación
            st.markdown("---")
            st.header("📈 Evaluación")
            
            confusion_matrix = classifier.get_confusion_matrix(test_set)
            accuracies = calculate_accuracy(confusion_matrix)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**🎯 Matriz de Confusión:**")
                import pandas as pd
                df = pd.DataFrame(confusion_matrix).T
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Precisión:**")
                for label, acc in accuracies.items():
                    st.metric(label.capitalize(), f"{acc*100:.1f}%")
                
                avg_acc = np.mean(list(accuracies.values()))
                st.metric("**Promedio**", f"{avg_acc*100:.1f}%")
            
            # Arquitectura
            st.markdown("---")
            st.markdown("**🏗️ Arquitectura:**")
            arch_col1, arch_col2, arch_col3 = st.columns(3)
            
            with arch_col1:
                st.info(f"**Entrada**\n\n{n_clusters} neuronas")
            with arch_col2:
                hidden = int((n_clusters * 2/3) + len(st.session_state.label_words))
                st.info(f"**Oculta**\n\n{hidden} neuronas")
            with arch_col3:
                st.info(f"**Salida**\n\n{len(st.session_state.label_words)} neuronas")
    
    # ------------------ MODO: CLASIFICACIÓN ------------------
    elif mode == "🔍 Clasificar Imagen":
        st.markdown("---")
        st.header("🖼️ Clasificación de Imágenes")
        
        if st.session_state.trained_model is None:
            st.warning("⚠️ Primero entrena el modelo")
            return
        
        st.success(f"✅ Detecta: **{', '.join(st.session_state.label_words)}**")
        
        input_option = st.radio(
            "Fuente:",
            ["📤 Subir Imagen", "🎨 Generar Forma"],
            horizontal=True
        )
        
        img_input = None
        
        if input_option == "📤 Subir Imagen":
            uploaded_file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        else:
            shape_choice = st.selectbox("Forma:", ["circulo", "cuadrado", "triangulo"])
            
            if st.button("Generar", use_container_width=True):
                img_input = np.ones((100, 100, 3), dtype=np.uint8) * 255
                
                if shape_choice == "circulo":
                    cv2.circle(img_input, (50, 50), 35, (0, 0, 255), -1)
                elif shape_choice == "cuadrado":
                    cv2.rectangle(img_input, (15, 15), (85, 85), (0, 255, 0), -1)
                else:
                    pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
                    cv2.fillPoly(img_input, [pts], (255, 0, 0))
        
        if img_input is not None:
            st.markdown("---")
            col_img, col_result = st.columns([1, 1])
            
            with col_img:
                st.markdown("**📸 Imagen:**")
                st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col_result:
                st.markdown("**🎯 Resultado:**")
                
                with st.spinner("🔍 Analizando..."):
                    img_resized = resize_image(img_input)
                    feature_vector = st.session_state.feature_extractor.get_feature_vector(
                        img_resized, st.session_state.kmeans)
                    
                    predicted_label = st.session_state.trained_model.predict(feature_vector)
                    probabilities = st.session_state.trained_model.predict_proba(feature_vector)
                
                if predicted_label:
                    st.success(f"### 🏆 **{predicted_label.upper()}**")
                    
                    icons = {'circulo': '🔴', 'cuadrado': '🟩', 'triangulo': '🔺'}
                    st.markdown(f"# {icons.get(predicted_label, '❓')}")
                    
                    st.markdown("**Probabilidades:**")
                    for label, prob in probabilities.items():
                        st.progress(prob, text=f"{label}: {prob*100:.1f}%")
                else:
                    st.error("❌ No se pudo clasificar")
    
    # ------------------ MODO: WEBCAM EN VIVO ------------------
    elif mode == "📹 Clasificación en Vivo (Webcam)":
        st.markdown("---")
        st.header("📹 Clasificación en Tiempo Real")
        
        if st.session_state.trained_model is None:
            st.warning("⚠️ Primero entrena el modelo")
            return
        
        st.success(f"✅ Modelo cargado: **{', '.join(st.session_state.label_words)}**")
        
        st.info("""
        📌 **Instrucciones Mejoradas:**
        1. Presiona **START** para activar la cámara
        2. Muestra objetos con formas geométricas CLARAS
        3. La clasificación incluye **suavizado temporal** (promedia 5 frames)
        4. **Funciona con cualquier rotación** y posición
        
        💡 **Tips para MÁXIMA precisión:**
        - 📄 **Dibuja formas grandes en papel blanco** con marcador negro
        - 🔦 **Buena iluminación** (evita sombras fuertes)
        - 📏 **Formas simples y claras** (sin adornos)
        - 🎯 **Centra la forma** en el frame
        - 🖼️ **Fondo uniforme** (preferiblemente claro)
        - 📐 **Formas de ~10-20cm** funcionan mejor
        - ⚡ La barra verde indica **alta confianza** (>70%)
        
        🎨 **Ejemplos recomendados:**
        - ⭕ Círculo: Plato, tapa, CD, pelota
        - ⬜ Cuadrado: Libro, caja, post-it (puede estar rotado)
        - 🔺 Triángulo: Dibujado en papel o formar con manos
        """)
        
        # Configurar WebRTC
        ice_servers = get_ice_servers()
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})
        
        # Crear y configurar el procesador
        webrtc_ctx = webrtc_streamer(
            key="ann_classifier",
            video_processor_factory=ANNVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Cargar modelo en el procesador
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.classifier = st.session_state.trained_model
            webrtc_ctx.video_processor.feature_extractor = st.session_state.feature_extractor
            webrtc_ctx.video_processor.kmeans = st.session_state.kmeans
        
        # Panel de información
        st.markdown("---")
        st.markdown("### 💡 Tips para Mejores Resultados")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            st.markdown("**🔴 Círculos**")
            st.caption("Muestra objetos redondos: pelotas, platos, etc.")
        
        with col_t2:
            st.markdown("**🟩 Cuadrados**")
            st.caption("Muestra objetos cuadrados: libros, cajas, etc.")
        
        with col_t3:
            st.markdown("**🔺 Triángulos**")
            st.caption("Forma triángulos con las manos o papel")


run_capitulo11()