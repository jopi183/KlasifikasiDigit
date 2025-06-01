import streamlit as st
import numpy as np
import joblib 
from PIL import Image
import cv2
import io
import warnings
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from scipy import ndimage
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

# --- Set Streamlit Page Configuration ---
st.set_page_config(
    page_title="Klasifikasi Angka Bahasa Isyarat",
    layout="wide",
    page_icon="🤟",
)

# --- Suppress Scikit-learn Version Mismatch Warnings ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Display Scikit-learn Version ---
st.write(f"Scikit-learn version used by app: {sklearn.__version__}")

# --- SignLanguagePreprocessor Class Definition (FIXED) ---
class SignLanguagePreprocessor:
    """Preprocessing class consistent with the training pipeline."""
    
    def __init__(self):
        pass 
    
    def simple_normalize(self, image):
        """Simple but effective normalization - EXACTLY from training code"""
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    def histogram_equalization(self, image):
        """Simple histogram equalization - EXACTLY from training code"""
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(image.shape).astype(np.uint8)

    def apply_gaussian_blur(self, image, sigma=1.0):
        """Apply Gaussian blur using scipy - EXACTLY from training code"""
        return ndimage.gaussian_filter(image, sigma=sigma)

    def preprocess_single_image(self, image):
        """Preprocess a single image - EXACTLY from training code"""
        original = image.copy()
        
        # 1. Simple normalization
        processed = self.simple_normalize(image)
        
        # 2. Histogram equalization for better contrast
        image_uint8 = (processed * 255).astype(np.uint8)
        image_eq = self.histogram_equalization(image_uint8)
        processed = self.simple_normalize(image_eq)
        
        # 3. Light Gaussian blur to reduce noise - FIXED: Use sigma=0.8 like training
        processed = self.apply_gaussian_blur(processed, sigma=0.8)
        
        return original, processed

    def preprocess_image_with_steps(self, image):
        """Applies preprocessing and stores intermediate steps for visualization."""
        # Convert to grayscale if the image is color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        steps = {}
        steps['original'] = image.copy()
        
        # Use the EXACT same preprocessing as training
        original, processed = self.preprocess_single_image(image)
        
        # Store intermediate steps for visualization
        img = image.astype(np.float32)
        
        # Step 1: Simple normalization
        processed_step1 = self.simple_normalize(image)
        processed_uint8_1 = (processed_step1 * 255).astype(np.uint8)
        steps['normalized'] = processed_uint8_1.copy()
        
        # Step 2: Histogram equalization
        image_eq = self.histogram_equalization(processed_uint8_1)
        steps['histogram_equalized'] = image_eq.copy()
        
        # Step 3: Second normalization
        processed_step2 = self.simple_normalize(image_eq)
        processed_uint8_2 = (processed_step2 * 255).astype(np.uint8)
        steps['renormalized'] = processed_uint8_2.copy()
        
        # Step 4: Gaussian blur - FINAL processed image
        final_processed = self.apply_gaussian_blur(processed_step2, sigma=0.8)
        final_uint8 = (final_processed * 255).astype(np.uint8)
        steps['final'] = final_uint8.copy()
        
        return processed, steps  # Return the actual processed float image

    def extract_pixel_features(self, images):
        """Extract raw pixel features - EXACTLY from training code"""
        if len(images.shape) == 2:
            images = images.reshape(1, *images.shape)
        n_samples = images.shape[0]
        flattened = images.reshape(n_samples, -1)
        return flattened

    def extract_statistical_features(self, images):
        """Extract statistical features - EXACTLY from training code"""
        if len(images.shape) == 2:
            images = images.reshape(1, *images.shape)
        
        features = []
        n_samples = images.shape[0]
        
        for i in range(n_samples):
            img = images[i]
            
            # Basic statistics
            feat = [
                np.mean(img), np.std(img), np.var(img),
                np.min(img), np.max(img), np.median(img),
                np.percentile(img, 25), np.percentile(img, 75),
            ]
            
            # Histogram features
            hist, _ = np.histogram(img, bins=16, range=(0, 1))
            hist = hist / np.sum(hist)
            feat.extend(hist)
            
            # Moments
            feat.extend([
                np.mean(img**2),  # Second moment
                np.mean((img - np.mean(img))**3) / (np.std(img)**3 + 1e-8),  # Skewness
                np.mean((img - np.mean(img))**4) / (np.std(img)**4 + 1e-8) - 3,  # Kurtosis
            ])
            
            features.append(feat)
        
        return np.array(features)

    def extract_gradient_features(self, images):
        """Extract gradient features - EXACTLY from training code"""
        if len(images.shape) == 2:
            images = images.reshape(1, *images.shape)
        
        features = []
        n_samples = images.shape[0]
        
        for i in range(n_samples):
            img = images[i]
            
            # Sobel gradients
            grad_x = ndimage.sobel(img, axis=1)
            grad_y = ndimage.sobel(img, axis=0)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            feat = [
                np.mean(magnitude), np.std(magnitude),
                np.max(magnitude),
                np.sum(magnitude > np.mean(magnitude)) / magnitude.size,
            ]
            
            features.append(feat)
        
        return np.array(features)

    def extract_texture_features(self, images):
        """Extract texture features using LBP - EXACTLY from training code"""
        if len(images.shape) == 2:
            images = images.reshape(1, *images.shape)
        
        features = []
        n_samples = images.shape[0]
        
        radius = 1
        n_points = 8 * radius
        
        for i in range(n_samples):
            img = images[i]
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Local Binary Pattern
            lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
            
            # LBP histogram
            n_bins = n_points + 2
            lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            
            features.append(lbp_hist)
        
        return np.array(features)

    def extract_comprehensive_features(self, images):
        """Extract all features and combine them - EXACTLY from training code"""
        pixel_feat = self.extract_pixel_features(images)
        stat_feat = self.extract_statistical_features(images)
        grad_feat = self.extract_gradient_features(images)
        texture_feat = self.extract_texture_features(images)
        
        st.write(f"Debug - Feature shapes:")
        st.write(f"  Pixel features: {pixel_feat.shape}")
        st.write(f"  Statistical features: {stat_feat.shape}")
        st.write(f"  Gradient features: {grad_feat.shape}")
        st.write(f"  Texture features: {texture_feat.shape}")
        
        # Combine all features
        all_features = np.hstack([pixel_feat, stat_feat, grad_feat, texture_feat])
        
        # Handle NaN values
        nan_mask = np.isnan(all_features)
        if np.any(nan_mask):
            st.warning(f"Found {np.sum(nan_mask)} NaN values, replacing with 0")
            all_features[nan_mask] = 0
        
        st.write(f"Combined features shape: {all_features.shape}")
        
        return all_features

    def extract_features(self, image):
        """Extract comprehensive features from a single image"""
        try:
            # Preprocess the image using EXACT same method as training
            processed_img, preprocessing_steps = self.preprocess_image_with_steps(image)
            
            # Extract comprehensive features from the processed image
            features = self.extract_comprehensive_features(processed_img)
            
            # Ensure features are properly shaped for single image
            if len(features.shape) > 1:
                features = features.flatten()
            
            return processed_img, features, preprocessing_steps
            
        except Exception as e:
            st.error(f"Error during feature extraction: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            # Provide fallback zero features
            features = np.zeros(4096 + 27 + 4 + 10)
            return processed_img if 'processed_img' in locals() else image, features, {}

# --- Cached Functions for Resource Loading ---
@st.cache_resource
def get_preprocessor():
    """Initializes and caches the SignLanguagePreprocessor object."""
    return SignLanguagePreprocessor()

@st.cache_resource
def load_model_components(path):
    """Loads the model components from the training script format."""
    try:
        model_data = joblib.load(path)
        
        # Extract components with validation
        required_keys = ['svm_model', 'scaler', 'pca', 'selector', 'accuracy']
        for key in required_keys:
            if key not in model_data:
                st.error(f"Missing required component: {key}")
                st.stop()
        
        svm_model = model_data['svm_model']
        scaler = model_data['scaler']
        pca = model_data['pca']
        selector = model_data['selector']
        accuracy = model_data['accuracy']
        model_info = model_data.get('model_info', {})
        
        st.success("✅ Model berhasil dimuat!")
        st.info(f"📊 Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Display additional model info
        if hasattr(svm_model, 'n_support_'):
            st.info(f"🔧 Support vectors: {np.sum(svm_model.n_support_)}")
        if hasattr(pca, 'explained_variance_ratio_'):
            st.info(f"📈 PCA variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        return svm_model, scaler, pca, selector, accuracy, model_info
        
    except FileNotFoundError:
        st.error(f"File model '{path}' tidak ditemukan!")
        st.info(f"Pastikan file '{path}' ada di direktori yang sama dengan app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.info("Pastikan Anda menggunakan model yang dilatih dengan script training yang benar.")
        st.stop()

def predict_with_model(svm_model, scaler, pca, selector, features):
    """Make prediction using the loaded model components - FIXED"""
    try:
        # Validate input features
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        st.write(f"Debug - Prediction pipeline:")
        st.write(f"  Input features shape: {features.shape}")
        st.write(f"  Input features range: [{np.min(features):.6f}, {np.max(features):.6f}]")
        
        # Apply PCA
        features_pca = pca.transform(features)
        st.write(f"  After PCA: {features_pca.shape}")
        st.write(f"  PCA features range: [{np.min(features_pca):.6f}, {np.max(features_pca):.6f}]")
        
        # Apply feature selection
        features_selected = selector.transform(features_pca)
        st.write(f"  After feature selection: {features_selected.shape}")
        st.write(f"  Selected features range: [{np.min(features_selected):.6f}, {np.max(features_selected):.6f}]")
        
        # Scale features
        features_scaled = scaler.transform(features_selected)
        st.write(f"  After scaling: {features_scaled.shape}")
        st.write(f"  Scaled features range: [{np.min(features_scaled):.6f}, {np.max(features_scaled):.6f}]")
        
        # Make prediction
        prediction = svm_model.predict(features_scaled)[0]
        
        # Get probabilities if available
        confidence = None
        probability = None
        
        if hasattr(svm_model, 'predict_proba'):
            try:
                proba = svm_model.predict_proba(features_scaled)[0]
                probability = np.max(proba)
                confidence = probability
                st.write(f"  Probabilities: {proba}")
                st.write(f"  Max probability: {probability}")
            except:
                st.write("  Probabilities not available")
        
        # Get decision function scores for additional confidence
        if hasattr(svm_model, 'decision_function'):
            try:
                decision_scores = svm_model.decision_function(features_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = decision_scores[0]
                st.write(f"  Decision scores: {decision_scores}")
                if confidence is None:
                    confidence = np.max(np.abs(decision_scores))
            except:
                st.write("  Decision function not available")
        
        st.write(f"  Final prediction: {prediction}")
        
        return prediction, confidence, probability
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

# --- Enhanced Visualization Function ---
def visualize_preprocessing_steps(preprocessing_steps):
    """Displays the various image preprocessing steps in Streamlit."""
    st.subheader("🔬 Tahapan Preprocessing Gambar")
    
    # Row 1 of images
    cols_row1 = st.columns(2)
    with cols_row1[0]:
        st.image(preprocessing_steps['original'], caption="1. Gambar Asli", use_column_width=True, clamp=True)
    with cols_row1[1]:
        st.image(preprocessing_steps['normalized'], caption="2. Normalisasi Awal", use_column_width=True, clamp=True)
    
    # Row 2 of images
    cols_row2 = st.columns(2)
    with cols_row2[0]:
        st.image(preprocessing_steps['histogram_equalized'], caption="3. Histogram Equalization", use_column_width=True, clamp=True)
    with cols_row2[1]:
        st.image(preprocessing_steps['final'], caption="4. Final (Gaussian Blur)", use_column_width=True, clamp=True)
    
    # Enhanced histogram comparison
    st.markdown("---")
    st.subheader("📊 Perbandingan Histogram")
    
    fig_hist, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Calculate statistics for each step
    steps_data = [
        ('Gambar Asli', preprocessing_steps['original'], 'blue'),
        ('Setelah Normalisasi', preprocessing_steps['normalized'], 'green'),
        ('Setelah Equalization', preprocessing_steps['histogram_equalized'], 'orange'),
        ('Gambar Akhir', preprocessing_steps['final'], 'red')
    ]
    
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (title, img, color) in enumerate(steps_data):
        axes[i].hist(img.flatten(), bins=50, alpha=0.7, color=color)
        axes[i].set_title(f'Histogram {title}')
        axes[i].set_xlabel('Intensitas Piksel')
        axes[i].set_ylabel('Frekuensi')
        
        # Add statistics text
        mean_val = np.mean(img)
        std_val = np.std(img)
        axes[i].text(0.02, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                     transform=axes[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

# --- Custom CSS Styling ---
st.markdown("""
<style>
.main-header {font-size: 2.5em; color: #4CAF50; text-align: center; font-weight: bold;}
.sub-header {font-size: 1.5em; color: #2196F3; text-align: center;}
.prediction-box {background-color: #e0f2f7; border-left: 5px solid #2196F3;
padding: 15px; border-radius: 8px; margin-top: 20px;}
.prediction-text {font-size: 2em; font-weight: bold; color: #333; text-align: center;}
.info-box {background-color: #f0f8ff; padding: 15px; border-radius: 8px;
border-left: 4px solid #2196F3; margin-bottom: 20px;}
.method-box {background-color: #fff3cd; padding: 10px; border-radius: 5px;
border-left: 3px solid #ffc107; margin: 10px 0;}
.stats-box {background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;}
.warning-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0;}
.success-box {background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# --- Main Streamlit Application Logic ---
def main():
    # Initialize the preprocessor
    preprocessor = get_preprocessor()

    # Application Title and Information
    st.markdown("<h1 class='main-header'>🤟 Klasifikasi Angka Bahasa Isyarat</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Identifikasi Angka (0–9) dari Gambar Tangan</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>🚀 Cara Penggunaan:</strong><br>
        1. Pilih sumber gambar (unggah dari perangkat atau ambil langsung dengan kamera web).<br>
        2. Pastikan gambar tangan Anda menunjukkan angka (0-9) dengan jelas.<br>
        3. Aplikasi akan secara otomatis menampilkan hasil klasifikasi dan tahapan pemrosesan gambar.<br><br>
        <strong>⚡ Teknologi:</strong> Model SVM (Support Vector Machine) dengan ekstraksi fitur komprehensif (Pixel + Statistical + Gradient + Texture) yang dikombinasikan dengan PCA dan feature selection, didukung oleh pipeline preprocessing yang terdiri dari normalisasi, histogram equalization, dan Gaussian blur.
    </div>
    """, unsafe_allow_html=True)

    # Load the model components
    model_filename = 'hand_sign_svm_model.pkl'  # Default filename from training script
    svm_model, scaler, pca, selector, accuracy, model_info = load_model_components(model_filename)

    # Display model information
    st.markdown("<div class='method-box'><strong>🔧 Informasi Model:</strong></div>", unsafe_allow_html=True)
    if model_info:
        st.info(f"📊 **Model Type**: {model_info.get('type', 'SVM with RBF kernel')}")
        st.info(f"🔧 **Preprocessing**: {model_info.get('preprocessing', 'Normalization + Histogram Equalization + Gaussian Blur')}")
        st.info(f"🎯 **Features**: {model_info.get('features', 'Pixel + Statistical + Gradient + Texture')}")
        st.info(f"📈 **Dimensionality Reduction**: {model_info.get('dimensionality_reduction', 'PCA + Feature Selection')}")

    # Quality recommendations
    st.markdown("""
    <div class='warning-box'>
        <strong>💡 Tips untuk Hasil Terbaik:</strong><br>
        • Gunakan latar belakang yang kontras dengan tangan<br>
        • Pastikan pencahayaan yang cukup dan merata<br>
        • Posisikan tangan di tengah frame<br>
        • Hindari bayangan yang mengaburkan bentuk tangan<br>
        • Pastikan gesture angka terlihat jelas dan tidak terpotong
    </div>
    """, unsafe_allow_html=True)

    # Image Source Selection: Upload or Camera Input
    choice = st.radio("📸 Pilih sumber gambar untuk klasifikasi:", ("Unggah Gambar", "Ambil Gambar dari Kamera"))

    image_data = None
    if choice == "Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah file gambar (JPG atau PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_data = Image.open(uploaded_file).convert("RGB")
    else:
        camera_image = st.camera_input("Ambil Gambar dari Kamera")
        if camera_image is not None:
            image_data = Image.open(io.BytesIO(camera_image.read())).convert("RGB")

    # Process and display results if an image has been provided
    if image_data is not None:
        image_np = np.array(image_data)
        
        # Resize image to 64x64, consistent with training data resolution
        image_resized = cv2.resize(image_np, (64, 64))
        
        st.markdown("---")
        st.subheader("📷 Gambar Input")
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            st.image(image_data, caption="Gambar Asli", use_column_width=True)
        with col_input2:
            st.image(image_resized, caption="Gambar Diresize (64x64 pixels)", use_column_width=True)

        # Perform preprocessing and feature extraction
        with st.spinner('⚙️ Memproses gambar dan mengekstraksi features...'):
            processed_img, features, preprocessing_steps = preprocessor.extract_features(image_resized)

        # Visualize preprocessing steps
        if preprocessing_steps:
            visualize_preprocessing_steps(preprocessing_steps)

        # Display processing information
        st.markdown("---")
        st.subheader("📊 Informasi Pemrosesan")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("Ukuran Gambar Akhir", f"{processed_img.shape[0]}×{processed_img.shape[1]}")
        with col4:
            st.metric("Jumlah Fitur", f"{features.shape[0]}")
        with col5:
            preprocessing_quality = processed_img.std() * 100 if hasattr(processed_img, 'std') else 0
            st.metric("Kualitas Pemrosesan (Std. Dev.)", f"{preprocessing_quality:.1f}%")
        with col6:
            original_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            processed_std = processed_img.std() if hasattr(processed_img, 'std') else 0
            original_std = original_gray.std() if hasattr(original_gray, 'std') else 1
            contrast_improvement = (processed_std - original_std) / (original_std + 1e-7) * 100
            st.metric("Peningkatan Kontras", f"{contrast_improvement:+.1f}%")

        # Feature analysis
        st.markdown("---")
        st.subheader("🔍 Analisis Fitur")
        
        # Check if we have enough features
        expected_features = 4096 + 27 + 4 + 10  # pixel + statistical + gradient + texture
        if len(features) < expected_features:
            st.error(f"❌ Jumlah fitur tidak sesuai! Expected: {expected_features}, Got: {len(features)}")
        else:
            col7, col8, col9, col10 = st.columns(4)
            
            with col7:
                pixel_features = features[:4096]
                st.metric("Pixel Features", f"Mean: {np.mean(pixel_features):.3f}")
            with col8:
                stat_features = features[4096:4096+27]
                st.metric("Statistical Features", f"Range: {np.ptp(stat_features):.3f}")
            with col9:
                grad_features = features[4096+27:4096+27+4]
                st.metric("Gradient Features", f"Max: {np.max(grad_features):.3f}")
            with col10:
                texture_features = features[4096+27+4:]
                st.metric("Texture Features", f"Sum: {np.sum(texture_features):.3f}")

        # Perform prediction and display results
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        prediction, confidence, probability = predict_with_model(svm_model, scaler, pca, selector, features)
        
        if prediction is not None:
            # Create confidence level
            if confidence is not None:
                if confidence > 0.8:
                    conf_level = "Sangat Tinggi"
                    conf_color = "#28a745"
                elif confidence > 0.6:
                    conf_level = "Tinggi"
                    conf_color = "#28a745"
                elif confidence > 0.4:
                    conf_level = "Sedang"
                    conf_color = "#ffc107"
                else:
                    conf_level = "Rendah"
                    conf_color = "#dc3545"
            else:
                conf_level = "Tidak tersedia"
                conf_color = "#6c757d"
            
            confidence_text = f"Skor Keyakinan: {confidence:.3f} ({conf_level})" if confidence is not None else "Confidence score tidak tersedia."
            probability_text = f"Probabilitas: {probability:.3f}" if probability is not None else ""
            
            st.markdown(f"""
            <div class='prediction-box'>
                <p class='prediction-text'>Prediksi Angka: <span style='color: #E91E63;'>{prediction}</span></p>
                <p style='text-align: center; color: {conf_color}; font-size: 1.2em; font-weight: bold;'>
                    {confidence_text}
                </p>
                {f"<p style='text-align: center; color: #666; font-size: 1.1em;'>{probability_text}</p>" if probability_text else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable section for detailed pipeline information
            with st.expander("🔍 Detail Pipeline Pemrosesan"):
                st.markdown("""
                **Tahapan Pipeline yang Diterapkan:**
                1. ✅ **Resizing Gambar:** Gambar diubah ukurannya menjadi 64x64 piksel untuk konsistensi.
                2. ✅ **Konversi Grayscale:** Gambar dikonversi ke skala abu-abu menggunakan OpenCV.
                3. ✅ **Normalisasi Awal:** Nilai piksel dinormalisasi ke rentang 0-1.
                4. ✅ **Histogram Equalization:** Kontras gambar ditingkatkan menggunakan histogram equalization.
                5. ✅ **Normalisasi Kedua:** Nilai piksel dinormalisasi kembali setelah histogram equalization.
                6. ✅ **Gaussian Blur:** Filter Gaussian (sigma=0.8) diterapkan untuk mengurangi noise.
                7. ✅ **Ekstraksi Fitur Komprehensif:** 
                   - Pixel features: Raw pixel values (4096 features)
                   - Statistical features: Mean, std,
                   - Gradient features: Sobel gradients magnitude statistics (4 features)
                   - Texture features: Local Binary Pattern (LBP) histogram (10 features)
                8. ✅ **PCA:** Dimensionality reduction dengan Principal Component Analysis.
                9. ✅ **Feature Selection:** Seleksi fitur terbaik menggunakan SelectKBest.
                10. ✅ **Scaling & Prediksi:** Fitur diskalakan dan digunakan untuk prediksi SVM.
                """)
                
                st.markdown("**Statistik Detail Gambar:**")
                if preprocessing_steps:
                    stats_data = {
                        "Gambar Asli (Setelah Resize)": preprocessing_steps.get('original', processed_img),
                        "Setelah Normalisasi": preprocessing_steps.get('normalized', processed_img),
                        "Setelah Histogram Equalization": preprocessing_steps.get('histogram_equalized', processed_img),
                        "Gambar Akhir (Setelah Gaussian Blur)": preprocessing_steps.get('final', processed_img)
                    }
                    
                    for name, img in stats_data.items():
                        st.markdown(f"""
                        <div class='stats-box'>
                        <strong>{name}:</strong> Min={img.min():.2f}, Max={img.max():.2f}, 
                        Mean={img.mean():.2f}, Std={img.std():.2f}
                        </div>
                        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()