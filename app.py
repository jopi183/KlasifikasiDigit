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
from scipy.ndimage import median_filter
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
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

# --- SignLanguageClassifier Class Definition ---
# This class wraps the trained SVM model and its scaler.
# It MUST be IDENTICAL to the class definition used in your training script
# when the 'sign_language_model.pkl' file was generated.
class SignLanguageClassifier:
    """Custom wrapper class for the sign language classifier."""
    
    def __init__(self, svm_model=None, scaler=None): 
        self.svm_model = svm_model if svm_model is not None else SVC(kernel='rbf', probability=True, random_state=42)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = True 

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Cannot make predictions.")
        X_scaled = self.scaler.transform(X)
        # Access the best estimator from GridSearchCV if svm_model is a GridSearchCV object
        if hasattr(self.svm_model, 'best_estimator_'):
            return self.svm_model.best_estimator_.predict(X_scaled)
        else:
            return self.svm_model.predict(X_scaled) # Fallback if not GridSearchCV
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Cannot predict probabilities.")
        X_scaled = self.scaler.transform(X)
        # Access the best estimator from GridSearchCV if svm_model is a GridSearchCV object
        if hasattr(self.svm_model, 'best_estimator_'):
            # Check if the best estimator itself has predict_proba
            if hasattr(self.svm_model.best_estimator_, 'predict_proba'):
                return self.svm_model.best_estimator_.predict_proba(X_scaled)
            else:
                raise AttributeError("Best estimator has no attribute 'predict_proba'. Set probability=True on SVC.")
        else:
            # Fallback if not GridSearchCV, check if svm_model itself has predict_proba
            if hasattr(self.svm_model, 'predict_proba'):
                return self.svm_model.predict_proba(X_scaled)
            else:
                raise AttributeError("Model has no attribute 'predict_proba'. Set probability=True on SVC.")
    
    def decision_function(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Cannot get decision function scores.") 
        X_scaled = self.scaler.transform(X)
        # Access the best estimator from GridSearchCV if svm_model is a GridSearchCV object
        if hasattr(self.svm_model, 'best_estimator_'):
            return self.svm_model.best_estimator_.decision_function(X_scaled)
        else:
            return self.svm_model.decision_function(X_scaled) # Fallback if not GridSearchCV
        
# --- SignLanguagePreprocessor Class Definition ---
# This class encapsulates the image preprocessing and feature extraction logic.
class SignLanguagePreprocessor:
    """Preprocessing class consistent with the training pipeline."""
    
    def __init__(self):
        # No scaler is needed here; the classifier object handles feature scaling.
        pass 
    
    def gaussian_filter_manual(self, image, sigma=0.5):
        """Manual implementation of Gaussian filter with mild smoothing."""
        image = self.normalize_image(image)
        
        size = int(2 * np.ceil(2 * sigma) + 1)
        if size < 3:
            size = 3
        
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        kernel = kernel / kernel.sum()
        
        filtered = ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def adaptive_contrast_stretching(self, image, min_percentile=0.5, max_percentile=99.5):
        """Adaptive contrast stretching that preserves details."""
        image = image.astype(np.float64)
        
        p_min = np.percentile(image, min_percentile)
        p_max = np.percentile(image, max_percentile)
        
        if p_max - p_min == 0:
            return self.normalize_image(image)
        
        gamma = 0.8
        normalized = (image - p_min) / (p_max - p_min)
        normalized = np.clip(normalized, 0, 1)
        gamma_corrected = np.power(normalized, gamma)
        stretched = gamma_corrected * 255
        
        return stretched.astype(np.uint8)
    
    def unsharp_mask(self, image, amount=0.3, radius=1.0):
        """Applies unsharp masking to enhance edges."""
        blurred = self.gaussian_filter_manual(image, sigma=radius)
        high_pass = image.astype(np.float64) - blurred.astype(np.float64)
        sharpened = image.astype(np.float64) + amount * high_pass
        sharpened = np.clip(sharpened, 0, 255)
        return sharpened.astype(np.uint8)
    
    def normalize_image(self, image):
        """Normalizes image pixel values to the 0-255 range."""
        image = image.astype(np.float64)
        img_min, img_max = image.min(), image.max()
        
        if img_max - img_min == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        normalized = (image - img_min) / (img_max - img_min) * 255
        return normalized.astype(np.uint8)
    
    def histogram_equalization(self, image):
        """Performs histogram equalization for contrast enhancement."""
        image = self.normalize_image(image)
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)
        equalized = cdf_normalized[image]
        return equalized
    
    def median_filter_manual(self, image, size=3):
        """Manual implementation of median filter."""
        return median_filter(image, size=size)
    
    def preprocess_image_with_steps(self, image):
        """Applies the complete preprocessing pipeline and stores intermediate steps for visualization."""
        # Convert to grayscale if the image is color
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        
        steps = {}
        steps['original'] = image.copy()
        
        image_norm = self.normalize_image(image)
        steps['normalized'] = image_norm.copy()
        
        if np.all(image_norm == 0):
            st.warning("Warning: Empty image detected, using original normalized image for further steps.")
            return image_norm, steps
        
        denoised = self.gaussian_filter_manual(image_norm, sigma=0.3)
        steps['gaussian_filtered'] = denoised.copy()
        
        local_var = ndimage.generic_filter(denoised.astype(np.float64), np.var, size=3)
        noise_threshold = np.percentile(local_var, 85)
        noise_mask = local_var > noise_threshold
        
        median_filtered = self.median_filter_manual(denoised, size=3)
        denoised = np.where(noise_mask, median_filtered, denoised)
        steps['selective_median'] = denoised.copy()
        
        enhanced = self.adaptive_contrast_stretching(denoised, min_percentile=0.5, max_percentile=99.5)
        steps['contrast_enhanced'] = enhanced.copy()
        
        hist_eq_applied = False
        if enhanced.std() < 30: # Apply histogram equalization only if contrast is low
            enhanced = self.histogram_equalization(enhanced)
            hist_eq_applied = True
        steps['after_hist_eq'] = enhanced.copy()
        steps['hist_eq_applied'] = hist_eq_applied
        
        final_image = self.unsharp_mask(enhanced, amount=0.2, radius=0.8)
        steps['final'] = final_image.copy()

        # Calculate edge difference for visualization
        edge_difference = final_image.astype(np.float64) - steps['after_hist_eq'].astype(np.float64)
        steps['edge_difference'] = edge_difference.copy()
        
        return final_image.astype(np.uint8), steps
    
    def preprocess_image(self, image):
        """Applies the complete preprocessing pipeline (returns final result only)."""
        final_image, _ = self.preprocess_image_with_steps(image)
        return final_image
    
    def extract_hog_features(self, image):
        """Extracts HOG features from an image."""
        features = hog(image, 
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       visualize=False,
                       transform_sqrt=True)
        return features
    
    def extract_lbp_features(self, image):
        """Extracts LBP features from an image."""
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7) # Add epsilon to prevent division by zero
        
        return hist
    
    def extract_features(self, image, feature_type='combined'):
        """Extracts features (HOG, LBP, or combined) from a single image."""
        processed_img, preprocessing_steps = self.preprocess_image_with_steps(image)
        
        # Fallback if preprocessing results in a black image
        if processed_img.max() == 0:
            st.warning("Warning: Processed image is completely black. Attempting feature extraction on original normalized image.")
            if len(image.shape) == 3:
                processed_img = np.mean(image, axis=2).astype(np.uint8)
            else:
                processed_img = image.copy()
            processed_img = self.normalize_image(processed_img) # Re-normalize if needed
        
        try:
            if feature_type == 'hog':
                features = self.extract_hog_features(processed_img)
            elif feature_type == 'lbp':
                features = self.extract_lbp_features(processed_img)
            elif feature_type == 'combined':
                hog_features = self.extract_hog_features(processed_img)
                lbp_features = self.extract_lbp_features(processed_img)
                features = np.concatenate([hog_features, lbp_features])
            
            # Handle potential NaN or Inf values in features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                st.warning("Warning: Invalid features detected (NaN/Inf). Replacing with zeros/finite numbers.")
                features = np.nan_to_num(features)
            
            return processed_img, features, preprocessing_steps
            
        except Exception as e:
            st.error(f"Error during feature extraction: {str(e)}")
            # Provide fallback zero features in case of extraction failure
            if feature_type == 'hog':
                features = np.zeros(324) # Default HOG feature size
            elif feature_type == 'lbp':
                features = np.zeros(26) # Default LBP feature size
            else:
                features = np.zeros(350) # Combined (HOG 324 + LBP 26)
            
            return processed_img, features, preprocessing_steps

# --- Cached Functions for Resource Loading ---
@st.cache_resource
def get_preprocessor():
    """Initializes and caches the SignLanguagePreprocessor object."""
    return SignLanguagePreprocessor()

@st.cache_resource
def load_full_classifier(path):
    """
    Loads the complete SignLanguageClassifier object from a joblib-exported file.
    This object includes the trained SVM model and its fitted StandardScaler.
    """
    try:
        with open(path, 'rb') as f:
            classifier_obj = joblib.load(f) # Using joblib.load
        
        # Crucial fix: Ensure the loaded object has the 'is_fitted' attribute
        # This covers cases where the training script didn't explicitly save it.
        if not hasattr(classifier_obj, 'is_fitted'):
            classifier_obj.is_fitted = True 
        
        st.success("✅ Classifier berhasil dimuat!")
        return classifier_obj
    except FileNotFoundError:
        st.error(f"❌ File classifier '{path}' tidak ditemukan!")
        st.info(f"💡 Pastikan file '{path}' ada di direktori yang sama dengan app.py.")
        st.stop() # Stop the app execution if model file is not found
    except Exception as e:
        st.error(f"❌ Error saat memuat classifier: {e}")
        st.info("💡 Pastikan Anda mengekspor model dengan joblib.dump(classifier, 'filename.pkl') dan definisi kelas di app.py sama persis dengan saat model dilatih.")
        st.stop() # Stop the app execution on other loading errors

# --- Visualization Function ---
def visualize_preprocessing_steps(preprocessing_steps):
    """Displays the various image preprocessing steps in Streamlit."""
    st.subheader("🔬 Tahapan Preprocessing Gambar")
    
    # Row 1 of images
    cols_row1 = st.columns(3)
    with cols_row1[0]:
        st.image(preprocessing_steps['original'], caption="1. Gambar Asli", use_column_width=True, clamp=True)
    with cols_row1[1]:
        st.image(preprocessing_steps['normalized'], caption="2. Normalisasi Pixel", use_column_width=True, clamp=True)
    with cols_row1[2]:
        st.image(preprocessing_steps['gaussian_filtered'], caption="3. Gaussian Filter (σ=0.3)", use_column_width=True, clamp=True)
    
    # Row 2 of images
    cols_row2 = st.columns(3)
    with cols_row2[0]:
        st.image(preprocessing_steps['selective_median'], caption="4. Median Filter Selektif", use_column_width=True, clamp=True)
    with cols_row2[1]:
        st.image(preprocessing_steps['contrast_enhanced'], caption="5. Peningkatan Kontras Adaptif", use_column_width=True, clamp=True)
    with cols_row2[2]:
        hist_eq_text = "6. Final & Peningkatan Tepi"
        if preprocessing_steps['hist_eq_applied']:
            hist_eq_text += " (+ Hist. Eq.)"
        st.image(preprocessing_steps['final'], caption=hist_eq_text, use_column_width=True, clamp=True)
    
    # Add Edge Enhancement Difference Map using Matplotlib
    st.markdown("---")
    st.subheader("⚡ Efek Peningkatan Tepi")
    fig_edge_diff, ax_edge_diff = plt.subplots(figsize=(8, 6))
    im = ax_edge_diff.imshow(preprocessing_steps['edge_difference'], cmap='RdBu_r', vmin=-50, vmax=50) # Adjust vmin/vmax as needed
    ax_edge_diff.set_title("Perbedaan Piksel setelah Peningkatan Tepi (Unsharp Masking)")
    ax_edge_diff.axis('off')
    fig_edge_diff.colorbar(im, ax=ax_edge_diff, label='Perbedaan Intensitas Piksel')
    st.pyplot(fig_edge_diff)
    plt.close(fig_edge_diff) # Close the figure to free up memory

    # Plotting histograms for comparison
    fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(preprocessing_steps['original'].flatten(), bins=50, alpha=0.7, color='blue')
    ax1.set_title('Histogram Gambar Asli')
    ax1.set_xlabel('Intensitas Piksel')
    ax1.set_ylabel('Frekuensi')
    
    ax2.hist(preprocessing_steps['final'].flatten(), bins=50, alpha=0.7, color='red')
    ax2.set_title('Histogram Gambar Akhir yang Diproses')
    ax2.set_xlabel('Intensitas Piksel')
    ax2.set_ylabel('Frekuensi')
    
    st.pyplot(fig_hist)
    plt.close(fig_hist) # Close the figure to free up memory

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
</style>
""", unsafe_allow_html=True)

# --- Main Streamlit Application Logic ---
def main():
    # Initialize the preprocessor (for image manipulation and feature extraction)
    preprocessor = get_preprocessor()

    # Application Title and Information
    st.markdown("<h1 class='main-header'>🤟 Klasifikasi Angka Bahasa Isyarat</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Identifikasi Angka (0–9) dari Gambar Tangan</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>🚀 Cara Penggunaan:</strong><br>
        1. Pilih sumber gambar (unggah dari perangkat atau ambil langsung dengan kamera web).<br>
        2. Pilih tipe fitur yang ingin digunakan untuk analisis gambar.<br>
        3. Pastikan gambar tangan Anda menunjukkan angka (0-9) dengan jelas.<br>
        4. Aplikasi akan secara otomatis menampilkan hasil klasifikasi dan tahapan pemrosesan gambar.<br><br>
        <strong>⚡ Teknologi:</strong> Model SVM (Support Vector Machine) dengan ekstraksi fitur HOG (Histogram of Oriented Gradients) dan LBP (Local Binary Pattern) yang dikombinasikan, didukung oleh pipeline preprocessing gambar yang menjaga detail tepi dan meningkatkan kontras secara adaptif.
    </div>
    """, unsafe_allow_html=True)

    # Feature Type Selection
    st.markdown("<div class='method-box'><strong>🔧 Pilih Tipe Fitur:</strong></div>", unsafe_allow_html=True)
    feature_type = st.selectbox(
        "Tipe Fitur untuk Ekstraksi:",
        options=['combined', 'hog', 'lbp'],
        index=0, # Default ke 'combined'
        help="Gabungan (Combined): Mengombinasikan HOG dan LBP untuk akurasi yang lebih baik. HOG: Fokus pada bentuk dan kontur. LBP: Fokus pada tekstur dan pola lokal."
    )

    feature_info = {
        'combined': "🔥 **Fitur Gabungan (HOG + LBP)**: Metode paling kuat, menggabungkan informasi bentuk dan tekstur untuk akurasi maksimal.",
        'hog': "📊 **Fitur HOG**: Histogram of Oriented Gradients - efektif untuk menangkap bentuk dan kontur utama tangan.",
        'lbp': "🔍 **Fitur LBP**: Local Binary Pattern - baik untuk mendeskripsikan tekstur dan pola halus pada gambar."
    }
    st.info(feature_info[feature_type])

    # --- Load the classifier dynamically based on selected feature type ---
    # This is the key change to match your training script's output
    model_filename = f'sign_language_model_{feature_type}.pkl'
    classifier = load_full_classifier(model_filename) 

    # Image Source Selection: Upload or Camera Input
    choice = st.radio("📸 Pilih sumber gambar untuk klasifikasi:", ("Unggah Gambar", "Ambil Gambar dari Kamera"))

    image_data = None
    if choice == "Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah file gambar (JPG atau PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_data = Image.open(uploaded_file).convert("RGB") # Ensure RGB for consistent processing
    else: # choice == "Ambil Gambar dari Kamera"
        camera_image = st.camera_input("Ambil Gambar dari Kamera")
        if camera_image is not None:
            # Read bytes from camera_image and convert to PIL Image
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
        with st.spinner(f'⚙️ Memproses gambar dan mengekstraksi {feature_type.upper()} features...'):
            # Pass the selected feature_type to the preprocessor
            processed_img, features, preprocessing_steps = preprocessor.extract_features(image_resized, feature_type=feature_type)

        # Visualize preprocessing steps
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
            # Evaluate preprocessing quality based on standard deviation
            preprocessing_quality = processed_img.std() / 255.0 * 100
            st.metric("Kualitas Pemrosesan (Std. Dev.)", f"{preprocessing_quality:.1f}%")
        with col6:
            # Calculate contrast improvement
            original_gray_std = np.mean(image_resized, axis=2).std() # Ensure grayscale for std calculation
            contrast_improvement = (processed_img.std() - original_gray_std) / (original_gray_std + 1e-7) * 100
            st.metric("Peningkatan Kontras", f"{contrast_improvement:+.1f}%")

        # Perform prediction and display results
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        try:
            # Use the loaded 'classifier' object for all prediction-related methods
            prediction = classifier.predict([features])[0]
            
            probabilities = None
            if hasattr(classifier, 'predict_proba'): 
                probabilities = classifier.predict_proba([features])[0]
            
            confidence = None
            if hasattr(classifier, 'decision_function'):
                decision_scores = classifier.decision_function([features])[0]
                confidence = np.max(decision_scores)
            
            if probabilities is not None:
                # Get top 3 predicted classes and their probabilities
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                
                confidence_text = f"Skor Keyakinan (Decision Function): {confidence:.3f}" if confidence is not None else "Probabilitas tidak tersedia."
                st.markdown(f"""
                <div class='prediction-box'>
                    <p class='prediction-text'>Prediksi Angka: <span style='color: #E91E63;'>{prediction}</span></p>
                    <p style='text-align: center; color: #666; font-size: 1.2em;'>
                        {confidence_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🏆 Top 3 Prediksi Teratas:")
                for i, idx in enumerate(top_3_indices):
                    prob_percent = probabilities[idx] * 100
                    col_pred1, col_pred2 = st.columns([1, 3])
                    with col_pred1:
                        st.write(f"**{i+1}. Angka {idx}**")
                    with col_pred2:
                        st.progress(prob_percent/100)
                        st.write(f"{prob_percent:.1f}%")
            else:
                confidence_text = f"Skor Keyakinan (Decision Function): {confidence:.3f}" if confidence is not None else "Probabilitas tidak tersedia."
                st.markdown(f"""
                <div class='prediction-box'>
                    <p class='prediction-text'>Prediksi Angka: <span style='color: #E91E63;'>{prediction}</span></p>
                    <p style='text-align: center; color: #666; font-size: 1.2em;'>
                        {confidence_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            # Expandable section for detailed pipeline information
            with st.expander("🔍 Detail Pipeline Pemrosesan"):
                st.markdown(f"""
                **Tahapan Pipeline yang Diterapkan:**
                1. ✅ **Resizing Gambar:** Gambar diubah ukurannya menjadi 64x64 piksel untuk konsistensi.
                2. ✅ **Konversi Grayscale:** Gambar dikonversi ke skala abu-abu untuk pemrosesan fitur.
                3. ✅ **Normalisasi Nilai Piksel:** Nilai piksel dinormalisasi ke rentang 0-255.
                4. ✅ **Reduksi Derau (Gaussian):** Filter Gaussian ringan (sigma=0.3) diterapkan untuk mengurangi derau dengan tetap menjaga tepi.
                5. ✅ **Filter Median Selektif:** Filter median diterapkan secara selektif pada area yang mungkin mengandung derau impuls.
                6. ✅ **Peningkatan Kontras Adaptif:** Kontras gambar ditingkatkan secara adaptif menggunakan peregangan persentil dan koreksi gamma.
                7. ✅ **Ekualisasi Histogram Kondisional:** Ekualisasi histogram diterapkan jika kontras gambar setelah peningkatan adaptif masih rendah.
                8. ✅ **Peningkatan Tepi (Unsharp Masking):** Detail tepi diasah menggunakan teknik unsharp masking.
                9. ✅ **Ekstraksi Fitur:** Fitur {feature_type.upper()} diekstrak dari gambar yang telah diproses.
                10. ✅ **Scaling Fitur & Prediksi SVM:** Fitur diskalakan menggunakan StandardScaler yang dilatih dan digunakan untuk prediksi oleh model SVM.
                """)
                
                st.markdown("**Statistik Detail Gambar (Min/Max/Mean/Std. Dev.):**")
                stats_data = {
                    "Gambar Asli (Setelah Resize)": preprocessing_steps['original'],
                    "Gambar Akhir yang Diproses": preprocessing_steps['final'],
                    "Perbedaan Peningkatan Tepi": preprocessing_steps['edge_difference']
                }
                
                for name, img in stats_data.items():
                    # Handle display for edge_difference specifically for proper range
                    if name == "Perbedaan Peningkatan Tepi":
                        st.markdown(f"""
                        <div class='stats-box'>
                        <strong>{name}:</strong> Min={img.min():.2f}, Max={img.max():.2f}, 
                        Mean={img.mean():.2f}, Std={img.std():.2f} (Range: -255 to 255)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='stats-box'>
                        <strong>{name}:</strong> Min={img.min():.2f}, Max={img.max():.2f}, 
                        Mean={img.mean():.2f}, Std={img.std():.2f} (Range: 0 to 255)
                        </div>
                        """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {str(e)}")
            st.write("**Informasi Debug:**")
            st.write(f"- Bentuk fitur yang diekstraksi: {features.shape}")
            st.write(f"- Tipe fitur: {type(features)}")
            st.write(f"- Rentang nilai fitur: {features.min():.4f} - {features.max():.4f}")
            st.write(f"- Tipe model SVM internal: {type(classifier.svm_model)}") 
            st.write(f"- Tipe scaler internal: {type(classifier.scaler)}")


if __name__ == "__main__":
    main()