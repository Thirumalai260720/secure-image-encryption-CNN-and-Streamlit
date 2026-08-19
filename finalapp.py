import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model.predict_model import PredictModel
from histogram import cnn_histogram_shifting
from helpers import calculate_complexity, arithenco, num2bitlist, calculate_tp_tn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set matplotlib style
plt.style.use('default')
sns.set_style("whitegrid")

# Load CNN model with proper error handling
model = None
try:
    # Load state dict first to check
    state_dict = torch.load('./model_parameter/model_state.pth', map_location=device)
    
    # Handle different state dict formats
    actual_state_dict = state_dict
    if 'network' in state_dict:
        actual_state_dict = state_dict['network']
    elif 'model_state_dict' in state_dict:
        actual_state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        actual_state_dict = state_dict['state_dict']
    
    # Create and load model
    model = PredictModel(device)
    model.model.load_state_dict(actual_state_dict, strict=True)
    model.model.to(device)
    model.model.eval()
    st.success("✅ CNN model loaded successfully!")
    
except Exception as e:
    st.warning(f"⚠️ Model loading issue: {e}")
    st.info("⚠️ Using placeholder predictions - embedding will still work")
    model = None

def calculate_metrics(original, watermarked):
    """Calculate various image quality metrics"""
    # Ensure images are the same size and type
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    # MSE (Mean Squared Error)
    mse = np.mean((original - watermarked) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM (Structural Similarity Index) - simplified version
    # Note: For full SSIM, you might want to use scikit-image
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu_x = np.mean(original)
    mu_y = np.mean(watermarked)
    sigma_x = np.std(original)
    sigma_y = np.std(watermarked)
    sigma_xy = np.cov(original.flatten(), watermarked.flatten())[0, 1]
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    # Correlation coefficient
    correlation = np.corrcoef(original.flatten(), watermarked.flatten())[0, 1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'correlation': correlation,
        'max_diff': np.max(np.abs(original - watermarked))
    }

def plot_quality_metrics(metrics_history):
    """Plot quality metrics over multiple embeddings"""
    if len(metrics_history) < 2:
        st.info("Need at least 2 embeddings to show trends")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR trend
    axes[0, 0].plot(range(len(metrics_history)), [m['psnr'] for m in metrics_history], 
                   marker='o', linewidth=2, color='blue')
    axes[0, 0].set_title('PSNR Trend', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Embedding Iteration')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE trend
    axes[0, 1].plot(range(len(metrics_history)), [m['mse'] for m in metrics_history], 
                   marker='s', linewidth=2, color='red')
    axes[0, 1].set_title('MSE Trend', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Embedding Iteration')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM trend
    axes[1, 0].plot(range(len(metrics_history)), [m['ssim'] for m in metrics_history], 
                   marker='^', linewidth=2, color='green')
    axes[1, 0].set_title('SSIM Trend', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Embedding Iteration')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation trend
    axes[1, 1].plot(range(len(metrics_history)), [m['correlation'] for m in metrics_history], 
                   marker='d', linewidth=2, color='purple')
    axes[1, 1].set_title('Correlation Trend', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Embedding Iteration')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_difference_analysis(original, watermarked):
    """Plot difference analysis between original and watermarked images"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Watermarked image
    axes[0, 1].imshow(watermarked, cmap='gray')
    axes[0, 1].set_title('Watermarked Image', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference image
    difference = np.abs(original - watermarked)
    im_diff = axes[0, 2].imshow(difference, cmap='hot')
    axes[0, 2].set_title('Absolute Difference', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)
    
    # Histogram of original image
    axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
    axes[1, 0].set_title('Original Image Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Histogram of watermarked image
    axes[1, 1].hist(watermarked.flatten(), bins=50, alpha=0.7, color='red', label='Watermarked')
    axes[1, 1].set_title('Watermarked Image Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Difference histogram
    axes[1, 2].hist(difference.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 2].set_title('Difference Histogram')
    axes[1, 2].set_xlabel('Absolute Difference')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_embedding_analysis(watermark_bits, extracted_bits=None):
    """Plot watermark embedding analysis"""
    fig, axes = plt.subplots(1, 2 if extracted_bits is not None else 1, figsize=(12, 5))
    
    if extracted_bits is None:
        # Single plot for embedding only
        axes = [axes] if not hasattr(axes, '__len__') else axes
    
    # Original watermark bits
    axes[0].stem(range(len(watermark_bits)), watermark_bits, basefmt=" ")
    axes[0].set_title('Original Watermark Bits', fontweight='bold')
    axes[0].set_xlabel('Bit Position')
    axes[0].set_ylabel('Bit Value')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    if extracted_bits is not None and len(extracted_bits) == len(watermark_bits):
        # Extracted watermark bits
        axes[1].stem(range(len(extracted_bits)), extracted_bits, basefmt=" ", linefmt='red')
        axes[1].set_title('Extracted Watermark Bits', fontweight='bold')
        axes[1].set_xlabel('Bit Position')
        axes[1].set_ylabel('Bit Value')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_bit_error_analysis(original_bits, extracted_bits):
    """Plot bit error analysis"""
    if len(original_bits) != len(extracted_bits):
        st.warning("Bit sequences have different lengths")
        return
    
    # Calculate bit errors
    errors = [1 if orig != ext else 0 for orig, ext in zip(original_bits, extracted_bits)]
    error_positions = [i for i, error in enumerate(errors) if error == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error positions
    if error_positions:
        axes[0].stem(error_positions, [1] * len(error_positions), basefmt=" ")
    axes[0].set_title('Bit Error Positions', fontweight='bold')
    axes[0].set_xlabel('Bit Position')
    axes[0].set_ylabel('Error (1 = Error)')
    axes[0].set_ylim(0, 1.5)
    axes[0].grid(True, alpha=0.3)
    
    # Error rate by byte
    byte_errors = []
    for i in range(0, len(errors), 8):
        byte_errors.append(sum(errors[i:i+8]))
    
    axes[1].bar(range(len(byte_errors)), byte_errors)
    axes[1].set_title('Errors per Byte', fontweight='bold')
    axes[1].set_xlabel('Byte Position')
    axes[1].set_ylabel('Number of Errors')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calculate overall statistics
    total_bits = len(original_bits)
    error_count = sum(errors)
    error_rate = (error_count / total_bits) * 100
    
    st.metric("Bit Error Rate", f"{error_rate:.2f}%")
    st.metric("Total Errors", f"{error_count}/{total_bits}")

def preprocess_image_for_cnn(image):
    """Preprocess image for CNN prediction"""
    try:
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor / 255.0
        return image_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit UI
st.title("🔒 AI CNN Watermark Embedding & Extraction System")
st.markdown("---")

# Initialize session state for metrics history
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []

# Upload image section
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None:
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_gray / 255.0, caption="🖼️ Original Grayscale Image", use_column_width=True)
        
        # Image info
        st.info(f"Image dimensions: {img_gray.shape[1]} x {img_gray.shape[0]} pixels")
        
        # Watermark text input
        text_to_embed = st.text_input("💧 Enter text to embed as watermark", max_chars=100)
        
        # Embed watermark section
        if text_to_embed and st.button("🔒 Embed Watermark", type="primary"):
            with st.spinner("Embedding watermark..."):
                try:
                    # Convert text to binary bits
                    watermark_bits = np.array([int(b) for char in text_to_embed for b in format(ord(char), '08b')])
                    
                    # Calculate maximum capacity
                    max_capacity = (img_gray.shape[0] * img_gray.shape[1]) // 8
                    if len(text_to_embed) > max_capacity:
                        st.error(f"Text too long! Maximum capacity: {max_capacity} characters")
                    else:
                        # Create predicted image
                        predicted_image = np.zeros_like(img_gray)
                        
                        if model is not None:
                            try:
                                # Use CNN model for prediction
                                input_tensor = preprocess_image_for_cnn(img_gray)
                                if input_tensor is not None:
                                    predicted_tensor = model.test_on_batch(input_tensor)
                                    predicted_image = predicted_tensor.squeeze().cpu().numpy() * 255.0
                                    predicted_image = np.clip(predicted_image, 0, 255).astype(np.float64)
                                    st.success("✅ Used CNN model for prediction")
                            except Exception as e:
                                st.warning(f"CNN prediction failed, using fallback: {e}")
                                predicted_image = img_gray.copy()
                        else:
                            predicted_image = img_gray.copy()
                            st.info("ℹ️ Using original image as prediction (model not available)")
                        
                        # Embed watermark using histogram shifting
                        img_embedded, index_complexity, inserable_place = cnn_histogram_shifting(
                            img_gray,                     # Original grayscale image
                            predicted_image,              # Predicted image from CNN
                            watermark_bits,               # Watermark bits
                            0,                            # odd_or_even_num
                            calculate_complexity,
                            arithenco,
                            num2bitlist,
                            calculate_tp_tn
                        )
                        
                        # Calculate quality metrics
                        metrics = calculate_metrics(img_gray, img_embedded)
                        
                        # Save to session state
                        st.session_state.img_embedded = img_embedded
                        st.session_state.watermark_bits = watermark_bits
                        st.session_state.index_complexity = index_complexity
                        st.session_state.inserable_place = inserable_place
                        st.session_state.original_image = img_gray
                        st.session_state.original_text = text_to_embed
                        st.session_state.current_metrics = metrics
                        
                        # Add to metrics history
                        st.session_state.metrics_history.append(metrics)
                        
                        # Display results
                        with col2:
                            st.image(img_embedded / 255.0, caption="🔏 Watermarked Image", use_column_width=True)
                        
                        st.success("✅ Watermark embedded successfully!")
                        
                        # Display metrics in columns
                        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                        with col_metrics1:
                            st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                        with col_metrics2:
                            st.metric("MSE", f"{metrics['mse']:.2f}")
                        with col_metrics3:
                            st.metric("SSIM", f"{metrics['ssim']:.4f}")
                        with col_metrics4:
                            st.metric("Correlation", f"{metrics['correlation']:.4f}")
                        
                        st.metric("Embedded bits", len(watermark_bits))
                        st.metric("Text length", f"{len(text_to_embed)} characters")
                        
                        # Show visualizations
                        st.markdown("---")
                        st.subheader("📊 Quality Analysis")
                        
                        # Difference analysis
                        plot_difference_analysis(img_gray, img_embedded)
                        
                        # Watermark bits visualization
                        plot_embedding_analysis(watermark_bits)
                        
                        # Metrics trend (if multiple embeddings)
                        if len(st.session_state.metrics_history) > 1:
                            st.subheader("📈 Metrics Trend Over Multiple Embeddings")
                            plot_quality_metrics(st.session_state.metrics_history)
                        
                except Exception as e:
                    st.error(f"Error during embedding: {e}")

# Extract watermark section
st.markdown("---")
st.subheader("🔍 Watermark Extraction")

# Option 1: Extract from embedded image in session
if 'img_embedded' in st.session_state and st.button("Extract Watermark from Embedded Image", type="secondary"):
    with st.spinner("Extracting watermark..."):
        try:
            imgW = st.session_state.img_embedded.copy()
            watermark_bits = st.session_state.watermark_bits
            index_complexity = st.session_state.index_complexity
            inserable_place = st.session_state.inserable_place
            
            # Extract bits from watermarked image using LSB extraction
            extracted_bits = []
            for k in range(len(watermark_bits)):
                if k < len(index_complexity):
                    i, j = inserable_place[k]
                    # Extract LSB (Least Significant Bit)
                    pixel_value = int(imgW[i, j])
                    extracted_bit = pixel_value & 1  # Get the LSB
                    extracted_bits.append(extracted_bit)
            
            # Convert bits to text
            extracted_text = ""
            for i in range(0, len(extracted_bits), 8):
                byte = extracted_bits[i:i+8]
                if len(byte) == 8:
                    char_code = int("".join(str(b) for b in byte), 2)
                    extracted_text += chr(char_code)
            
            # Display extraction results
            col3, col4 = st.columns(2)
            with col3:
                st.image(st.session_state.original_image / 255.0, 
                        caption="📷 Original Image", use_column_width=True)
            with col4:
                st.image(st.session_state.img_embedded / 255.0, 
                        caption="🔏 Watermarked Image", use_column_width=True)
            
            st.subheader("Extraction Results")
            
            # Show original and extracted text for comparison
            col5, col6 = st.columns(2)
            with col5:
                st.text_area("📝 Original Watermark Text", 
                           st.session_state.original_text, height=100)
            with col6:
                st.text_area("📝 Extracted Watermark Text", extracted_text, height=100)
            
            # Verification
            if extracted_text == st.session_state.original_text:
                st.success("✅ Watermark verification: PASSED - Exact match!")
            else:
                # Check if it's a partial match
                min_len = min(len(extracted_text), len(st.session_state.original_text))
                matches = sum(1 for i in range(min_len) if extracted_text[i] == st.session_state.original_text[i])
                accuracy = (matches / len(st.session_state.original_text)) * 100
                
                st.warning(f"⚠️ Watermark verification: {accuracy:.1f}% match")
                st.info(f"Original length: {len(st.session_state.original_text)}, Extracted length: {len(extracted_text)}")
            
            # Show bit-level analysis
            st.subheader("🔍 Bit-level Analysis")
            plot_embedding_analysis(watermark_bits, extracted_bits)
            plot_bit_error_analysis(watermark_bits, extracted_bits)
                
        except Exception as e:
            st.error(f"Error during extraction: {e}")

# Option 2: Upload a watermarked image for extraction
st.markdown("---")

# Instructions
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    **Embedding:**
    1. Upload an image
    2. Enter text to embed
    3. Click 'Embed Watermark'
    
    **Extraction (Two Methods):**
    1. **From Session**: Extract from the recently embedded image (preserves embedding positions)
    2. **From Upload**: Upload any watermarked image for extraction (uses LSB analysis)
    
    **New Visualization Features:**
    - **Quality Metrics**: PSNR, MSE, SSIM, Correlation trends
    - **Difference Analysis**: Visual comparison of original vs watermarked
    - **Bit-level Analysis**: Watermark bit patterns and error analysis
    - **Histogram Comparison**: Pixel distribution changes
    
    **Features:**
    - CNN-based histogram shifting for robust watermarking
    - High PSNR values for quality preservation
    - Secure embedding in complex image regions
    - Dual extraction methods for flexibility
    """)

# Footer
st.markdown("---")
st.markdown("🔒 *Secure Digital Watermarking System with Advanced Analytics*")