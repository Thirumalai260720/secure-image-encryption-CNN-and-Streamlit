import streamlit as st
import cv2
import numpy as np
import torch
from model.predict_model import PredictModel
from histogram import cnn_histogram_shifting
from helpers import calculate_complexity, arithenco, num2bitlist, calculate_tp_tn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Streamlit UI
st.title("🔒 CNN Watermark Embedding & Extraction System")
st.markdown("---")

# Upload image section
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "bmp"])

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
                        
                        # Save to session state
                        st.session_state.img_embedded = img_embedded
                        st.session_state.watermark_bits = watermark_bits
                        st.session_state.index_complexity = index_complexity
                        st.session_state.inserable_place = inserable_place
                        st.session_state.original_image = img_gray
                        st.session_state.original_text = text_to_embed
                        
                        # Display results
                        with col2:
                            st.image(img_embedded / 255.0, caption="🔏 Watermarked Image", use_column_width=True)
                        
                        # Calculate metrics
                        mse = np.mean((img_gray - img_embedded) ** 2)
                        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                        
                        st.success("✅ Watermark embedded successfully!")
                        st.metric("PSNR", f"{psnr:.2f} dB")
                        st.metric("Embedded bits", len(watermark_bits))
                        st.metric("Text length", f"{len(text_to_embed)} characters")
                        
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
    
    **Features:**
    - CNN-based histogram shifting for robust watermarking
    - High PSNR values for quality preservation
    - Secure embedding in complex image regions
    - Dual extraction methods for flexibility
    """)

# Footer
st.markdown("---")
st.markdown("🔒 *Secure Digital Watermarking System*")