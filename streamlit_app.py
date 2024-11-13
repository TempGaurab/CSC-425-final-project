import streamlit as st
import cv2
import numpy as np
from PIL import Image
from part1 import main  # Function for person detection
from part3 import main3  # Function for image classification
from part2 import get_output
from io import BytesIO
import os
import zipfile

def get_image_download_bytes(pil_image, format='PNG'):
    """Convert PIL Image to bytes for downloading"""
    buf = BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL format"""
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def about_the_club():
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: --text-color;
        margin-bottom: 20px;
    }
    .club-description {
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 30px;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #003366;
    }
    .team-member {
        margin-bottom: 15px;
        color: --text-color;
        padding: 10px;
        border-radius: 5px;
    }
    .team-member-role {
        font-weight: bold;
        color: --text-color;
    }
    .team-member-name {
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">About the Project</p>', unsafe_allow_html=True)
    
    st.subheader("Our Team")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="team-member">
            <span class="team-member-role">Gaurab Baral</span><br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="team-member">
            <span class="team-member-role">Sushant Shrestha</span><br>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="team-member">
            <span class="team-member-role">Abhishek Shrestha</span><br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="team-member">
            <span class="team-member-role">Nabin Lama</span><br>
        </div>
        """, unsafe_allow_html=True)



# Title of the app
st.title("🎈 CSC-425-3-STEP-MODEL-DETECTION 🎈")
st.write(
    '''
The proposed system uses a three-model approach for driver drowsiness detection, where the first model activates only when the car is running and checks for a person in the driver's seat. When a person is detected, the second model locates the driver's eyes and sends these images to a custom-built third model. The third model analyzes eye features to detect drowsiness, potentially triggering various alert mechanisms like intermittent braking or honking if the driver appears to be sleeping. 
    '''
)

# Sidebar navigation
st.sidebar.title("Navigation")
model = st.sidebar.radio("Select a model:", 
    ("Model 1: Person Detection", 
     "Model 2: Eye Extraction", 
     "Model 3: Image Classification",
     "Models Together",
     "Team Members"))

# Display content based on the selected model
if model == "Model 1: Person Detection":
    st.header("Model 1: Person Detection")
    st.subheader("Detect whether an image has a driver or not!")
    
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Display uploaded image
        pil_image = cv2_to_pil(image)
        st.image(pil_image, caption='Uploaded Image', use_container_width  =True)
        
        # Button for detection
        if st.button("Detect Person"):
            person_detected = main(image)  # Pass the image to the main function
            st.write(f"👤 Person detected: {person_detected}")  # Display result
            
            # Add download button for the processed image
            img_bytes = get_image_download_bytes(pil_image)
            st.download_button(
                label="Download Image",
                data=img_bytes,
                file_name="person_detection.png",
                mime="image/png"
            )
    else:
        st.warning("Please upload an image to detect a person.")

elif model == "Model 2: Eye Extraction":
    st.header("Model 2: Eye Extraction")
    st.subheader("Extract the eye from the image of the driver.")

    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Read image directly from uploaded file using BytesIO
            file_bytes = BytesIO(uploaded_file.read())
            image = Image.open(file_bytes)
            
            # Convert to RGB if image is in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Display uploaded image
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button("Extract Eyes"):
                with st.spinner("Extracting eyes..."):
                    try:
                        # Save image temporarily with proper error handling
                        temp_file = "temp_upload.jpg"  # Using jpg instead of png for better compatibility
                        image.save(temp_file, format='JPEG', quality=95)
                        
                        # Get the annotated image and both eyes
                        annotated_image, left_eye, right_eye = get_output(temp_file)
                        
                        # Convert CV2 format to PIL for display
                        if isinstance(annotated_image, np.ndarray):
                            st.image(annotated_image, caption='Detected Face Landmarks', use_container_width=True)
                        
                            if left_eye is not None and right_eye is not None:
                                st.success("Successfully extracted both eyes!")
                                
                                # Ensure eye images are in correct format
                                left_eye_pil = left_eye
                                right_eye_pil = right_eye
                                
                                # Display extracted eyes
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(left_eye_pil, caption='One Eye', use_container_width=True)
                                with col2:
                                    st.image(right_eye_pil, caption='Other Eye', use_container_width=True)

                                left_eye_pil = Image.fromarray(left_eye)
                                right_eye_pil = Image.fromarray(right_eye)
                                # Create ZIP file with extracted eyes

                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                    # Save left eye
                                    left_eye_buffer = BytesIO()
                                    left_eye_pil.save(left_eye_buffer, format='PNG')
                                    zip_file.writestr("left_eye.png", left_eye_buffer.getvalue())
                                    
                                    # Save right eye
                                    right_eye_buffer = BytesIO()
                                    right_eye_pil.save(right_eye_buffer, format='PNG')
                                    zip_file.writestr("right_eye.png", right_eye_buffer.getvalue())
                                
                                # Add download button
                                st.download_button(
                                    label="Download Extracted Eyes",
                                    data=zip_buffer.getvalue(),
                                    file_name="extracted_eyes.zip",
                                    mime="application/zip"
                                )
                            else:
                                st.warning("Could not detect eyes clearly in the image. Please ensure the face is clearly visible.")
                        else:
                            st.error("Failed to process the image. Please try with a different image.")
                            
                    except Exception as e:
                        st.error(f"Error during eye extraction: {str(e)}")
                        st.info("Try uploading a different image with a clearly visible face.")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception:
                                pass
                            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.info("Please ensure you're uploading a valid image file.")
    else:
        st.info("Please upload an image to extract eyes.")



elif model == "Model 3: Image Classification":
    st.header("Model 3: Sleepiness Detection!")
    st.subheader("Check for drowsiness in the eye of the driver.")
    
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.image(image, caption='Uploaded Image', use_container_width =True)

        # Button for classification
        if st.button("Classify Image"):
            result = main3(image)  # Pass the image to the main3 function
            st.write(f"🖼️ Prediction: {result}")  # Display the prediction result
            
            # Add download button for the processed image
            img_bytes = get_image_download_bytes(image)
            st.download_button(
                label="Download Image",
                data=img_bytes,
                file_name="drowsiness_detection.png",
                mime="image/png"
            )
    else:
        st.warning("Please upload an image for drowsiness detection.")

        

elif model == "Models Together":
    st.header("Complete Drowsiness Detection Pipeline")
    st.subheader("Run all three models in sequence")
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"], key="combined")
    
    if uploaded_file is not None:
        # Read the uploaded image
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Display uploaded image
        pil_image = cv2_to_pil(image)
        st.image(pil_image, caption='Uploaded Image', use_container_width  =True)
        if st.button("Run Complete Analysis"):
            st.write("🔄 Running complete analysis...")

            # Step 1: Person Detection (from Model 1)
            st.write("Step 1: Person Detection")
            person_detected = main(image)  # Assuming `main()` is your person detection function
            st.write(f"👤 Person detected: {person_detected}")
            
            if person_detected:
                # Step 2: Eye Extraction (from Model 2)
                st.write("Step 2: Eye Extraction")
                try: 
                    try: 
                        # Get the annotated image and both eyes
                        annotated_image, left_eye, right_eye = get_output(image)
                        
                        # Convert CV2 format to PIL for display
                        if isinstance(annotated_image, np.ndarray):
                            st.image(annotated_image, caption='Detected Face Landmarks', use_container_width=True)
                        
                            if left_eye is not None and right_eye is not None:
                                st.success("Successfully extracted both eyes!")
                                
                                # Ensure eye images are in correct format
                                left_eye_pil = left_eye
                                right_eye_pil = right_eye
                                
                                # Display extracted eyes
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(left_eye_pil, caption='One Eye', use_container_width=True)
                                with col2:
                                    st.image(right_eye_pil, caption='Other Eye', use_container_width=True)

                                left_eye_pil = Image.fromarray(left_eye)
                                right_eye_pil = Image.fromarray(right_eye)
                                # Create ZIP file with extracted eyes               
                    except Exception as e:
                        st.error(f"Error during eye extraction: {str(e)}")
                        st.info("Try uploading a different image with a clearly visible face.")          
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    st.info("Please ensure you're uploading a valid image file.")
    else:
        st.info("Please upload an image to extract eyes.")



elif model == "Team Members":
    about_the_club()