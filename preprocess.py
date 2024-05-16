import dlib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

angles = [0, 45, 90, 135]
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

# Load the scaler
scaler = joblib.load('scaler.joblib')

# -------------------- GLCM functions ------------------------

def calc_glcm_all_agls(img, props, dists, agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    return np.array(feature)

# -------------------- Cropping function ------------------------

def crop_and_extract_glcm(file, dists=[1], props=properties):
    image = cv2.resize(file, (128, 128))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram of the grayscale image
    gray_equalized = cv2.equalizeHist(gray)

    # Detect faces in the grayscale image
    faces = detector(gray_equalized, 0)

    if len(faces) == 0:
        print(f"No face detected in")
        return None
    else:
        for face in faces:
            # Predict facial landmarks for the detected face
            landmarks = predictor(gray_equalized, face)

            # Define indices for eyes, eyebrows, and mouth
            left_eye_pts = list(range(36, 42))  # Indices for left eye (36-41)
            right_eye_pts = list(range(42, 48))  # Indices for right eye (42-47)
            left_eyebrow_pts = list(range(17, 22))  # Indices for left eyebrow (17-21)
            right_eyebrow_pts = list(range(22, 27))  # Indices for right eyebrow (22-26)
            mouth_pts = list(range(48, 68))  # Indices for mouth (48-67)

            # Extract landmark coordinates for eyes, eyebrows, and mouth
            left_eye_coords = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in left_eye_pts]
            right_eye_coords = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in right_eye_pts]
            left_eyebrow_coords = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in left_eyebrow_pts]
            right_eyebrow_coords = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in right_eyebrow_pts]
            mouth_coords = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in mouth_pts]

            # Function to create a mask for the specified region
            def create_mask(coords, image_shape):
                mask = np.zeros(image_shape, dtype=np.uint8)
                pts = np.array(coords, np.int32)
                cv2.fillPoly(mask, [pts], 255)
                return mask

            # Create masks for eyes, eyebrows, and mouth regions
            eye_mask = create_mask(left_eye_coords, gray.shape) + create_mask(right_eye_coords, gray.shape)
            left_eyebrow_mask = create_mask(left_eyebrow_coords, gray.shape)
            right_eyebrow_mask = create_mask(right_eyebrow_coords, gray.shape)
            mouth_mask = create_mask(mouth_coords, gray.shape)

            # Combine masks to create a mask for eyes, eyebrows, and mouth
            eyes_eyebrows_mouth_mask = cv2.bitwise_or(eye_mask, left_eyebrow_mask)
            eyes_eyebrows_mouth_mask = cv2.bitwise_or(eyes_eyebrows_mouth_mask, right_eyebrow_mask)
            eyes_eyebrows_mouth_mask = cv2.bitwise_or(eyes_eyebrows_mouth_mask, mouth_mask)

            # Apply the mask to the original image to extract eyes, eyebrows, and mouth
            extracted_features_image = cv2.bitwise_and(image, image, mask=eyes_eyebrows_mouth_mask)

            # Convert the extracted features image to grayscale
            extracted_gray = cv2.cvtColor(extracted_features_image, cv2.COLOR_BGR2GRAY)

            # Calculate GLCM features
            glcm_features = calc_glcm_all_agls(extracted_gray, props, dists)
            return glcm_features

# -------------------- Function to get GLCM features for an image ------------------------

def get_glcm_features(image_path):
    glcm_features = crop_and_extract_glcm(image_path)
    if glcm_features is not None and glcm_features.size > 0:
        glcm_features = glcm_features.reshape(1, -1)  # Reshape for the scaler
        scaled_features = scaler.transform(glcm_features)
        return scaled_features
        # return glcm_features
    else:
        return None

def preprocess(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None and image.size > 0:
        features = get_glcm_features(image)
        if features is not None and features.size > 0:
            return features
        else:
            return None
    else:
        print("Error: Unable to decode or load the uploaded image.")