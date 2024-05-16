import dlib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os 
import re
import pandas as pd

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
distance = [1]
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

# -------------------- Utility function ------------------------
def normalize_desc(folder):
    text = folder
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

# ----------------- calculate graycomatrix() & graycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists, agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
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
    feature.append(label) 
    
    return feature

def crop(file, image_label):
    # Load face image and resize it to a larger size
    image = cv2.imread(file)
    image = cv2.resize(image, (128, 128))

    # Convert image into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram of the grayscale image
    gray_equalized = cv2.equalizeHist(gray)

    # Detect faces in the grayscale image
    faces = detector(gray_equalized, 0)

    if len(faces) == 0:
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
            glcm_features = calc_glcm_all_agls(extracted_gray, label=image_label, props=properties, dists=distance)
            return glcm_features
        
if __name__ == "__main__":
    dataset_dir = "DATASET"

    imgs = [] 
    descs = []
    extracted_images = []
    #loops through dataset
    for folder in os.listdir(dataset_dir):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder))
        len_sub_folder = len(sub_folder_files) - 1
        folder_name = normalize_desc(folder)
        for i, filename in enumerate(sub_folder_files):
            img_path = f"{dataset_dir}/{folder_name}/{filename}"
            features = crop(file=img_path, image_label=folder_name)
            if features is not None:
                extracted_images.append(features)

    columns = []
    angles = ['0', '45', '90','135']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)
    columns.append("label")

    print(f"Columns: {columns}")
    print(extracted_images)

      # Create the pandas DataFrame for GLCM features data
    glcm_df = pd.DataFrame(extracted_images, columns = columns)

    #save to csv
    glcm_df.to_csv(f"result/result.csv", index=False)