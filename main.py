import streamlit as st
import numpy as np
from keras.models import load_model
import preprocess
from preprocess import AppError
import pandas as pd
import math

error_code = {
    "1" : "Error: Unable to decode or load the uploaded image.",
    "2" : "Error: No face detected in image.",
    "3" : "Error: Unable to resize image.",
    "4" : "Error: Unable to extract image.",
    "5" : "Error: Unable to fetch image features.",
}
#load model
model = load_model("PPDMD5.h5")

def predict_class(input_data):
    predictions = model(np.array(input_data))
    return predictions

# Streamlit app layout
def main():
    st.title('Face Expression Classification')
    st.header('User Input')

    input_data = st.file_uploader(label="Upload Image (.png/.jpg)", type=['png', 'jpg'], accept_multiple_files=True)
    if st.button('Predict') and input_data is not None:

        # Preprocess input data
        try:
            result = []
            # Display predictions
            st.write('### Prediction')
            for data in input_data:
                preprocessed_data = preprocess.preprocess(data)
                # Get model predictions
                predictions = predict_class(preprocessed_data)
                max_index = np.array(predictions[0]).argmax()
                if max_index == 0:
                    prediction_result = "Happy"
                elif max_index == 1:
                    prediction_result = "Neutral"
                elif max_index == 2:
                    prediction_result = "Sad"

                # Append the result to the result list
                pred_nump = predictions.numpy()
                pred_detail = f'Happy : {format(pred_nump[0][0], ".4f")}<br>Neutral : {format(pred_nump[0][1], ".4f")}<br>Sad : {format(pred_nump[0][2], ".4f")}'
                result.append({
                    "File Name": data.name,
                    "Prediction": prediction_result,
                    "Details": pred_detail
                })

            # Create a DataFrame from the data
            df = pd.DataFrame(result, columns=["File Name", "Prediction", "Details"])

            # Show as a static table
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

            st.write('### Table of images')

            #calculate media rows and columns
            image_count = len(input_data)
            rows = math.ceil(image_count/5)
            columns = 5
            

            for i in range(rows): # number of rows in your table! = 2
                cols = st.columns(columns) # number of columns in each row! = 2
                # first column of the ith row
                for j in range(columns):
                # Calculate the index of the current image
                    index = i * columns + j
                    if index < image_count:  # Ensure that index is within bounds
                        # Display the image in the current column
                        cols[j].image(input_data[index], width=100, caption=input_data[index].name)

        except AppError as a:
            st.error(error_code[a.message])

if __name__ == '__main__':
    main()
