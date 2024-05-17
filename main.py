import streamlit as st
import numpy as np
from keras.models import load_model
import preprocess
from preprocess import AppError
import pandas as pd

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
    st.title('Machine Learning Model Deployment with Streamlit')
    st.header('User Input')

    input_data = st.file_uploader(label="Upload Image (.png/.jpg)", type=['png', 'jpg'])
    if st.button('Predict') and input_data is not None:

        # Preprocess input data
        try:
            result = []
            preprocessed_data = preprocess.preprocess(input_data)
            # Display predictions
            st.write('### Prediction')
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
                "File Name": input_data.name,
                "Prediction": prediction_result,
                "Details": pred_detail
            })

            #Create the table
            table_rows = []
            
            for item in result:
                table_rows.append([item["File Name"], item["Prediction"], item["Details"]])

            # Create a DataFrame from the data
            df = pd.DataFrame(table_rows, columns=["File Name", "Prediction", "Details"])

            # Show as a static table
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

            # Display predictions
            st.write('### Image Uploaded')
            st.image(input_data, width=100, caption=input_data.name)
        except AppError as a:
            st.error(error_code[a.message])

if __name__ == '__main__':
    main()
