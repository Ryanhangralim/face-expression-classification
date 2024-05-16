import streamlit as st
import numpy as np
from keras.models import load_model
import preprocess


def predict_class(input_data):
    model = load_model("PPDMD5.h5")
    predictions = model.predict(np.array(input_data))
    return predictions

# Streamlit app layout
def main():
    st.title('Machine Learning Model Deployment with Streamlit')
    st.header('User Input')

    # Example: Collect user input
    # Implement your specific input components (e.g., sliders, text inputs)
    input_data = st.file_uploader(label="Upload Image")

    if st.button('Predict') and input_data is not None:

        # Preprocess input data
        # try:
            preprocessed_data = preprocess.preprocess(input_data)
            # Get model predictions

            # Display predictions
            st.write('### Predictions')
            if preprocessed_data is not None:
                predictions = predict_class(preprocessed_data)
                st.image(input_data, use_column_width=True)
                st.write(predictions[0])
                max_index = np.array(predictions[0]).argmax()
                print(max_index)
                if max_index == 0:
                    st.write("HAPPY")
                elif max_index == 1:
                    st.write("NEUTRAL")
                elif max_index == 2:
                    st.write("SAD")
            else:
                st.write("No Face Detected")
        # except BaseException:
        #     st.write("Model can't analyze image")

if __name__ == '__main__':
    main()
