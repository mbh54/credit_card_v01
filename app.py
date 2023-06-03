import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import random
import base64


def plot_image_with_source(image_path, source_text):
    with open(image_path, "rb") as file:
        contents = file.read()

    data_url = base64.b64encode(contents).decode("utf-8")
    image_tag = f'<img src="data:image/jpeg;base64,{data_url}" alt="image" width="600" height="250">'
    source_tag = f'<p style="font-family:sans-serif; font-size:10px;">Image source: {source_text}</p>'
    image_with_source = f"{image_tag}\n{source_tag}"

    return image_with_source


def plot_cross_validation_results(model):
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']

    c_values = [param['C'] for param in params]
    c_values_str = [str(c) for c in c_values]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(c_values_str, means, yerr=stds, fmt='o-', markersize=3, capsize=4, linewidth=.5)
    ax.set_xlabel('C values', fontsize=7)
    ax.set_ylabel('Mean Test Score', fontsize=7)
    ax.set_title('Cross-Validation Results', fontsize=8)
    ax.set_xticklabels(c_values_str, rotation=20, fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    fontprops = fm.FontProperties(size=5) 
    ax.legend(prop=fontprops)
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{data}"

    return data_url


def credit_card_app():

    title = '<p style="font-family:sans-serif; color:Gray; font-size: 30px;">Credit Card Defualt Prediction Web App</p>' 
    st.markdown(title, unsafe_allow_html=True)

    image_path = "images/Probability-of-default-definition.jpeg"
    source_text = "https://capital.com/probability-of-default-definition"
    image_with_source = plot_image_with_source(image_path, source_text)
    st.markdown(image_with_source, unsafe_allow_html=True) 
    st.write('\n')
    st.write('\n')


    explanation = '''
    <p style="font-family: sans-serif; color: black; font-size: 14px;">
    Credit default prediction plays a crucial role in effectively managing risk within a consumer lending business. It empowers lenders to make informed lending decisions, leading to an enhanced customer experience and robust business economics.
    </p>

    <p style="font-family: sans-serif; color: black; font-size: 14px;">
    Introducing our web app, we developed a credit default predictor that utilizes customer information to estimate the probability of default. Our predictor is trained using a logistic regression classifier and achieves an accuracy of 75% on an independent test dataset.
    </p>

    <p style="font-family: sans-serif; color: black; font-size: 14px;">
    As an example, we provide a user-friendly interface where you can automatically generate customer information and use the model to predict the probability of default. In summary, our model generates reliable predictions that can be used by lenders to make well-informed decisions, manage risk effectively, and ensure the financial well-being of their lending business.
    </p>
    
    <p style="font-family: sans-serif; color: black; font-size: 14px;">
    Below are the cross-validation results for the model:
    </p>
    '''
    st.markdown(explanation, unsafe_allow_html=True)

    model = load('model/pre_trained_model.joblib')
    cross_val_plot_url = plot_cross_validation_results(model)
    # Display the plot in Streamlit
    st.image(cross_val_plot_url)

    st.sidebar.markdown('''#### Want to know more about defaults and their impacts? See below:''')
    st.sidebar.markdown('''
    - [Credit card default: How it happens, what to do about it](https://www.bankrate.com/finance/credit-cards/credit-card-default)
    - [Default vs delinquency: How they impact credit](https://www.chase.com/personal/credit-cards/education/build-credit/default-vs-delinquency)
    ''')

    st.markdown("""<style>.stButton button {background-color: #02075d  ; color: white;} </style>""", unsafe_allow_html=True)

    if 'random_number_clicked' not in st.session_state:
        st.session_state.random_number_clicked = False
    
    random_number_placeholder = st.empty()
    random_number = st.button('Get Customer Information')
    
    df_random = pd.read_pickle('data/random_data_container.pkl')  # Replace with your actual file path
    
    if 'selected_customer' not in st.session_state:
        st.session_state.selected_customer = None
    
    if random_number or st.session_state.random_number_clicked:
        try:
            total_customers = len(df_random)
            feature_names = df_random.columns.tolist()
            random_index = random.randint(0, total_customers - 1)
            selected_customer = df_random.iloc[random_index]
            st.session_state.selected_customer = selected_customer
            st.write('\n')
            text = '<p style="font-family:sans-serif; color: black; font-size: 15px;">This is the customer information: </p>'
            st.markdown(text, unsafe_allow_html=True)
            #st.write('This is the customer information for the random data that is generated: ')
            st.write('\n')
            
            st.write(selected_customer)  
            st.markdown("<p style='font-size:18px;line-height:0.6;'><strong></strong> Variable categories in the above table:</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:14px; line-height:0.6;'><strong> D_*:</strong> Delinquency variables</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:14px; line-height:0.6;'><strong> S_*:</strong> Spend variables</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:14px; line-height:0.6;'><strong> P_*:</strong> Payment variables</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:14px; line-height:0.6;'><strong> B_*:</strong> Balance variables</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:14px; line-height:0.6;'><strong> R_*:</strong> Risk variables</p>", unsafe_allow_html=True)

        except Exception as e:
            st.write(f"Error: {e}")
    
    if random_number:
        st.session_state.random_number_clicked = True
        
    st.write('\n')
    st.write('\n')
    second_button_result = st.button('Get Predictions')

    if second_button_result and st.session_state.selected_customer is not None:
        selected_customer = st.session_state.selected_customer
        
        input_sample = pd.DataFrame(selected_customer.values.reshape(1, -1), columns=feature_names)
        prediction = model.predict(input_sample)
        probabilities = model.predict_proba(input_sample)
        positive_class_probabilities = probabilities[:, 1]
        negative_class_probabilities = probabilities[:, 0]    

        st.write(f'<p style="font-family: sans-serif; font-size: 16px; color: black;"><strong>The probability of this customer defaulting is: <span style="background-color: yellow;">{round(positive_class_probabilities[0], 3)}</span></strong></p>', unsafe_allow_html=True)

        st.write('\n')

        st.write('The top five most important features in this prediction are: ')
        feature_weights = model.best_estimator_.coef_[0]
        feature_weights_df = pd.DataFrame({'Feature': feature_names, 'Weight': feature_weights})
        feature_weights_df['Absolute Weight'] = abs(feature_weights_df['Weight'])
        feature_weights_df = feature_weights_df.sort_values('Absolute Weight', ascending=False)
        top_features = feature_weights_df.nlargest(5, 'Absolute Weight')
        top_features = top_features.reset_index(drop=True)
        top_features = top_features.drop('Weight', axis=1)
        st.write(top_features)

if __name__ == '__main__':
    credit_card_app()
    