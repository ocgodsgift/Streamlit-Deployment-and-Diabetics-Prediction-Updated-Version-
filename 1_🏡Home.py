import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import utils as ut
from visuals import visuals


df = ut.load_dataset()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title= 'Poise Project',
    page_icon= ":stethoscope:",
    layout='wide'
)

with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        st.write("##")
        
        st.title("Diabetes Prediction :stethoscope:")
        st.subheader("This project showcases the application of machine learning classifiers for predicting diabetes.")
        st.write("""
                    By leveraging features derived from this Dataset and employing Gradient Boosting Classifier,
                    The Random Forest Classifier, Logistic Regression, and Support Vector Classifier. We aim to accurately identify individuals at risk of diabetes.
                
                """ )
    with right_col:
        st.write("#")
        st.image('2023_07_15_11_09_IMG_4213.PNG', width=300)

with st.container():
        st.subheader('Dataset Sample')
        code = '''
                 df.sample(20)
               '''
        st.code(code,language='python')
        st.dataframe(df.sample(20),use_container_width=True)

with st.container():
    st.divider()

    left_col, right_col = st.columns((2,1))
    with left_col:

        st.write('#### Description')
        st.write('Statistical insights of the dataset')
        st.dataframe(df.describe(), use_container_width=True)
        st.dataframe(df.isnull().sum().rename('Missing Values').to_frame().T,use_container_width=True) 
        st.markdown(f""" ##### Shape
        The dataset has {df.shape[0]} rows and {df.shape[1]} columns
        """)

    with right_col:
        st.write("##")
        st.write('##')
        st.write('##')
        plt.figure(figsize=(4,3.5))
        sns.countplot(x='diabetes', data=df)
        plt.ylabel("frequency")
        plt.title('Imbalanced Diabetes Distribution')
        st.pyplot()
        st.markdown(" ##### Group By")
        st.dataframe(df['diabetes'].value_counts(), use_container_width=True) 
      

    st.divider()
    st.write("#### Data Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""##### Observation
                            Minimum Age recorded was 0.8
                """)
        
        st.write("#")
        st.write("In this analysis, rows with Age less than 20 and Gender with Others were dropped.")
        code = '''
        
                df['age'] = df['age'].astype(int)
                # Create a new Dataset as df_new with age greater or equal to 20
                df_new = df[df['age'] >= 20]

                # Remove rows on column gender with value Other
                df_new = df_new.drop(df_new[df_new['gender'] == 'Other'].index, axis=0))

                    '''

        st.code(code, language='python')

        df['age'] = df['age'].astype(int)
        # Create a new Dataset as df_new with age greater or equal to 20
        df_new = df[df['age'] >= 20]

        # Remove rows on column gender with value Other
        df_new = df_new.drop(df_new[df_new['gender'] == 'Other'].index, axis=0)

    with col2:

        # Dataframe of diabetes eqaul to One
        df_positive = df_new[df_new['diabetes'] == 1]
        # Dataframe of diabetes eqaul to Zero
        df_negative = df_new[df_new['diabetes'] == 0].sample(len(df_positive))

        # To validate the concatenated Dataframe rows
        validate_rows = len(df_negative) + len(df_positive)

        # Concatenate the Positive and Negative Dataframe
        df_clean = pd.concat([df_positive, df_negative]).sample(validate_rows)


        st.write("##### To form a balance data to avoid bias in model prediction")
        
        code = f'''
                # Dataframe of diabetes eqaul to One
                df_positive = df_new[df_new['diabetes'] == 1]
                # Dataframe of diabetes eqaul to Zero
                df_negative = df_new[df_new['diabetes'] == 0].sample(len(df_positive))

                # To validate the concatenated Dataframe rows
                validate_rows = len(df_negative) + len(df_positive)

                # Concatenate the Positive and Negative Dataframe
                df_clean = pd.concat([df_positive, df_negative]).sample(validate_rows)

                New Dataset has {df_clean.shape[0]} rows and {df_clean.shape[1]} columns

            '''
        st.code(code, language='python')

    st.divider()


with st.container():

    st.write("#### New Dataset Overview")
    col1, _ = st.columns([4,4])
    with col1:
        code = '''
                df_clean.sample(20)
                '''
        st.code(code,language='python')

    col1, col2 = st.columns((2,1))

    with col1:
        st.dataframe(df_clean.sample(20), use_container_width=True)

    with col2:
        plt.figure(figsize=(4,3.5))
        sns.countplot(x='diabetes', data=df_clean)
        plt.title('Balanced Diabetes Distribution')
        plt.ylabel("frequency")
        st.pyplot()

with st.container():

    st.divider()
    st.subheader("Data Insights Unleased:")
    if st.button("View Charts"):
        visuals(df_clean)


