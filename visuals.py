def visuals(df_clean):
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    with st.container():
        waiting = st.progress(0)

        for perc_completed in range(100):
            time.sleep(0.005)
            waiting.progress(perc_completed + 1)


        inner1, inner2, inner3 = st.columns((2,2,2))
        with inner1:
            plt.figure(figsize=(5,4))
            sns.histplot(x='age', bins=50, kde=True, data=df_clean)
            plt.title("Age Distribution")
            plt.ylabel('Frequency')
            st.pyplot() 
            
            plt.figure(figsize=(5,4))
            sns.countplot(x='gender', data=df_clean)
            plt.title('Gender Distribution')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.countplot(x='hypertension', data=df_clean)
            plt.ylabel('Frequency')
            plt.title('Hypertension Distribution')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.countplot(x='heart_disease', hue='diabetes',  palette='Set1', data=df_clean)
            plt.ylabel('Frequency')
            plt.title('Heart Disease Distribution')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.lineplot(y='age', x='blood_glucose_level', data=df_clean)
            plt.title('Age vs Blood Glucose Level')
            st.pyplot()

        with inner2:
            plt.figure(figsize=(5,4))            
            sns.boxplot(x='diabetes', y='age', hue='gender', palette='Set1', data=df_clean)
            plt.title("Age")
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.boxplot(y='age', x='diabetes', hue='gender', palette='Set1', data=df_clean)
            plt.title("Age vs Diabetes with Gender")
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.histplot(x='blood_glucose_level', kde=True, data=df_clean)
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.countplot(x='smoking_history', data=df_clean)
            plt.title("Smoking History")
            plt.ylabel("Frequency")
            plt.xlabel('smoking history')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.countplot(x='gender', hue='diabetes', palette='Set1', data=df_clean)
            plt.title('Gender vs Diabetes Distribution')
            plt.ylabel("Frequency")
            st.pyplot()
        
        with inner3:

            plt.figure(figsize=(5,4))
            sns.countplot(x='hypertension', hue='diabetes', palette='Set1', data=df_clean)
            plt.title('Hypertension vs Diabetes Distribution')
            plt.ylabel('Frequency')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.countplot(x='heart_disease', data=df_clean)
            plt.ylabel('Frequency')
            plt.title('Heart Disease Distribution')
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.boxplot(x='hypertension', y='age', hue='diabetes', data=df_clean)
            plt.title("Hypertension vs Diabetes")
            st.pyplot()

            plt.figure(figsize=(5,4))
            sns.boxplot(y='age', x='diabetes', data=df_clean)
            plt.title("Age vs Diabetes")
            st.pyplot()    
            
            plt.figure(figsize=(5,4))
            sns.boxplot(y='blood_glucose_level', x='diabetes', data=df_clean)
            plt.title("Blood Glucose Level")
            st.pyplot()
