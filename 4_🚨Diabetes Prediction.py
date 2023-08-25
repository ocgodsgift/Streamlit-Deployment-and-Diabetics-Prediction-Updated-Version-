import streamlit as st
import pandas as pd
import numpy as np
import utils as ut
import time

st.set_page_config(
    page_title= 'Poise Project',
    page_icon= ":stethoscope:",
    layout='wide'
)
    
df = ut.load_dataset()

df['age'] = df['age'].astype(int)
df_new = df[df['age'] >= 20]

df_new = df_new.drop(df_new[df_new['gender'] == 'Other'].index, axis=0)

df_positive = df_new[df_new['diabetes'] == 1]
df_negative = df_new[df_new['diabetes'] == 0].sample(len(df_positive))

validate_rows = len(df_negative) + len(df_positive)
df_clean = pd.concat([df_positive, df_negative]).sample(validate_rows)

gender = {
"Male":1,
"Female":0
}

smoking_history = {
    'never': 0,
    'No Info': 1,
    'ever': 2,
    'former': 3,
    'not current': 4,
    'current': 5
}

df_copy = df_clean.copy()

df_copy['gender'] = df_copy['gender'].map(gender)
df_copy['smoking_history'] = df_copy['smoking_history'].map(smoking_history)

scaling = ['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

sc =  StandardScaler()
X_scale = sc.fit_transform(df_copy[scaling])

tranScale = pd.DataFrame(X_scale, columns=['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
tranScale = tranScale.reset_index(drop=True)

subset_df = df_copy.drop(['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'], axis=1)
subset_df = subset_df.reset_index(drop=True)

featured_df = pd.concat([tranScale, subset_df], axis=1)

from sklearn.model_selection import train_test_split

final_features = ['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'hypertension', 'heart_disease']

X = featured_df[final_features].values
y = featured_df.iloc[:,-1].values

X_train, _, y_train, _ = train_test_split(X,y,test_size=0.25, random_state=42)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

with st.container():
    st.subheader('Diabetes Prediction')
    inner1, _ = st.columns((3,4))
    with inner1:
        data = {
            'gender':[st.selectbox("##### Gender: 0-Female | 1-Male", options=[0,1])],
            'age': [st.slider("##### Age:",20,120,value=30)],
            'hypertension': [st.selectbox('##### Hypertension: 0-No | 1-Yes',options=[0,1])],
            'heart_disease': [st.selectbox('##### Heart Disease: 0-No | 1-Yes',options=[0,1])],
            'smoking_history': [st.selectbox("##### Smoking History: 0-Never | 1-No Info | 2-Current | 3-Former | 4-Ever | 5-Not Current",options=[0,1,2,3,4,5])],
            'bmi': [st.slider("##### BMI:",5.0,50.0,value=10.0)],
            'HbA1c_level': [st.slider("##### HbA1c Level:",0.0,10.0,value=5.0)],
            'blood_glucose_level': [st.slider(" ##### Blood Glucose Level:",10,250,value=50)]
            }

        new_data = pd.DataFrame(data)

        X_new_data = sc.transform(new_data[scaling])
        X_new_data = np.hstack((X_new_data, new_data[['gender', 'hypertension','heart_disease']].values))
        
        new_ypred = model.predict(X_new_data)

        result = new_ypred[0]
        

        if st.button("Predict"):
            with st.spinner('Predicting'):
                 time.sleep(2)
            if result == 1:
                st.success('Positive !!', icon="✅")
            else:
                st.error('Negative !!', icon="🚨")
            

        