def data_clean(split,random_state):    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    import time
    import utils as ut
    
    
    df = ut.load_dataset()

    st.set_option('deprecation.showPyplotGlobalUse', False)

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
    
    inner1,_ = st.columns((3,4))
    with inner1:
        plt.figure(figsize=(8,6))
        corr = df_clean[['age','bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'diabetes']].corr()
        sns.heatmap(corr, annot=True, cmap='Blues')
        plt.title("Correlation")    
        st.pyplot()

    st.divider()

    with st.spinner('Modeling'):
         time.sleep(2)
    st.success("Model Prediction Succesfull !")
    
    df_copy = df_clean.copy()

    df_copy['gender'] = df_copy['gender'].map(gender)
    df_copy['smoking_history'] = df_copy['smoking_history'].map(smoking_history)

    scaling = ['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    from sklearn.preprocessing import StandardScaler

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

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=random_state)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import classification_report

    models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'SVC': SVC(),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
    }

    with st.container():
        for name, model in models.items():
            # Training model
            model.fit(X_train,y_train)

            # Create Prediction from trained model
            y_pred = model.predict(X_test)

            # Determine the score and confusion matrix
            score = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Classification report
            report = classification_report(y_test, y_pred)
            
            inner1, inner2 = st.columns(2)
            with inner1:
                # print model name
                st.markdown(f""" ##### Model:
                            {name}
                        """)

                st.divider()

                # print accuracy score
                st.markdown(f"""##### Accuracy Score:
                         {round(np.floor(score * 100))}%
                         """)
                
                st.divider()

                # print classification report
                st.write("##### Classification Report:")
                st.text(report)

            with inner2:
                # Create a heatmap showing the confusion
                st.write("#")
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, fmt='d', cmap='Blues', annot=True)
                plt.title("Confusion Matrix")
                st.pyplot()

            st.divider()
