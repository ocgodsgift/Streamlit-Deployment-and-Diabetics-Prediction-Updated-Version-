import streamlit as st

st.set_page_config(
    page_title= 'Poise Project',
    page_icon= ":stethoscope:",
    layout='wide'
)

with st.container():
    st.divider()
    
    st.image(r'pages\2023_07_15_11_12_IMG_4214.PNG', width=50)

    st.subheader("Machine Models Prediction")
    
    with st.sidebar:
        st.subheader("Select")
        split = st.slider("#### Train Test Split",0.20,0.30,0.25)
        random_state = st.slider("#### Random State",1,42,12)

    if st.button("Model Prediction"):
        from data_clean import data_clean
        data_clean(split,random_state)
