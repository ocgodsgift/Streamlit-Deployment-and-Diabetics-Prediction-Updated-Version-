import streamlit as st
import time

with st.container():
    st.subheader("Power BI Report Embedded in Streamlit:")
    st.write('###### Gain deeper insights and enhanced analytics on the Dataset')

    waiting = st.progress(0)

    for perc_completed in range(100):
        time.sleep(0.005)
        waiting.progress(perc_completed + 1)

    import streamlit.components.v1 as components

    iframe_code = f'<iframe title="Report Section" width="1400" height="1000" src="https://app.powerbi.com/view?r=eyJrIjoiZTY5MzQzZGItYTdiYy00MGRmLWE0OWMtYmI3MmQyZDZkMzBhIiwidCI6ImNlMDU2NTFlLWY5NGMtNDk5Ny04YzI1LTVhMjM5OWZmZGFlYiJ9" frameborder="0" allowFullScreen="true"></iframe>'

    # Display the Power BI report in Streamlit using the HTML component
    components.html(iframe_code, width=1400, height=1000)