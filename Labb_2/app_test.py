import streamlit as st

st.markdown("<p style='color: blue;'>Custom Label for Slider:</p>", unsafe_allow_html=True)
slider_value = st.slider("", 0, 100, 50) # Empty label for the actual slider
st.write(f"Selected value: {slider_value}")

col=":orange[This text is dark orange]"
min_val=0
max_val=100
val=25
st.slider(f'**{col}**', min_value = float(min_val), max_value = float(max_val), value = float(val))