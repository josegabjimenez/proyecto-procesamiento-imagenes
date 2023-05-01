import streamlit as st
import matplotlib.pyplot as plt # Plots

st.title("Preprocesamiento")

st.session_state["hola"] = "HOLA MI PERRO"

image = st.session_state["image"]
# Plot image
fig, ax = plt.subplots()
ax.imshow(image[:, :, 20])
st.pyplot(fig)