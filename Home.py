import streamlit as st
import os

@st.cache
def disp_demo_image():
    st.image("./demo/image.jpg")

@st.cache
def disp_demo_video():
    st.video("./demo/video.webm")

st.title("A Crowd Counting Application.")
st.write("***The result of a video processed by the app***")
disp_demo_video()

st.write('---')
st.write("***The result of an image processed by the app***")
disp_demo_image()

st.sidebar.header("About")