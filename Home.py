import streamlit as st
import os

img_path = "demo\image.jpg"
video_path = "demo\\video.webm"

st.title("A Crowd Counting Application.")

st.write("***The result of a video processed by the app***")
st.video(video_path)

st.write('---')
st.write("***The result of an image processed by the app***")
st.image(img_path)


st.sidebar.header("About")