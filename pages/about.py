import streamlit as st
def app():
    st.header("About")
    st.write("This app uses a lightweight face-detector to detect regions containing faces from image and video streams\
         and give a total count of the people present.\
         It combines face-tracking with the face detector to give an accurate estimate of the people present in a video stream.")