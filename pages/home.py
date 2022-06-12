import streamlit as st
import os

def app():
    st.title("A Crowd Counting Application.")
    st.write("***The result of a video processed by the app***")
    st.video("./demo/video.webm")

    st.write('---')
    st.write("***The result of an image processed by the app***")
    st.image("./demo/image.jpg")

    