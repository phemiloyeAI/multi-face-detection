import streamlit as st

from face_detection import process_uploaded_file

mode = st.sidebar.selectbox("Select Input type", options=["Image", "Video"])
st.sidebar.write("---")
st.sidebar.write("Confidence Threshold")
conf_thresh = st.sidebar.slider("Confidence value", min_value=0, max_value=100, value=30)
st.sidebar.write("---")
st.sidebar.write("IoU Threshold")
iou_thresh = st.sidebar.slider("Intersection over Union value", min_value=0, max_value=100, value=50)

model_weights = "weights\yolov5-blazeface.pt"
device = "cpu"
output_path = "out_image.jpg" if mode == "Image" else "out_video.webm"

conf_thresh = conf_thresh / 100
iou_thresh = iou_thresh / 100

if mode == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_name = str(uploaded_file.name)
        with st.spinner("Processing image progress"):
            process_uploaded_file(model_weights, file_name, conf_thresh, iou_thresh, device, output_path)
            st.image(output_path)

        with open(output_path, "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="image.png",
                    mime="image/png"
                )

if mode == "Video":
    video_file = st.file_uploader("Choose a video", type=["mp4", "webm", "mkv", "avi", "asf"])
    if video_file:
        if st.button("Process Video"):  
            st.write("\n")
            st.write("***Processing video progress.***")
            process_uploaded_file(model_weights, video_file.name, conf_thresh, iou_thresh, device, output_path)
            st.video(output_path)

            with open(output_path, "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="video.webm",
                        mime="video/webm"
                    )
