import streamlit as st
from streamlit_webrtc import webrtc_streamer
import VideoProcessor


st.title("My Stick Figure Streamlit app")

option = st.selectbox("option?",
                     ["Nomal", "姿勢推定", "棒人間"])

if option == "Nomal":
    video_processor_factory = VideoProcessor.Nomal_VideoProcessor

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=video_processor_factory,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )


elif option == "姿勢推定":
    video_processor_factory = VideoProcessor.mp_pose_VideoProcessor

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=video_processor_factory,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    if ctx.video_processor:
        st.write("Parameters")
        ctx.video_processor.radius = int(st.slider("点の大きさ", min_value=1.0, max_value=5.0, step=1.0, value=3.0))
        ctx.video_processor.thickness = int(st.slider("線の太さ", min_value=1.0, max_value=5.0, step=1.0, value=2.0))
        ctx.video_processor.detection = st.slider("検出度", min_value=0.0, max_value=1.0, step=0.1, value=0.5)




elif option == "棒人間":
    video_processor_factory = VideoProcessor.StickFigure_VideoProcessor
    
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=video_processor_factory,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    if ctx.video_processor:
        st.write("Parameters")
        ctx.video_processor.radius = int(st.slider("顔の大きさ", min_value=1.0, max_value=500.0, step=1.0, value=30.0))
        ctx.video_processor.thickness = int(st.slider("手足の太さ", min_value=1.0, max_value=100.0, step=1.0, value=15.0))
        ctx.video_processor.R = st.slider('赤', min_value=0, max_value=255, value=0)
        ctx.video_processor.G = st.slider('緑', min_value=0, max_value=255, value=255)
        ctx.video_processor.B = st.slider('青', min_value=0, max_value=255, value=0)

