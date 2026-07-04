"""G2B Posture Correction Coach — live Streamlit app."""
import os
import queue
import cv2
import av
import streamlit as st
from src.cv.pipeline import PosturePipeline 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- Hardware Auto-Detection ---
@st.cache_data
def is_cloud_environment():
    try:
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            return True
        cap.release()
        return False
    except Exception:
        return True

IS_CLOUD = is_cloud_environment()

from src.workers.shared_state import SharedPostureState
from src.workers.chat_worker import ChatWorker

if not IS_CLOUD:
    from src.workers.cv_worker import CVWorker
    from src.utils.config import mediapipe_complexity, target_fps, camera_resolution

st.set_page_config(page_title="G2B Posture Coach", layout="wide")
st.markdown("# G2B Posture Correction Coach")
st.caption("Live posture analysis + AI coaching")

# === Init singletons ===
if "shared" not in st.session_state:
    st.session_state.shared = SharedPostureState()
    st.session_state.chat_worker = ChatWorker(st.session_state.shared)
    st.session_state.chat_history = []
    
    # Initialize UI metrics at absolute zero
    st.session_state.shared.posture_class = "Waiting for video..."
    st.session_state.shared.confidence = 0.0
    st.session_state.shared.ear_shoulder_offset_x = 0.0
    st.session_state.shared.shoulder_roll_z = 0.0
    st.session_state.shared.shoulder_tilt_angle = 0.0

    if not IS_CLOUD:
        try:
            w, h = camera_resolution()
            st.session_state.cv_worker = CVWorker(
                st.session_state.shared,
                width=w, height=h,
                model_complexity=mediapipe_complexity(),
                target_fps=target_fps(),
            )
            st.session_state.cv_worker.start()
        except Exception:
            pass

# === WebRTC AI Processor (Handles Bounding Box & Live AI) ===
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        # Queue to securely pass real-time AI data to the Streamlit UI
        self.result_queue = queue.Queue()
        # Initialize your actual Mapúa thesis AI pipeline!
        self.pipe = PosturePipeline(model_complexity=1)
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert web frame to an image array
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 1. PROCESS AI HERE: 
        # Pass the live web frame into your custom MediaPipe math
        state = self.pipe.step(img)
        
        if state is not None:
            # 2. DRAW BOUNDING BOX ON VIDEO
            # Draws a green rectangle around the center of the frame
            cv2.rectangle(img, (int(w*0.15), int(h*0.1)), (int(w*0.85), int(h*0.95)), (0, 255, 0), 2)
            
            # Draw the real-time verdict and accuracy text on the video feed
            label = f"{state.posture_class.upper()} | Conf: {state.confidence:.2f}"
            cv2.putText(img, label, (int(w*0.15), int(h*0.1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. SEND TO UI
            # Push the real calculated degrees to the Streamlit dashboard
            self.result_queue.put({
                "posture_class": state.posture_class,
                "confidence": state.confidence,
                "forward_head": state.ear_shoulder_offset_x,
                "shoulder_roll": state.shoulder_roll_z,
                "tilt": state.shoulder_tilt_angle
            })

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
# === Layout: two columns ===
left, right = st.columns([1, 1])

with left:
    st.subheader("Live view")
    
    # Render WebRTC video player
    ctx = webrtc_streamer(
        key="g2b-posture-coach-stream",
        video_processor_factory=PostureProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # Fetch live data from the video processor if it is running
    if ctx.state.playing and ctx.video_processor:
        try:
            live_data = ctx.video_processor.result_queue.get(timeout=0.1)
            st.session_state.shared.posture_class = live_data["posture_class"]
            st.session_state.shared.confidence = live_data["confidence"]
            st.session_state.shared.ear_shoulder_offset_x = live_data["forward_head"]
            st.session_state.shared.shoulder_roll_z = live_data["shoulder_roll"]
            st.session_state.shared.shoulder_tilt_angle = live_data["tilt"]
        except queue.Empty:
            pass
    elif not ctx.state.playing:
        # Reset back to zero if recording/streaming stops
        st.session_state.shared.posture_class = "Waiting for video..."
        st.session_state.shared.confidence = 0.0
        st.session_state.shared.ear_shoulder_offset_x = 0.0
        st.session_state.shared.shoulder_roll_z = 0.0
        st.session_state.shared.shoulder_tilt_angle = 0.0

    st.markdown("---")
    
    # Render UI Metrics based on current state
    state = st.session_state.shared
    LABEL_EMOJI = {"correct_posture": "🟢", "slouching": "🟠", "neck_forward": "🟡", "lean": "🟣", "Waiting for video...": "⚪"}
    
    # Fallback if dictionary key doesn't match perfectly
    emoji = LABEL_EMOJI.get(state.posture_class.lower().replace(" ", "_"), "⚪") 
    
    st.markdown(f"### {emoji} **{state.posture_class.replace('_', ' ').title()}**")
    st.markdown(f"`conf={state.confidence:.2f}`")
    
    cols = st.columns(3)
    cols[0].metric("Forward head", f"{state.ear_shoulder_offset_x:+.2f}")
    cols[1].metric("Shoulder roll", f"{state.shoulder_roll_z:+.2f}")
    cols[2].metric("Tilt", f"{state.shoulder_tilt_angle:+.1f}°")

with right:
    st.subheader("Chat with your coach")
    chat_area = st.container(height=600)
    user_input = st.chat_input("Ask about your posture...")

    with chat_area:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with chat_area:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.chat_worker.respond(user_input)
                st.write(result["text"])
                st.session_state.chat_history.append(("assistant", result["text"]))

# === Execution Control ===
if IS_CLOUD and ctx.state.playing:
    import time
    time.sleep(0.5)
    st.rerun()
elif not IS_CLOUD:
    import time
    time.sleep(0.2)
    st.rerun()