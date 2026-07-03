"""G2B Posture Correction Coach — live Streamlit app.

Previous (v2) Streamlit UI is preserved in app_v2_backup.py.
"""
import os
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer

DEMO_MODE = os.environ.get("G2B_DEMO_MODE", "0") == "1"

from src.workers.shared_state import SharedPostureState
from src.workers.chat_worker import ChatWorker

if not DEMO_MODE:
    from src.workers.cv_worker import CVWorker
    from src.utils.config import mediapipe_complexity, target_fps, camera_resolution

st.set_page_config(page_title="G2B Posture Coach", layout="wide")

st.markdown("# G2B Posture Correction Coach")
st.caption("Live posture analysis + AI coaching — built for Raspberry Pi 5")

# === Init singletons (survive across reruns) ===
if "shared" not in st.session_state:
    st.session_state.shared = SharedPostureState()
    st.session_state.chat_worker = ChatWorker(st.session_state.shared)
    st.session_state.chat_history = []

    if not DEMO_MODE:
        try:
            w, h = camera_resolution()
            st.session_state.cv_worker = CVWorker(
                st.session_state.shared,
                width=w, height=h,
                model_complexity=mediapipe_complexity(),
                target_fps=target_fps(),
            )
            # Try to start physical camera (fails gracefully on Streamlit Cloud)
            st.session_state.cv_worker.start()
        except Exception:
            pass
    else:
        from demo_state import get_demo_state
        st.session_state.shared.update(get_demo_state())
        st.info("📷 Demo mode — webcam disabled. Posture is simulated as 'Slouching'.")

# === Session summary bar ===
summary_slot = st.empty()

# === Layout: two columns ===
left, right = st.columns([1, 1])

with left:
    st.subheader("Live view")
    if DEMO_MODE:
        st.info("📷 No webcam in demo mode. Chat with the coach on the right →")
    
    # 1. Native WebRTC component completely replaces the old frame_slot logic
    ctx = webrtc_streamer(
        key="g2b-posture-coach-stream",
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
    state_slot = st.empty()
    metrics_slot = st.empty()

with right:
    st.subheader("Chat with your coach")
    chat_area = st.container()
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

# === Extract and process frame from WebRTC browser feed ===
if ctx and ctx.video_receiver:
    try:
        frame = ctx.video_receiver.get_frame(timeout=1).to_ndarray(format="bgr24")
        if frame is not None and "cv_worker" in st.session_state:
            # If your cv_worker has a process method, you can pass the browser frame in here!
            # st.session_state.cv_worker.process_frame(frame)
            pass
    except Exception:
        pass

# === Continuously refresh UI Metrics (No loops!) ===
LABEL_EMOJI = {
    "correct_posture": "🟢",
    "slouching":       "🟠",
    "neck_forward":    "🟡",
    "lean":            "🟣",
}

state = st.session_state.shared.snapshot()
if state is not None:
    emoji = LABEL_EMOJI.get(state.posture_class, "")
    
    # Clear the markdown text slot and rewrite it fresh
    state_slot.empty()
    state_slot.markdown(
        f"### {emoji} **{state.posture_class.replace('_', ' ').title()}**  "
        f"`conf={state.confidence:.2f}`  "
        f"_(holding for {state.posture_duration_sec:.0f}s)_"
    )
    
    # Clear the metrics slot and rewrite the columns fresh
    metrics_slot.empty()
    with metrics_slot.container():
        cols = st.columns(3)
        cols[0].metric("Forward head", f"{state.ear_shoulder_offset_x:+.2f}",
                       delta=f"{state.feature_deviations['forward_head']:.2f}")
        cols[1].metric("Shoulder roll", f"{state.shoulder_roll_z:+.2f}",
                       delta=f"{state.feature_deviations['shoulder_roll']:.2f}")
        cols[2].metric("Tilt", f"{state.shoulder_tilt_angle:+.1f}°",
                       delta=f"{state.feature_deviations['shoulder_tilt']:.1f}")

    # Clear the summary slot and rewrite the summary columns fresh
    summary_slot.empty()
    with summary_slot.container():
        scols = st.columns(4)
        scols[0].metric("Session", f"{state.session_duration_sec/60:.1f} min")
        scols[1].metric("Correct posture",
                        f"{state.posture_distribution.get('correct_posture', 0)*100:.0f}%")
        scols[2].metric("Corrections", state.correction_events)
        scols[3].metric("Worst streak", f"{state.longest_bad_posture_streak_sec:.0f}s")

# Pace the reruns slightly so the WebRTC stream stays stable
time.sleep(0.5)
st.rerun()