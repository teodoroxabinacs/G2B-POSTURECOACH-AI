# demo_state.py
class DemoState:
    posture_class = "slouching"
    confidence = 0.87
    is_reliable = True
    posture_duration_sec = 0.0
    ear_shoulder_offset_x = -0.20
    shoulder_roll_z = 0.78
    shoulder_tilt_angle = 3.0
    session_duration_sec = 0.0
    correction_events = 0
    longest_bad_posture_streak_sec = 0.0
    posture_distribution = {"correct_posture": 0.0, "slouching": 1.0, "neck_forward": 0.0, "lean": 0.0}
    feature_deviations = {"forward_head": 0.0, "shoulder_roll": 0.78, "shoulder_tilt": 0.0}

def get_demo_state():
    return DemoState()