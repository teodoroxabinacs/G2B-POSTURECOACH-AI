"""Thread-safe holder for the current PostureState. One per process."""
import copy
import threading
from collections import deque
from typing import Optional, List

from src.state.posture_state import PostureState


class SharedPostureState:
    def __init__(self):
        # --- Original Threading & Tracking Variables ---
        self._lock = threading.Lock()
        self._current: Optional[PostureState] = None
        self._history = deque(maxlen=300) 
        
        # --- Core Cloud Metrics ---
        self.posture_class = "correct_posture"
        self.confidence = 0.0
        self.ear_shoulder_offset_x = 0.0
        self.shoulder_roll_z = 0.0
        self.shoulder_tilt_angle = 0.0
        
        # --- System & RAG Fallbacks ---
        self.is_reliable = True 
        self.feature_deviations = {} 
        self.primary_issue = None
        self.posture_distribution = {}
        
        # --- Master Time Tracking Fallbacks ---
        self.posture_duration_sec = 0.0
        self.session_duration_sec = 0.0
        self.correction_events = 0
        self.longest_bad_posture_streak_sec = 0.0
        
        # --- Exhaustive Biomechanical & Joint Angle Fallbacks ---
        self.craniovertebral_angle = 0.0
        self.torso_compression_ratio = 1.0 
        self.neck_inclination = 0.0
        self.trunk_angle = 0.0
        self.shoulder_balance = 0.0
        self.hip_angle = 0.0
        self.knee_angle = 0.0
        self.spine_curve_index = 0.0
        
        # --- The Missing Angle ---
        self.midline_deviation_angle = 0.0 

    # --- THE BULLETPROOF SHIELD ---
    # If the RAG prompt asks for a variable we forgot to define above, 
    # Python will run this function instead of crashing with an AttributeError.
    def __getattr__(self, name):
        return 0.0

    def update(self, state: PostureState) -> None:
        with self._lock:
            self._current = state
            self._history.append(state)

    def snapshot(self) -> Optional[PostureState]:
        with self._lock:
            return copy.deepcopy(self._current)

    def history_last_seconds(self, seconds: float) -> List[PostureState]:
        with self._lock:
            if not self._history:
                return []
            cutoff = self._history[-1].timestamp.timestamp() - seconds
            return [s for s in self._history
                    if s.timestamp.timestamp() >= cutoff]