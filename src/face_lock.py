# Face Locking Feature
import time
import os
from collections import deque

class FaceLock:
    def __init__(self, identity, history_dir='data/db'):
        self.identity = identity
        self.locked = False
        self.last_seen = None
        self.locked_bbox = None
        self.locked_landmarks = None
        self.action_history = []
        self.history_dir = history_dir
        self.missing_frames = 0
        self.max_missing_frames = 60  # ~2 seconds at 30fps
        self.action_buffer = deque(maxlen=5)
        self.history_file = None
        self._init_history_file()

    def _init_history_file(self):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        fname = f"{self.identity.lower()}_history_{timestamp}.txt"
        self.history_file = os.path.join(self.history_dir, fname)
        with open(self.history_file, 'w') as f:
            f.write(f"Face Locking History for {self.identity}\n")

    def try_lock(self, recognized_id, bbox, landmarks, confidence, threshold=0.8):
        if not self.locked and recognized_id == self.identity and confidence >= threshold:
            self.locked = True
            self.locked_bbox = bbox
            self.locked_landmarks = landmarks
            self.last_seen = time.time()
            self.missing_frames = 0
            self.record_action('lock', f"Locked onto {self.identity}")
            return True
        return False

    def update(self, bbox, landmarks, recognized_id=None, confidence=None):
        if self.locked:
            self.locked_bbox = bbox
            self.locked_landmarks = landmarks
            self.last_seen = time.time()
            self.missing_frames = 0
            
            # Record position periodically (e.g., every 30 frames or 1 second)
            # For now, let's just record it if it moved significantly or just log it
            # But user asked for "where the user is detected to be found"
            # We can log the center of the bounding box
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self.record_action('position', f"User at ({cx:.1f}, {cy:.1f})")

        else:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing_frames:
                self.release_lock()

    def release_lock(self):
        if self.locked:
            self.locked = False
            self.locked_bbox = None
            self.locked_landmarks = None
            self.record_action('unlock', f"Released lock on {self.identity}")

    def detect_actions(self, prev_bbox, prev_landmarks, curr_bbox, curr_landmarks):
        actions = []
        # Face movement
        if prev_bbox and curr_bbox:
            dx = curr_bbox[0] - prev_bbox[0]
            if dx > 15:
                actions.append(('move_right', f"Face moved right by {dx:.1f} px"))
            elif dx < -15:
                actions.append(('move_left', f"Face moved left by {abs(dx):.1f} px"))
        
        # Landmarks for actions (assumes 12-point format: 
        # 0-3: Left Eye (L, R, T, B)
        # 4-7: Right Eye (L, R, T, B)
        # 8-11: Mouth (L, R, T, B)
        if prev_landmarks is not None and curr_landmarks is not None and len(curr_landmarks) >= 12:
            try:
                # Eye blink: measure EAR (Eye Aspect Ratio) approximation or just height
                # Left eye height
                le_h_prev = abs(prev_landmarks[2][1] - prev_landmarks[3][1])
                le_h_curr = abs(curr_landmarks[2][1] - curr_landmarks[3][1])
                
                # Right eye height
                re_h_prev = abs(prev_landmarks[6][1] - prev_landmarks[7][1])
                re_h_curr = abs(curr_landmarks[6][1] - curr_landmarks[7][1])

                # Thresholds for blink (eyes closing)
                # If eye height drops significantly
                if (le_h_prev > 5 and le_h_curr < 3) or (re_h_prev > 5 and re_h_curr < 3):
                     actions.append(('eye_blink', "Blink detected"))

                # Smile: Mouth width and corner lift
                # Mouth width
                mw_prev = abs(prev_landmarks[8][0] - prev_landmarks[9][0])
                mw_curr = abs(curr_landmarks[8][0] - curr_landmarks[9][0])
                
                # Mouth height (opening) can also indicate laugh
                mh_prev = abs(prev_landmarks[10][1] - prev_landmarks[11][1])
                mh_curr = abs(curr_landmarks[10][1] - curr_landmarks[11][1])

                # Smile typically widens mouth
                if mw_curr > mw_prev * 1.1:
                    actions.append(('smile', "Smile detected"))
                
                # Laugh often opens mouth wide
                if mh_curr > mh_prev * 1.5 and mh_curr > 10:
                    actions.append(('laugh', "Laugh detected"))

            except Exception as e:
                # print(f"Action detection error: {e}")
                pass
                
        for action, desc in actions:
            self.record_action(action, desc)
        return actions

    def record_action(self, action_type, description):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        record = f"{timestamp}\t{action_type}\t{description}\n"
        self.action_history.append(record)
        self.action_buffer.append((action_type, description))
        with open(self.history_file, 'a') as f:
            f.write(record)

    def is_locked(self):
        return self.locked

    def get_locked_identity(self):
        return self.identity if self.locked else None
