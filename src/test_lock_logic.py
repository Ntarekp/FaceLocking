
import time
import os
import shutil
import numpy as np
from src.face_lock import FaceLock

def test_face_locking_logic():
    print("=== Testing Face Locking Logic ===")
    
    # 1. Setup
    test_identity = "TestUser"
    test_history_dir = "data/test_db"
    if os.path.exists(test_history_dir):
        shutil.rmtree(test_history_dir)
    os.makedirs(test_history_dir, exist_ok=True)
    
    fl = FaceLock(test_identity, history_dir=test_history_dir)
    
    print(f"[1] Initialized FaceLock for '{test_identity}'")
    assert not fl.is_locked(), "Should not be locked initially"

    # 2. Test Locking
    # Simulate a recognized face with high confidence
    bbox_start = (100, 100, 200, 200)
    # Dummy landmarks (needs to be capable of action detection later, but for lock just needs to exist)
    # 12 points: LE(4), RE(4), Mouth(4)
    # Just random points for now
    kps_dummy = np.zeros((12, 2), dtype=np.float32)
    
    # Try lock with low confidence
    fl.try_lock(test_identity, bbox_start, kps_dummy, confidence=0.5, threshold=0.8)
    assert not fl.is_locked(), "Should not lock with low confidence"
    print("[2] Low confidence lock check passed (not locked)")
    
    # Try lock with wrong identity
    fl.try_lock("OtherUser", bbox_start, kps_dummy, confidence=0.9, threshold=0.8)
    assert not fl.is_locked(), "Should not lock on wrong identity"
    print("[3] Wrong identity lock check passed (not locked)")
    
    # Try lock with correct identity and high confidence
    success = fl.try_lock(test_identity, bbox_start, kps_dummy, confidence=0.9, threshold=0.8)
    assert success, "Should return True on successful lock"
    assert fl.is_locked(), "Should be locked now"
    print("[4] Successful lock check passed")

    # 3. Test Action Detection
    print("\n[5] Testing Action Detection...")
    
    # Helper to generate dummy landmarks
    def make_kps(eye_h=10, mouth_w=20, mouth_h=5):
        # 0-3: Left Eye (L, R, T, B)
        le = [[10, 10], [20, 10], [15, 10-eye_h/2], [15, 10+eye_h/2]]
        # 4-7: Right Eye (L, R, T, B)
        re = [[40, 10], [50, 10], [45, 10-eye_h/2], [45, 10+eye_h/2]]
        # 8-11: Mouth (L, R, T, B)
        mo = [[20, 40], [20+mouth_w, 40], [20+mouth_w/2, 40-mouth_h/2], [20+mouth_w/2, 40+mouth_h/2]]
        return np.array(le + re + mo, dtype=np.float32)

    # Initial state
    prev_bbox = (100, 100, 200, 200)
    prev_kps = make_kps(eye_h=6, mouth_w=20) # Open eyes, normal mouth

    # A. Move Right
    curr_bbox = (130, 100, 230, 200) # Moved +30px
    curr_kps = make_kps(eye_h=6, mouth_w=20)
    
    actions = fl.detect_actions(prev_bbox, prev_kps, curr_bbox, curr_kps)
    print(f"   -> Move Right Actions: {actions}")
    assert any(a[0] == 'move_right' for a in actions), "Should detect move_right"
    
    # Update prev
    prev_bbox = curr_bbox
    prev_kps = curr_kps

    # B. Blink (Eye height goes from 6 to 2)
    curr_bbox = (130, 100, 230, 200)
    curr_kps = make_kps(eye_h=2, mouth_w=20) # Closed eyes
    
    actions = fl.detect_actions(prev_bbox, prev_kps, curr_bbox, curr_kps)
    print(f"   -> Blink Actions: {actions}")
    assert any(a[0] == 'eye_blink' for a in actions), "Should detect eye_blink"

    # Update prev
    prev_bbox = curr_bbox
    prev_kps = curr_kps
    
    # C. Smile (Mouth width goes from 20 to 24 (1.2x))
    curr_bbox = (130, 100, 230, 200)
    curr_kps = make_kps(eye_h=6, mouth_w=24) # Wide mouth
    
    actions = fl.detect_actions(prev_bbox, prev_kps, curr_bbox, curr_kps)
    print(f"   -> Smile Actions: {actions}")
    assert any(a[0] == 'smile' for a in actions), "Should detect smile"

    # 4. Test History File
    print("\n[6] Checking History File...")
    files = os.listdir(test_history_dir)
    history_files = [f for f in files if f.endswith('.txt') and test_identity.lower() in f]
    assert len(history_files) == 1, f"Should have exactly 1 history file, found {len(history_files)}"
    
    history_path = os.path.join(test_history_dir, history_files[0])
    with open(history_path, 'r') as f:
        content = f.read()
        print(f"   History File Content:\n---\n{content}---\n")
        assert "lock" in content
        assert "move_right" in content
        assert "eye_blink" in content
        assert "smile" in content
    
    print("=== Verification Successful! Logic is sound. ===")

if __name__ == "__main__":
    test_face_locking_logic()
