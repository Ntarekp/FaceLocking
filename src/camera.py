# src/camera.py
import cv2

def main():
    print("Testing camera indices...")
    working_index = -1
    
    # Try indices 0 to 3
    for idx in range(4):
        print(f"Trying index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS: Camera found at index {idx}")
                working_index = idx
                cap.release()
                break
            else:
                print(f"Index {idx} opened but failed to read frame.")
                cap.release()
        else:
            print(f"Index {idx} failed to open.")
            
    if working_index == -1:
        print("No working camera found.")
        return

    print(f"\nStarting camera test on index {working_index}...")
    cap = cv2.VideoCapture(working_index)
    
    print("Camera test. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        cv2.imshow(f"Camera Test (Index {working_index})", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
