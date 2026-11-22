import time
import cv2
import numpy as np
from collections import deque

LABELS = ["Landfill", "Recycle", "Compost"]

# place holder for model prediction
def predict(image): 
    p = np.random.rand(3)
    p[1] += 0.3 # mock prediction
    return p / p.sum()

def preprocess(crop):
    color = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(color, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized 


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try: 
        prev = time.perf_counter()
        fps = 0

        history = deque(maxlen = 8)

        while True:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            ok, frame = cap.read()
            if not ok:
                break

            # center box
            h, w = frame.shape[:2]
            size = 300

            # top left
            x1 = (w - size) // 2  
            y1 = (h - size) // 2 
            
            # bottom right
            x2 = x1 + size 
            y2 = y1 + size

            # crop before drawing box to avoid drawing on the crop
            crop = frame[y1:y2, x1:x2].copy()

            # mock prediction
            probs = predict(preprocess(crop))
            history.append(probs)
            avg_probs = np.mean(history, axis=0)


            i = np.argmax(avg_probs)
            text = f"{LABELS[i]} {avg_probs[i]*100:.0f}%"

            # Draw box, label, FPS
            fps = 1 / dt
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Waste Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()