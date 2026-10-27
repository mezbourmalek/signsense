import sys, platform

print(f"Python {sys.version.split()[0]} sur {platform.system()}")

def check(nom, test):
    try:
        print(f"[OK]  {nom} : {test()}")
    except Exception as e:
        print(f"[ECHEC] {nom} : {e}")

def numpy_pandas():
    import numpy, pandas, pyarrow
    return f"numpy {numpy.__version__}, pandas {pandas.__version__}"

def sklearn_():
    import sklearn
    return sklearn.__version__

def mediapipe_():
    import mediapipe as mp
    with mp.solutions.holistic.Holistic():
        pass
    return f"mediapipe {mp.__version__}, Holistic chargé"

def webcam():
    import cv2
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("caméra illisible (essaie l'index 1, ferme Zoom/Teams)")
    return f"image {frame.shape[1]}x{frame.shape[0]}"

check("NumPy / Pandas / PyArrow", numpy_pandas)
check("scikit-learn", sklearn_)
check("MediaPipe", mediapipe_)
check("Webcam", webcam)