import numpy as np

# --- Extraction unique des landmarks MediaPipe ---
def extract_landmark_vector(results):
    """
    Extrait les landmarks MediaPipe Holistic d'un objet results en un vecteur numpy de shape (1662,)
    (33*4 pose + 468*3 face + 21*3 main gauche + 21*3 main droite)
    """
    # Pose: 33 points (x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in getattr(results, 'pose_landmarks', None).landmark]) if getattr(results, 'pose_landmarks', None) else np.zeros(33*4)
    # Face: 468 points (x, y, z)
    face = np.array([[res.x, res.y, res.z] for res in getattr(results, 'face_landmarks', None).landmark]) if getattr(results, 'face_landmarks', None) else np.zeros(468*3)
    # Main gauche: 21 points (x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in getattr(results, 'left_hand_landmarks', None).landmark]) if getattr(results, 'left_hand_landmarks', None) else np.zeros(21*3)
    # Main droite: 21 points (x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in getattr(results, 'right_hand_landmarks', None).landmark]) if getattr(results, 'right_hand_landmarks', None) else np.zeros(21*3)
    # Concaténation
    vec = np.concatenate([pose, face, lh, rh])
    assert vec.shape == (1662,), f"Landmark vector shape mismatch: {vec.shape} (should be 1662)"
    return vec 