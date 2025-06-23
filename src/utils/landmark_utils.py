import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Extraction unique des landmarks MediaPipe ---
def extract_landmark_vector(results):
    """
    Extrait les landmarks MediaPipe Holistic d'un objet results en un vecteur numpy de shape (1692,)
    (33*4 pose + 478*3 face [refine_landmarks=True] + 21*3 main gauche + 21*3 main droite)
    """
    # Pose: 33 points (x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in getattr(results, 'pose_landmarks', None).landmark]).flatten() if getattr(results, 'pose_landmarks', None) else np.zeros(33*4)
    # Face: 478 points (x, y, z) -- refine_face_landmarks=True
    face_landmarks = getattr(results, 'face_landmarks', None)
    expected_n_face_points = 478
    if face_landmarks:
        n_face_points = len(face_landmarks.landmark)
        assert n_face_points == expected_n_face_points, f"Face landmarks count mismatch: {n_face_points} (should be {expected_n_face_points})"
        face = np.array([[res.x, res.y, res.z] for res in face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(expected_n_face_points * 3)
    # Main gauche: 21 points (x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in getattr(results, 'left_hand_landmarks', None).landmark]).flatten() if getattr(results, 'left_hand_landmarks', None) else np.zeros(21*3)
    # Main droite: 21 points (x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in getattr(results, 'right_hand_landmarks', None).landmark]).flatten() if getattr(results, 'right_hand_landmarks', None) else np.zeros(21*3)
    # Logs debug
    logger.info(f"[refine_landmarks=True] Pose shape: {pose.shape}")
    logger.info(f"[refine_landmarks=True] Face shape: {face.shape}")
    logger.info(f"LH shape: {lh.shape}")
    logger.info(f"RH shape: {rh.shape}")
    vec = np.concatenate([pose, face, lh, rh])
    logger.info(f"Vecteur final shape: {vec.shape} (attendu: {33*4 + expected_n_face_points*3 + 21*3 + 21*3})")
    expected_shape = 33*4 + expected_n_face_points*3 + 21*3 + 21*3
    assert vec.shape == (expected_shape,), f"Landmark vector shape mismatch: {vec.shape} (should be {expected_shape})"
    return vec 