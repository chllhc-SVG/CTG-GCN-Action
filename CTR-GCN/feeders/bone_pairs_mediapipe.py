# MediaPipe 33-point bone pairs (v1, v2): v1 - v2
# Parent joints for bone computation, mirroring the NTU bone_pairs convention.
# Each pair defines: bone[v1] = joint[v1] - joint[v2]

mediapipe_pairs = (
    # Head → shoulders
    (0, 11), (0, 12),

    # Face
    (1, 0), (2, 1), (3, 2), (7, 3),
    (4, 0), (5, 4), (6, 5), (8, 6),
    (9, 0), (10, 0),

    # Torso
    (12, 11), (23, 11), (24, 12), (24, 23),

    # Left arm
    (13, 11), (15, 13),

    # Left hand
    (17, 15), (19, 15), (21, 15), (19, 17),

    # Right arm
    (14, 12), (16, 14),

    # Right hand
    (18, 16), (20, 16), (22, 16), (20, 18),

    # Left leg
    (25, 23), (27, 25),

    # Left foot
    (29, 27), (31, 27), (31, 29),

    # Right leg
    (26, 24), (28, 26),

    # Right foot
    (30, 28), (32, 28), (32, 30),
)
