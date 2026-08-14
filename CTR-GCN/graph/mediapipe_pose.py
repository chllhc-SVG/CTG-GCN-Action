import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# MediaPipe Pose 33 关键点
# 0:nose 1:left_eye_inner 2:left_eye 3:left_eye_outer 4:right_eye_inner
# 5:right_eye 6:right_eye_outer 7:left_ear 8:right_ear
# 9:mouth_left 10:mouth_right
# 11:left_shoulder 12:right_shoulder
# 13:left_elbow 14:right_elbow
# 15:left_wrist 16:right_wrist
# 17:left_pinky 18:right_pinky
# 19:left_index 20:right_index
# 21:left_thumb 22:right_thumb
# 23:left_hip 24:right_hip
# 25:left_knee 26:right_knee
# 27:left_ankle 28:right_ankle
# 29:left_heel 30:right_heel
# 31:left_foot_index 32:right_foot_index

num_node = 33
self_link = [(i, i) for i in range(num_node)]

# inward: (child, parent)  从远端指向身体中心
# outward: (parent, child) 从身体中心指向远端
# 每条无向边都出现在 inward 和 outward 中

inward = [
    # 面部（从外围指向 nose）
    (1, 0), (2, 1), (3, 2), (7, 3),
    (4, 0), (5, 4), (6, 5), (8, 6),
    (10, 9),

    # nose → 左肩（虚拟边，连通头部与躯干）
    (0, 11),

    # 躯干
    (12, 11), (23, 11), (24, 12), (24, 23),

    # 左臂（从手腕指向肩膀）
    (13, 11), (15, 13),

    # 左手（从指尖指向手腕）
    (17, 15), (19, 15), (21, 15), (19, 17),

    # 右臂
    (14, 12), (16, 14),

    # 右手
    (18, 16), (20, 16), (22, 16), (20, 18),

    # 左腿（从脚踝指向髋）
    (25, 23), (27, 25),

    # 左脚
    (29, 27), (31, 27), (31, 29),

    # 右腿
    (26, 24), (28, 26),

    # 右脚
    (30, 28), (32, 28), (32, 30),
]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
