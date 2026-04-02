import numpy as np

# 径向反转（radial inversion）
def radial_inversion(pc):
    c = pc.mean(0)
    rel = pc - c
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    rmax = r.max()
    rel_unit = rel / (r + 1e-9)
    r_new = rmax - r
    return c + rel_unit * r_new

# 质心镜像（centroid reflection）
def centroid_reflection(pc):  # pc: Nx3 array
    c = pc.mean(0)
    return c - (pc - c)

# PCA 翻转
def pca_flip(pc):
    c = pc.mean(0)
    U, S, Vt = np.linalg.svd((pc - c).T)
    u = U[:,0]  # first principal axis
    return pc - 2 * np.outer(((pc - c) @ u), u)  # reflect along u
