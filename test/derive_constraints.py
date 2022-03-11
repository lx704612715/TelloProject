import numpy as np
import simpy
import pytransform3d
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def revolute_generator(n_point=100, length=90):
    position = np.random.randint(0, 10, 2)
    # create points on the x planer
    revolute_tra = np.ones([n_point, 3])
    revolute_tra[:, 0] = 0
    theta = np.pi * np.random.rand() - 2 * np.pi
    r = 0.1 * np.random.rand() + 0.03
    d_theta = length / 180 * np.pi / 100
    for i in range(n_point):
        curt_theta = theta + i * d_theta
        # position = [y, z]
        revolute_tra[i, 1] = position[0] + r * np.cos(curt_theta)
        revolute_tra[i, 2] = position[1] + r * np.sin(curt_theta)

    return revolute_tra, [0, position[0], position[1], r]


def prismatic_generator(n_point=100, length=0.04):
    position = np.random.randint(0, 10, 3)
    theta = np.pi * np.random.rand() - 2 * np.pi
    phi = np.pi * np.random.rand()
    dx = np.sin(phi) * np.cos(theta)
    dy = np.sin(phi) * np.sin(theta)
    dz = np.cos(phi)

    step_length = length / n_point
    trajectory = np.zeros([n_point, 3])
    for i in range(n_point):
        trajectory[i, 0] = position[0] + i * step_length * dx
        trajectory[i, 1] = position[1] + i * step_length * dy
        trajectory[i, 2] = position[2] + i * step_length * dz

    # return x, y, z and dx, dy, dz 
    return trajectory, [position[0], position[1], position[2], dx, dy, dz]


pris_tra, pri_const = prismatic_generator()
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(pris_tra)
first_component = pca.components_[0]
print("variance is: ", pca.explained_variance_ratio_[0])

rev_tra, rev_const = revolute_generator()

center_rev = np.sum(rev_tra, axis=0) / rev_tra.shape[0]
u, s, vh = np.linalg.svd(rev_tra - center_rev)
norm_vec = vh[2, :]#
a, b, c = norm_vec
d = -1 * np.matmul(center_rev.T, norm_vec)

def residualsCircle(parameters, dataPoint):
    cen_y, cen_z, Ri = parameters
    cen_x = (-d - b*cen_y - c*cen_z)/a
    distance = [np.linalg.norm(np.array([cen_x, cen_y, cen_z]) - np.array([x, y, z])) for x, y, z in dataPoint]
    res = [(Ri-dist) for dist in distance]
    return res

estimateCircle = [center_rev[1], center_rev[2], 0.01] # center y and center z
bestCircleFitValues, ier = leastsq(residualsCircle, estimateCircle, args=(rev_tra))
rev_center_y, rev_center_z, rev_r = bestCircleFitValues[0], bestCircleFitValues[1], bestCircleFitValues[2]
rev_center_x = (-d-b*rev_center_y-c*rev_center_z)/a

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot3D(rev_tra[:, 0], rev_tra[:, 1], rev_tra[:, 2])
ax.scatter(rev_center_x, rev_center_y, rev_center_z, c="r", marker="o")
plt.show()

print("debug")
