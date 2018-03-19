import numpy as np
import pickle
import os
import bpy


def create_ranges_nd(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return start[..., None] + steps[..., None] * np.arange(N)


os.chdir("./")

rck = bpy.data.objects["rck"]
eng = bpy.data.objects["eng"]
fir = bpy.data.objects["fir"]

X_in = pickle.load(open("X.p", "rb"))
U_in = pickle.load(open("U.p", "rb"))

FPS = 25  # not really FPS, but frames per discretization step

X = np.empty((len(X_in) * FPS, 14))
U = np.empty((len(U_in) * FPS, 3))

for i in range(len(X_in) - 1):
    X[i * FPS:(i + 1) * FPS, :] = create_ranges_nd(X_in[i, :], X_in[i + 1, :], FPS).T
    U[i * FPS:(i + 1) * FPS, :] = create_ranges_nd(U_in[i, :], U_in[i + 1, :], FPS).T

bpy.data.scenes["Scene"].frame_current = 1

rck.animation_data_clear()
eng.animation_data_clear()
fir.animation_data_clear()

for i in range(len(U) - FPS):
    x = X[i, :]
    u = U[i, :]
    rck.location = np.array((x[2], x[3], x[1]))
    rck.rotation_quaternion = (x[7], x[10], x[9], x[8])

    throttle = np.linalg.norm(u)
    ry = -np.arctan(u[1] / u[0])
    rx = np.arctan(u[2] / u[0])
    eng.rotation_euler = (rx, ry, 0)
    fir.scale[2] = throttle * 2.3

    rck.keyframe_insert(data_path='location')
    rck.keyframe_insert(data_path='rotation_quaternion')
    eng.keyframe_insert(data_path='rotation_euler')
    fir.keyframe_insert(data_path='scale')
    bpy.data.scenes["Scene"].frame_current += 1

bpy.data.scenes["Scene"].frame_current = 1
bpy.data.scenes["Scene"].frame_end = len(U) + 2 * FPS

print("Final mass:", X_in[-1, 0] / 10)
