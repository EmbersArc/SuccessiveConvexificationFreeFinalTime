import numpy as np
import os
import bpy

rck = bpy.data.objects["rck"]
eng = bpy.data.objects["eng"]
fir = bpy.data.objects["fir"]

folder_number = str(int(max(os.listdir('trajectory/final/')))).zfill(3)

X = np.load(open("trajectory/final/{}/X.npy".format(folder_number), "rb"))
U = np.load(open("trajectory/final/{}/U.npy".format(folder_number), "rb"))
sigma = np.load(open("trajectory/final/{}/sigma.npy".format(folder_number), "rb"))

FPS = int(sigma)  # frames per discretization step

bpy.data.scenes["Scene"].frame_current = 1

rck.animation_data_clear()
eng.animation_data_clear()
fir.animation_data_clear()

fir.hide_render = False
fir.keyframe_insert(data_path='hide_render')
fir.hide = False
fir.keyframe_insert(data_path='hide')

for i in range(X.shape[1]):
    x = X[:, i]
    u = U[:, i]

    rck.location = np.array((x[2], x[3], x[1])) / 100
    rck.rotation_quaternion = (x[7], x[9], x[10], x[8])
    
    ry = -np.arctan(u[2] / u[0])
    rx = np.arctan(u[1] / u[0])
    eng.rotation_euler = (rx, ry, 0)
    fir.scale[2] = np.linalg.norm(u) / 800000

    rck.keyframe_insert(data_path='location')
    rck.keyframe_insert(data_path='rotation_quaternion')
    eng.keyframe_insert(data_path='rotation_euler')
    fir.keyframe_insert(data_path='scale')

    bpy.data.scenes["Scene"].frame_current += FPS

fir.hide_render = True
fir.keyframe_insert(data_path='hide_render')
fir.hide = True
fir.keyframe_insert(data_path='hide')

bpy.data.scenes["Scene"].frame_current = 1
bpy.data.scenes["Scene"].frame_end = U.shape[1] * FPS + 60

