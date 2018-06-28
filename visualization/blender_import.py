import numpy as np
import os
import bpy

abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
os.chdir(dname)

rck = bpy.data.objects["rck"]
eng = bpy.data.objects["eng"]
fir = bpy.data.objects["fir"]

folder_number = str(int(max(os.listdir('trajectory/final/')))).zfill(3)

X = np.load(open("trajectory/final/{}/X.npy".format(folder_number), "rb"))
U = np.load(open("trajectory/final/{}/U.npy".format(folder_number), "rb"))
sigma = np.load(open("trajectory/final/{}/sigma.npy".format(folder_number), "rb"))

K = X.shape[1]
FPS = 60  # frames per discretization step
total_frames = FPS * sigma
step_size = int(total_frames / K)

bpy.data.scenes["Scene"].frame_current = 1

rck.animation_data_clear()
eng.animation_data_clear()
fir.animation_data_clear()
for l in range(4):
    name = "leg." + str(l).zfill(3)
    leg = bpy.data.objects[name]
    leg.animation_data_clear()

fir.hide_render = False
fir.keyframe_insert(data_path='hide_render')
fir.hide_viewport = False
fir.keyframe_insert(data_path='hide_viewport')

for i in range(K):
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

    if i == int(K / 3 * 2):
        for l in range(4):
            name = "leg." + str(l).zfill(3)
            leg = bpy.data.objects[name]
            leg.rotation_euler[0] = 0.
            leg.keyframe_insert(data_path='rotation_euler')

    if i == int(K / 5 * 4):
        for l in range(4):
            name = "leg." + str(l).zfill(3)
            leg = bpy.data.objects[name]
            leg.rotation_euler[0] = 130 / 180 * np.pi
            leg.keyframe_insert(data_path='rotation_euler')

    bpy.data.scenes["Scene"].frame_current += step_size

fir.hide_render = True
fir.keyframe_insert(data_path='hide_render')
fir.hide_viewport = True
fir.keyframe_insert(data_path='hide_viewport')

bpy.data.scenes["Scene"].frame_current = 1
bpy.data.scenes["Scene"].frame_end = K * step_size + FPS
