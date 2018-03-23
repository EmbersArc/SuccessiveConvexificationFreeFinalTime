import numpy as np
import pickle
import os
import bpy

abspath = os.path.abspath(os.path.dirname(__file__))
dname = os.path.dirname(abspath)
os.chdir(dname)

rck = bpy.data.objects["rck"]
eng = bpy.data.objects["eng"]
fir = bpy.data.objects["fir"]

X = pickle.load(open("trajectory/X.p", "rb"))
U = pickle.load(open("trajectory/U.p", "rb"))
sigma = pickle.load(open("trajectory/sigma.p", "rb"))

FPS = int(sigma * 1.5) # frames per discretization step

bpy.data.scenes["Scene"].frame_current = 1

rck.animation_data_clear()
eng.animation_data_clear()
fir.animation_data_clear()

fir.hide_render = False
fir.keyframe_insert(data_path='hide_render')
fir.hide = False
fir.keyframe_insert(data_path='hide')

for i in range(len(X)):
    x = X[i, :]
    u = U[i, :]
    rck.location = np.array((x[2], x[3], x[1]))
    rck.rotation_quaternion = (x[7], x[10], x[9], x[8])

    throttle = np.linalg.norm(u)
    ry = -np.arctan(u[2] / u[0])
    rx = np.arctan(u[1] / u[0])
    eng.rotation_euler = (rx, ry, 0)
    fir.scale[2] = throttle

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
bpy.data.scenes["Scene"].frame_end = len(U) * FPS + 60

print("Final mass:", X[-1, 0])
