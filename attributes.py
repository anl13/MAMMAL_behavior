import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
import numpy as np 
import trimesh 
from pig_renderer.pig_render.Render import OBJ

rgb = cm.get_cmap(plt.get_cmap('jet'))(np.linspace(0.0,1.0,256))[:,:3]
bgr = rgb[:,(2,1,0)]
RGB_256_CM = (rgb.copy() * 255).astype(np.int)
BGR_256_CM = (bgr.copy() * 255).astype(np.int)  
color1 = np.loadtxt("colormaps/color1.txt") / 255
g_model = OBJ("PIG_model/other_data/reduced_pig.obj")
g_faces = []
for f in g_model.faces: 
    g_faces.append(f[0]) 
g_mesh = trimesh.Trimesh() 
g_mesh.faces = g_faces 

# nose: 262 
# tail end: 120 
def get_reduced_ids():
    data = np.loadtxt("PIG_model/other_data/reduced_ids.txt", dtype=np.int)
    return data 
def get_reduced_parts():
    data = np.loadtxt("PIG_model/other_data/reduced_parts.txt", dtype=np.int) 
    names = [
    "NOT_BODY", # 0
	"MAIN_BODY", # 1
	"HEAD",
	"L_EAR",
	"R_EAR",
	"L_F_LEG", # 5
	"R_F_LEG",
	"L_B_LEG",
	"R_B_LEG",
	"TAIL"
    ]
    return data, names

def get_body_ids(index):
    ids = get_reduced_ids()
    parts, _ = get_reduced_parts() 
    body = [] 
    for i in range(len(ids)):
        if parts[i] == index: 
            body.append(i) 
    return body 

g_head_ids = get_body_ids(2) 
g_body_ids = get_body_ids(1) 
g_left_front_leg_ids = get_body_ids(5)
g_right_front_leg_ids = get_body_ids(6)
g_left_hind_leg_ids = get_body_ids(7)
g_right_hind_leg_ids = get_body_ids(8)

g_noseid = 262
g_tailid = 120 
g_lefteyeid = 15 
g_righteyeid = 263