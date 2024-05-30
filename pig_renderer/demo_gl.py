
import numpy as np
import cv2

import sys
import os 
from OpenGL.GL import *
import glfw
from glfw.GLFW import *
from numpy.lib.twodim_base import fliplr
from pig_render.common import *
from pig_render.Render import *
from pig_render import MainRender
from pig_render.Render import OBJ
import glm 

def test_renderer():
    renderer = MainRender(1920,1080)

    window = renderer.window

    ## demo1: draw a textured hand 
    model = OBJ("pig_render/data/obj_model/hand.obj")   
    uvimage = cv2.imread("pig_render/data/chessboard_black_large.png")
    mesh = TexRender()
    mesh.load_data_basic(vertex=model.vertices + np.array([0,-0.3, 0.1]), face=model.faces_vert)
    mesh.load_data_tex(uv=model.texcoords, faceUV=model.faces_tex, TextureImg=uvimage)
    mesh.bind_arrays()
    renderer.tex_object_list.append(mesh)   

    ## demo2: draw a pig with single surface color 
    pig_model = OBJ("pig_render/data/obj_pig/PIG.obj")
    mesh2 = ColorRender()
    mesh2.load_data_basic(vertex=pig_model.vertices + np.array([0,0,0.21]), face = pig_model.faces_vert)
    mesh2.set_color(np.asarray([1,0.7,0.5], dtype=np.float32))
    mesh2.bind_arrays() 
    renderer.color_object_list.append(mesh2)

    ## demo3: draw a floor with per-vertex color
    floormodel = OBJ("pig_render/data/obj_model/floor_z+_gray.obj")
    mesh3 = MeshRender() 
    mesh3.load_data_basic(vertex = floormodel.vertices, face = floormodel.faces_vert)
    mesh3.load_data_colors(colors=floormodel.colors)
    mesh3.bind_arrays()
    renderer.mesh_object_list.append(mesh3) 

    ## demo 4: draw a cube skeleton 
    joints = np.asarray([
        [0,0,0],
        [1,0,0],
        [1,0,1],
        [0,0,1],
        [1,1,0],
        [1,1,1],
        [0,1,1],
        [0,1,0]
    ]) / 3
    bones = [ 
        [0,1],
        [1,2],
        [2,3],
        [3,0],
        [4,5],
        [5,6],
        [6,7],
        [7,4],
        [0,7],
        [1,4],
        [2,5],
        [3,6]
    ]
    skel = SkelRender(jointnum=joints.shape[0], topo = bones) 
    skel.isShowZeroPoint = True 
    skel.set_joint_position(joints) 
    skel.bind_arrays() 
    renderer.skel_object_list.append(skel) 

    ## demo5: draw a pig living cage
    renderer.build_scene() 

    renderer.is_use_background_img = False 
    g_mouseState["print_cam_pos"] = False 
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS and glfwWindowShouldClose(window) == 0):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        renderer.set_bg_color((1,1,1))
        renderer.draw() 
        renderer.draw_fg_text("This is a tiny_renderer. Copyright@LiangAn", (400, 480), 1, (0,0,0.8))

        if g_mouseState["saveimg"]: 
            img = renderer.readImage() 
            while os.path.exists("saveimg{}.jpg".format(g_mouseState["saveimg_index"])): 
                g_mouseState["saveimg_index"] += 1
            cv2.imwrite("saveimg{}.jpg".format(g_mouseState["saveimg_index"]), img) 
            g_mouseState["saveimg"] = False 
            g_mouseState["saveimg_index"] += 1
  
        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == '__main__':
    test_renderer()