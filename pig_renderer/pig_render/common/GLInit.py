from os import PathLike
from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np
import glfw 
from .Camera import Camera
import math 
from scipy.spatial.transform import Rotation 

g_renderCamera = Camera()

g_mouseState = { 
    "left": False, 
    "right": False, 
    "middle": False, 
    "leftClickTimeSeconds": 0, 
    "posBefore": np.asarray([0,0]), 
    "arcballRadius": 1.0,
    "window_change": False,
    "window_w": 1920, 
    "window_h": 1080,
    "sensitivity": 2.0,
    "saveimg" : False, 
    "saveimg_index": 0, 
    "pause" : False, 
    "key_left" : False, 
    "key_right" : False, 
    "key_up" : False, 
    "key_down" : False,
    "print_cam_pos": False
}

def getArcballCoordinate(plane_coord, front, up, right): 
    x = plane_coord[0]
    y = plane_coord[1] 
    z = 0
    r = x**2 + y**2
    if r > 1: 
        x = x / r 
        y = y / r 
        z = 0 
    else:
        z = math.sqrt(1 - r)
    return right * x + up * y + front * z

def keyboard_callback(window, key, scancode, action, mods): 
    if action == glfw.PRESS: 
        if chr(key) == "S":
            g_mouseState["saveimg"] = True
        if key == glfw.KEY_SPACE: 
            g_mouseState["pause"] = not g_mouseState["pause"]
    elif action == glfw.RELEASE: 
        pass 
    
    if key == glfw.KEY_LEFT and action==glfw.PRESS: 
        g_mouseState["key_left"] = True 
    if key == glfw.KEY_RIGHT and action==glfw.PRESS: 
        g_mouseState["key_right"] = True 
    if key == glfw.KEY_UP and action==glfw.PRESS: 
        g_mouseState["key_up"] = True 
    if key == glfw.KEY_DOWN and action==glfw.PRESS: 
        g_mouseState["key_down"] = True 

def mousebutton_callback(window, button, action, mods): 
    pos = glfwGetCursorPos(window) 
    global g_mouseState
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS: 
        g_mouseState["left"] = True 
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE: 
        g_mouseState["left"] = False 
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS: 
        g_mouseState['right'] = True 
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE: 
        g_mouseState['right'] = False 
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS: 
        g_mouseState['middle'] = True 
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE: 
        g_mouseState['middle'] = False 

    
def cursorpose_callback(window, xpos, ypos): 
    newpos = np.asarray([xpos, ypos], np.float32)
    if newpos[0] < 0 or newpos[0] > g_mouseState["window_w"] or newpos[0] < 0 or newpos[1] > g_mouseState["window_h"]: 
        return 
    if g_mouseState["left"] == False and g_mouseState["right"] == False and g_mouseState["middle"] == False: 
        g_mouseState["posBefore"] = newpos
        return 
    center = g_renderCamera.center.copy()
    up = g_renderCamera.up.copy() 
    pos = g_renderCamera.pos.copy() 
    front = g_renderCamera.front.copy() 
    right = g_renderCamera.right.copy()
    _newpos = newpos.copy() 
    _newpos[0] /= g_mouseState["window_w"]
    _newpos[1] /= g_mouseState["window_h"]
    nowArcBall = getArcballCoordinate(_newpos*2-np.ones(2,dtype=np.float32), front, up, right)
    _before_pos = g_mouseState["posBefore"].copy() 
    _before_pos[0] /= g_mouseState["window_w"]
    _before_pos[1] /= g_mouseState["window_h"]
    beforeArcBall = getArcballCoordinate(_before_pos*2-np.ones(2,dtype=np.float32), front, up, right)
    if g_mouseState["left"]: 
        theta = math.acos( beforeArcBall.dot(nowArcBall) )
        if theta < 0.00001: 
            rotation_axis = np.asarray([0,0,1],dtype=np.float32)
        else: 
            rotation_axis = np.cross(beforeArcBall, nowArcBall) 
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rot = Rotation.from_rotvec(rotation_axis * theta * g_mouseState["sensitivity"])
        rotmat = rot.as_matrix()
        newCamPos = rotmat.dot(pos - center) + center 
        g_renderCamera.computeExtrinsic(newCamPos, up, center)

        if g_mouseState["print_cam_pos"]:
            print("new cam pos : ", newCamPos)
            print("new cam up  : ", up) 
            print("new cam cen : ", center) 
    if g_mouseState["right"]: 
        dist = np.linalg.norm(pos - center)
        newCamCenter = center + dist * (nowArcBall - beforeArcBall) 
        g_renderCamera.computeExtrinsic(pos, up, newCamCenter) 
        if g_mouseState["print_cam_pos"]:
            print("new cam pos : ", pos)
            print("new cam up  : ", up) 
            print("new cam cen : ", newCamCenter) 
    g_mouseState["posBefore"] = newpos

def scroll_callback(window, xoffset, yoffset): 
    sense = 0.2
    newPos = g_renderCamera.pos - sense * yoffset * g_renderCamera.front
    if (newPos - g_renderCamera.center).dot(g_renderCamera.pos - g_renderCamera.center) > 0: 
        g_renderCamera.computeExtrinsic(newPos, g_renderCamera.up, g_renderCamera.center)
    if g_mouseState["print_cam_pos"]:
        print("new cam pos : ", newPos)
        print("new cam up  : ", g_renderCamera.up) 
        print("new cam cen : ", g_renderCamera.center) 

def windowresize_callback(window, w, h): 
    global g_mouseState
    g_mouseState["window_change"] = True 
    g_mouseState["window_w"] = w 
    g_mouseState["window_h"] = h

def myGLInit(Height, Width, hide=False):
    if not glfwInit():
        raise RuntimeError('Failed to initialize GLFW')

    glfwWindowHint(GLFW_SAMPLES, 4)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    if hide:
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE)
    # Open a window and create its OpenGL context
    window = glfwCreateWindow(Width, Height, "MyRender", None, None)
    if window is None:
        glfwTerminate()
        raise RuntimeError('window is None')
    glfwMakeContextCurrent(window)


    glfwSetKeyCallback(window, keyboard_callback)
    glfwSetCursorPosCallback(window, cursorpose_callback) 
    glfwSetMouseButtonCallback(window, mousebutton_callback)
    glfwSetScrollCallback(window, scroll_callback) 
    glfwSetWindowSizeCallback(window, windowresize_callback)

    glClearColor(0.0, 0.0, 0.0, 0.0)

    # Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE)

    # Enable depth test
    glEnable(GL_DEPTH_TEST)
    # Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS)

    # Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE)
    return window
