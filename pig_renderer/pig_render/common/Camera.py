from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np
import glfw 
import math 

g_NEAR_PLANE = 0.1 
g_FAR_PLANE = 10

def lookAt(in_pos, in_center, in_up): 
    direct = in_pos - in_center 
    direct = direct / np.linalg.norm(direct) 
    right = np.cross(in_up,direct) 
    right = right / np.linalg.norm(right) 
    up = np.cross(direct, right) 
    up = up / np.linalg.norm(up) 
    mat = np.identity(4, dtype='float32')
    R = np.asarray([right, up, direct])
    mat[0:3,0:3] = R 
    mat[0:3,3] = -R@in_pos
    return mat

def perspective(fovy, aspect, zNear, zFar):
    mat = np.zeros((4,4))
    tangent = math.tan(0.5*fovy)
    mat[0,0] = 1 / (tangent * aspect)
    mat[1,1] = 1 / tangent
    mat[2,2] = (zNear + zFar) / (zNear - zFar) 
    mat[3,2] = -1 
    mat[2,3] = 2 * zFar * zNear / (zNear - zFar)

class Camera(object): 
    def __init__(self):
        self.viewMat = np.identity(4, np.float32)
        self.projectionMat = np.identity(4, np.float32) 

        # set init params 
        self.CameraIntrinsic = np.asarray([
            [1340.0378,  0., 964.7579],
            [0., 1342.6888, 521.4926],
            [0.,        0.,      1] 
        ])
        _pos = np.asarray([-1, 1.5, 0.8], dtype=np.float32)
        _up = np.asarray([0, 0, 1], dtype=np.float32)
        _center= np.zeros(3, dtype=np.float32)

        self.pos = _pos.copy()
        self.up = _up.copy() 
        self.center = _center.copy() 
        self.front = np.zeros(3, dtype=np.float32)
        self.right = np.zeros(3, dtype=np.float32)
        self.setIntrinsic(self.CameraIntrinsic, 1920, 1080)
        self.computeExtrinsic(_pos, _up, _center)

    def computeExtrinsic(self, pos, up, center):
        self.pos = pos.copy()
        self.up = up.copy()
        self.center = center.copy()   
        front = pos - center
        self.front = front/ np.linalg.norm(front) 
        right = np.cross(front,up)
        self.right = right / np.linalg.norm(right)
        up = np.cross(self.right, self.front) 
        self.up = up / np.linalg.norm(up) 
        self.viewMat = lookAt(pos, center, self.up) 

    def setExtrinsic(self, _R, _T): 
        R = _R.copy()
        T = _T.copy() 
        front = -R[2,:].squeeze() 
        up = -R[1,:].squeeze() 
        pos = -R.T@T 
        center = pos - front 
        self.computeExtrinsic(pos, up, center) 

    def setIntrinsic(self, K, width, height): 
        fx = K[0,0] / width 
        fy = K[1,1] / height 
        cx = K[0,2] / width 
        cy = K[1,2] / height 

        self.projectionMat = np.zeros([4,4], np.float32)
        self.projectionMat[0,0] = 2 * fx 
        self.projectionMat[0,2] = -(2*cx-1)
        self.projectionMat[1,1] = 2 * fy 
        self.projectionMat[1,2] = 2 * cy - 1 
        self.projectionMat[2,2] = (g_NEAR_PLANE + g_FAR_PLANE)/(g_NEAR_PLANE-g_FAR_PLANE)
        self.projectionMat[2,3] = 2*g_FAR_PLANE*g_NEAR_PLANE/(g_NEAR_PLANE-g_FAR_PLANE)
        self.projectionMat[3,2] = -1 

    def setIntrinsicFov(self, fovy, aspect, zNear=g_NEAR_PLANE, zFar = g_FAR_PLANE): 
        self.projectionMat = perspective(fovy, aspect, zNear, zFar)

    def configShader(self, inShader): 
        glUseProgram(inShader.programID) 
        #  transpose is necessary here. 
        inShader.set_mat4f("view", self.viewMat.transpose())
        inShader.set_mat4f("projection", self.projectionMat.transpose())
        inShader.set_vec3f("view_pos", self.pos)
        glUseProgram(0) 