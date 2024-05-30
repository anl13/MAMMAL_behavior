from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np
import cv2 as cv
import os
import sys

sys.path.append('../../')
from .RendUtils import * 
from .ColorRender import ColorRender 
from .objfileloader import OBJ

class SkelRender:
    def __init__(self, jointnum=19, topo=[]): 
        # GL buffer 
        self.VAO = glGenVertexArrays(1)
        self.VertexBuffer = glGenBuffers(1)
        self.NormalBuffer = glGenBuffers(1)
        self.VAO_2 = glGenVertexArrays(1) 
        self.VertexBuffer_2 = glGenBuffers(1) 
        self.NormalBuffer_2 = glGenBuffers(1) 

        # basic param 
        self.jointnum = jointnum 
        self.topo = np.asarray(topo)
        self.bone_num = self.topo.shape[0] 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ballobj = OBJ(current_dir + "/../data/obj_model/ball.obj")
        self.stickobj = OBJ(current_dir + "/../data/obj_model/stick.obj")
        self.cubeobj = OBJ(current_dir + "/../data/obj_model/cube.obj")

        self.stick_size = 0.005 
        self.ball_size = 0.01

        self.cube_v = self.cubeobj.vertices * self.ball_size 
        self.cube_n = compute_normal(self.cube_v, self.cubeobj.faces_vert) 
        self.cube_v3 = self.cube_v[self.cubeobj.faces_vert].astype('float32')
        self.cube_n3 = self.cube_n[self.cubeobj.faces_vert].astype('float32')
        self.cube_face_num = self.cubeobj.faces_vert.shape[0]

        self.ball_v = self.ballobj.vertices * self.ball_size 
        self.stick_v = self.stickobj.vertices * self.stick_size
        self.ball_n = compute_normal(self.ball_v, self.ballobj.faces_vert)
        self.stick_n = compute_normal(self.stick_v, self.stickobj.faces_vert) 

        self.ball_v3 = self.ball_v[self.ballobj.faces_vert].astype('float32')
        self.stick_v3 = self.stick_v[self.stickobj.faces_vert].astype('float32')
        self.ball_n3 = self.ball_n[self.ballobj.faces_vert].astype('float32')
        self.stick_n3 = self.stick_n[self.stickobj.faces_vert].astype('float32')
        self.ball_face_num = self.ballobj.faces_vert.shape[0] 
        self.stick_face_num = self.stickobj.faces_vert.shape[0] 

        self.mono_joint_color = np.asarray([1,0,0], np.float32)
        self.mono_bone_color = np.asarray([0,1,0], np.float32)
        self.per_joint_color = [] 
        self.per_bone_color = [] 
        self.per_joint_size = [] # TODO: write render code for it .
        self.per_bone_size = [] 

        self.isFill = True 
        self.isPerJointColor = False 
        self.points = None 
        self.isShowZeroPoint = False # TODO 
        self.isUseCube = False 

    def set_ball_size(self, _bonesize): 
        self.ball_size = _bonesize 
        self.ball_v = self.ballobj.vertices * _bonesize
        self.ball_v3 = self.ball_v[self.ballobj.faces_vert].astype('float32')

        self.cube_v = self.cubeobj.vertices * self.ball_size 
        self.cube_v3 = self.cube_v[self.cubeobj.faces_vert].astype('float32')

    def set_stick_size(self, _sticksize): 
        self.stick_size = _sticksize
        self.stick_v = self.stickobj.vertices * _sticksize 
        self.stick_v3 = self.stick_v[self.stickobj.faces_vert].astype('float32') 
    def set_ball_size_per_joint(self, _ballsizes): 
        self.per_joint_size = _ballsizes 
        
    def set_stick_size_per_bone(self, _sticksizes): 
        self.per_bone_size = _sticksizes 

    def bind_arrays(self):
        if self.isUseCube:
            glBindVertexArray(self.VAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.cube_v3.nbytes, self.cube_v3, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.cube_n3.nbytes, self.cube_n3, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
            glVertexAttribPointer(
                0,  # attribute
                3,  # size
                GL_FLOAT,  # type
                GL_FALSE,  # normalized?
                0,  # stride
                None  # array buffer offset
            )
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
            glVertexAttribPointer(
                1,  # attribute
                3,  # size
                GL_FLOAT,  # type
                GL_FALSE,  # normalized?
                0,  # stride
                None  # array buffer offset
            )
            glBindVertexArray(0)
        else: 
            glBindVertexArray(self.VAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.ball_v3.nbytes, self.ball_v3, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.ball_n3.nbytes, self.ball_n3, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
            glVertexAttribPointer(
                0,  # attribute
                3,  # size
                GL_FLOAT,  # type
                GL_FALSE,  # normalized?
                0,  # stride
                None  # array buffer offset
            )
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
            glVertexAttribPointer(
                1,  # attribute
                3,  # size
                GL_FLOAT,  # type
                GL_FALSE,  # normalized?
                0,  # stride
                None  # array buffer offset
            )
            glBindVertexArray(0)

        glBindVertexArray(self.VAO_2)
        glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer_2)
        glBufferData(GL_ARRAY_BUFFER, self.stick_v3.nbytes, self.stick_v3, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer_2)
        glBufferData(GL_ARRAY_BUFFER, self.stick_n3.nbytes, self.stick_n3, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer_2)
        glVertexAttribPointer(
            0,  # attribute
            3,  # size
            GL_FLOAT,  # type
            GL_FALSE,  # normalized?
            0,  # stride
            None  # array buffer offset
        )
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer_2)
        glVertexAttribPointer(
            1,  # attribute
            3,  # size
            GL_FLOAT,  # type
            GL_FALSE,  # normalized?
            0,  # stride
            None  # array buffer offset
        )
        glBindVertexArray(0)

    def _compute_rot(self, direct):
        a = np.asarray([0,0,1])
        d = direct.copy() 
        if np.linalg.norm(d) == 0: 
            return np.eye(3)
        d = d / np.linalg.norm(d) 
        b = np.cross(a,d)
        if np.linalg.norm(b) == 0: 
            return np.eye(3)
        b = b / np.linalg.norm(b) 
        r = b * np.math.acos(d[2])
        R = cv.Rodrigues(r)[0]
        return R 

    def set_joint_position(self, points): 
        # assert points.shape[0] == self.jointnum, "require jointnum=19, while get {}".format(points.shape[0])
        self.points = points # [19,3]
        if self.bone_num == 0:
            return  
        self.bones = self.points[self.topo] # [19,2,3]
        self.bone_lengths = np.linalg.norm(self.bones[:,0,:] - self.bones[:,1,:], axis=1)
        self.bone_trans = self.bones.mean(axis=1)
        self.bone_direct = self.bones[:,0,:] - self.bones[:,1,:]
        self.bone_rots = np.zeros([self.bone_num,3,3])
        for k in range(self.bone_num):
            a,b = self.topo[k]
            if (np.linalg.norm(self.points[a]) == 0 or np.linalg.norm(self.points[b] == 0)) and not self.isShowZeroPoint: 
                self.bone_lengths[k] = 0 
            else: 
                self.bone_rots[k] = self._compute_rot(self.bone_direct[k])

    def set_joint_bone_color(self, jointcolor, bonecolor): 
        self.mono_bone_color = bonecolor 
        self.mono_joint_color = jointcolor

    def set_joint_bone_color_list(self, jointcolorlist, bonecolorlist): 
        self.per_bone_color = bonecolorlist
        self.per_joint_color = jointcolorlist

    def draw(self, inShader):  # must be colorShader
        if self.isFill: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        for k in range(self.points.shape[0]): 
            if (np.abs(self.points[k,0]) + np.abs(self.points[k,1]) == 0) and not self.isShowZeroPoint: 
                continue 
            glBindVertexArray(self.VAO)
            inShader.use()
            if self.isPerJointColor: 
                inShader.set_vec3f("object_color", self.per_joint_color[k])
            else:
                inShader.set_vec3f("object_color", self.mono_joint_color)
            model = np.identity(4, np.float32)
            model[0:3,3] = self.points[k]
            inShader.set_mat4f("model", model.transpose())
            if self.isUseCube: 
                glDrawArrays(GL_TRIANGLES, 0, 3 * self.cube_face_num)
            else:
                glDrawArrays(GL_TRIANGLES, 0, 3 * self.ball_face_num)
            glUseProgram(0)
            glBindVertexArray(0)
        for k in range(self.bone_num): 
            if self.bone_lengths[k] == 0: 
                continue
            glBindVertexArray(self.VAO_2)
            inShader.use()
            if self.isPerJointColor: 
                inShader.set_vec3f("object_color", self.per_bone_color[k])
            else:
                inShader.set_vec3f("object_color", self.mono_bone_color)
            model = np.identity(4, np.float32)
            model[0:3,0:3] = self.bone_rots[k].astype('float32')
            model[0:3,3] = self.bone_trans[k]
            
            model[0:3,2] *= self.bone_lengths[k] / 2 / self.stick_size
            inShader.set_mat4f("model", model.transpose())
            glDrawArrays(GL_TRIANGLES, 0, 3 * self.stick_face_num)
            glUseProgram(0)
            glBindVertexArray(0)