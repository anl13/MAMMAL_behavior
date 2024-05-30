from IPython.core.pylabtools import select_figure_formats
from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np
import cv2 as cv
import os

import sys
sys.path.append('../../')
from ..common.shader import SimpleShader
from .RendUtils import * 

'''
Render object with mono color only
'''
class ColorRender:
    def __init__(self):
        self.VAO = glGenVertexArrays(1)
        self.VertexBuffer = glGenBuffers(1)
        self.NormalBuffer = glGenBuffers(1)

        self.faceNum = 0
        self.vertexNum = 0 
        self.color = np.zeros(3, dtype=np.float32)
        # render parameter 
        self.isFill = True 
        self.model = np.identity(4, np.float32) # model in MVP

    # def __del__(self):
    #     glDeleteBuffers(1, self.VertexBuffer)
    #     glDeleteBuffers(1, self.NormalBuffer)
    #     glDeleteVertexArrays(1, self.VAO)

    def load_data_basic(self, vertex, face):
        self.faceNum = np.size(face, 0)
        self.vertexNum = vertex.shape[0]
        
        self.face_vert = face 
        normal = compute_normal(vertex, face)
        self.vertex3 = vertex[face].astype('float32')
        self.normal3 = normal[face].astype('float32') 

    def bind_arrays(self):
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex3.nbytes, self.vertex3, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.normal3.nbytes, self.normal3, GL_DYNAMIC_DRAW)
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

    def set_color(self, color):
        self.color = color 

    def draw(self, inShader):
        if self.isFill: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBindVertexArray(self.VAO)
        inShader.use()
        inShader.set_vec3f("object_color", self.color)
        inShader.set_mat4f("model", self.model.transpose())
        glDrawArrays(GL_TRIANGLES, 0, 3 * self.faceNum)
        glUseProgram(0)
        glBindVertexArray(0)


