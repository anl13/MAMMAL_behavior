from IPython.core.pylabtools import select_figure_formats
from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np
import os
import cv2 as cv

import sys
sys.path.append('../../')
from ..common.shader import loadShaders
from .RendUtils import * 

'''
Render Object with Texture 
'''
class TexRender:
    def __init__(self):
        self.VAO = glGenVertexArrays(1)
        self.VertexBuffer = glGenBuffers(1)
        self.UvBuffer = glGenBuffers(1)
        self.NormalBuffer = glGenBuffers(1)
        self.TextureBuffer = glGenTextures(1)

        self.TextureImg = np.ones((8, 8, 3), dtype='uint8')
        self.height = 8
        self.width = 8

        self.faceNum = 0
        self.vertexNum = 0 
        # render parameter 
        self.isFill = True 
        self.model = np.identity(4, np.float32) # model in MVP

    # def __del__(self):
    #     glDeleteBuffers(1, self.VertexBuffer)
    #     glDeleteBuffers(1, self.UvBuffer)
    #     glDeleteBuffers(1, self.NormalBuffer)
    #     glDeleteTextures(1, self.TextureBuffer)
    #     glDeleteVertexArrays(1, self.VAO)

    def load_data_basic(self, vertex, face):
        self.faceNum = np.size(face, 0)
        self.vertexNum = vertex.shape[0]
        
        self.face_vert = face 
        normal = compute_normal(vertex, face)
        self.vertex3 = vertex[face].astype('float32')
        self.normal3 = normal[face].astype('float32')

    def load_data_tex(self, uv=None, faceUV=None, TextureImg=None): 
        if uv is None or TextureImg is None or faceUV is None:
            self.TextureImg = np.ones((8, 8, 3), dtype='uint8') * 128
            self.height = 8
            self.width = 8
            uv = np.zeros((self.vertexNum, 2), dtype='float32')
            faceUV = self.face_vert
        else:
            self.TextureImg = cv.flip(TextureImg, 0)
            self.TextureImg = cv.resize(self.TextureImg, (1024,1024))
            self.height = np.size(self.TextureImg, 0)
            self.width = np.size(self.TextureImg, 1)
            
        self.uv3 = uv[faceUV].astype('float32')

    def bind_arrays(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex3.nbytes, self.vertex3, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.UvBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.uv3.nbytes, self.uv3, GL_DYNAMIC_DRAW)

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
        glBindBuffer(GL_ARRAY_BUFFER, self.UvBuffer)
        glVertexAttribPointer(
            1,  # attribute
            2,  # size
            GL_FLOAT,  # type
            GL_FALSE,  # normalized?
            0,  # stride
            None  # array buffer offset
        )

        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.NormalBuffer)
        glVertexAttribPointer(
            2,  # attribute
            3,  # size
            GL_FLOAT,  # type
            GL_FALSE,  # normalized?
            0,  # stride
            None  # array buffer offset
        )

        glBindVertexArray(0)

        

    def bind_texture(self):
        glBindVertexArray(self.VAO)
        glBindTexture(GL_TEXTURE_2D, self.TextureBuffer)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_BGR, GL_UNSIGNED_BYTE, self.TextureImg)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.TextureBuffer)
        glBindVertexArray(0)

    def draw(self, inShader):
        if self.isFill: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.bind_texture() 
        glBindVertexArray(self.VAO)
        inShader.use()
        inShader.set_int("object_texture", 1) 
        inShader.set_mat4f("model", self.model.transpose())
        glDrawArrays(GL_TRIANGLES, 0, 3 * self.faceNum)
        glUseProgram(0)
        glBindVertexArray(0)