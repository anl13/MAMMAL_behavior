from IPython.core.pylabtools import select_figure_formats
from OpenGL.GL import *
import numpy as np
import os
import cv2 as cv

import sys
from ..common import *
from . import * 
import freetype 

class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')

def _get_rendering_buffer(xpos, ypos, w, h, zfix=0.0):
    # return np.asarray([
    #     xpos,     ypos - h, 0, 0,
    #     xpos,     ypos,     0, 1,
    #     xpos + w, ypos,     1, 1,
    #     xpos,     ypos - h, 0, 0,
    #     xpos + w, ypos,     1, 1,
    #     xpos + w, ypos - h, 1, 0
    # ], np.float32)
    return np.asarray([
        xpos,     ypos + h, 0, 0,
        xpos,     ypos,     0, 1,
        xpos + w, ypos,     1, 1,
        xpos,     ypos + h, 0, 0,
        xpos + w, ypos,     1, 1,
        xpos + w, ypos + h, 1, 0
    ], np.float32)

class TextRender(object): 
    def __init__(self): 
        self.VBO = None 
        self.VAO = None 
        self.Characters = dict() 
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.text_shader = SimpleShader(self.current_dir + "/shader/text_v.shader", self.current_dir + "/Render/shader/text_f.shader")
        self.fontfile = self.current_dir + "/../data/arial.ttf"
        print(self.fontfile)
        self.face = freetype.Face(self.fontfile)
        self.face.set_char_size( 48*64 )

        self.initliaze()

    def initliaze(self):
        #disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        #load first 128 characters of ASCII set
        for i in range(0,128):
            self.face.load_char(chr(i))
            glyph = self.face.glyph
            #generate texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, glyph.bitmap.width, glyph.bitmap.rows, 0,
                        GL_RED, GL_UNSIGNED_BYTE, glyph.bitmap.buffer)

            #texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            #now store character for later use
            self.Characters[chr(i)] = CharacterSlot(texture,glyph)
            
        glBindTexture(GL_TEXTURE_2D, 0)

        #configure VAO/VBO for texture quads
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
    def render_text(self,inShader,text,x,y,scale):
        inShader.use()
        glActiveTexture(GL_TEXTURE0)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.VAO)
        for c in text:
            ch = self.Characters[c]
            w, h = ch.textureSize
            w = w*scale
            h = h*scale
            local_x = x 
            local_y = y 

            tunning = 0 # my personal tune for symbols
            # if c=='-':
            #     tunning = -4 
            # if c=='*': 
            #     tunning = 8
            # if c=='|' or c=='(' or c==')': 
            #     tunning = -4
            
            local_x = x + ch.bearing[0] * scale 
            local_y = y - (h- ch.bearing[1] + tunning) * scale
   
            vertices = _get_rendering_buffer(local_x,local_y,w,h)

            #render glyph texture over quad
            glBindTexture(GL_TEXTURE_2D, ch.texture)
            #update content of VBO memory
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            #render quad
            glDrawArrays(GL_TRIANGLES, 0, 6)
            #now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (ch.advance>>6)*scale

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)