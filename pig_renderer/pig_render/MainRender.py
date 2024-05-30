import numpy as np
import cv2

import sys
import os 
from OpenGL.GL import *
import glfw
from glfw.GLFW import *
from .common import *
from .Render import *
from PIL import Image 
import glm 

class MainRender(): 
    def __init__(self, w=1920, h=1080, hide=False): 
        g_mouseState["window_h"] = h
        g_mouseState["window_w"] = w

        self.window = myGLInit(h, w, hide)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_dir = current_dir

        self.texShader = SimpleShader(current_dir + "/Render/shader/texture_v.shader", 
            current_dir + "/Render/shader/texture_f.shader")
        self.meshShader = SimpleShader(current_dir + "/Render/shader/mesh_v.shader", 
            current_dir + "/Render/shader/mesh_f.shader")
        self.colorShader = SimpleShader(current_dir + "/Render/shader/color_v.shader", 
            current_dir + "/Render/shader/color_f.shader")

        self.textShader = SimpleShader(current_dir + "/Render/shader/text_v.shader", 
            current_dir + "/Render/shader/text_f.shader")

        ## render targets 
        self.color_object_list = [] 
        self.tex_object_list = [] 
        self.mesh_object_list = [] 
        self.skel_object_list = [] 
        self.background = BGRender()
        self.background.load_img(cv2.imread(current_dir + "/data/white_tex.png"))
        
        self.text_render = TextRender() 
        self.text_render.initliaze() 

        # self.foreground = BGRender(False, 0.52, 1, -1, -0.73)
        self.foreground= BGRender(False, 0.5,1,0.5,1)
        ## render params 
        self.is_use_background_img = False 

    def set_background_img(self, img):
        self.background.load_img(img) 

    def build_floor(self, scale=1): 
        floormodel = OBJ(self.current_dir + "/data/obj_model/floor_z+_gray.obj")
        mesh = MeshRender() 
        mesh.load_data_basic(vertex = floormodel.vertices*scale, face = floormodel.faces_vert)
        mesh.load_data_colors(colors=floormodel.colors)
        mesh.bind_arrays()
        self.mesh_object_list.append(mesh)

    def build_scene(self, scale=1): 
        self.build_floor(scale=scale) 
        for k in range(5,7):
            part1obj = OBJ(self.current_dir + "/data/obj_pig/zhujuan_new_part{}.obj".format(k))
            mesh = ColorRender() 
            mesh.load_data_basic(vertex = part1obj.vertices*scale, face=part1obj.faces_vert)
            mesh.set_color(np.asarray([0.9,0.9,0.9], dtype=np.float32))
            mesh.bind_arrays() 
            self.color_object_list.append(mesh)

    def init_some_skels(self, num=1):
        for k in range(num): 
            skel_rend = SkelRender()
            skel_rend.bind_arrays() 
            self.skel_object_list.append(skel_rend)

    def draw(self): 
        g_renderCamera.configShader(self.texShader)
        g_renderCamera.configShader(self.meshShader)
        g_renderCamera.configShader(self.colorShader)
            
        for m in self.color_object_list: 
            m.draw(self.colorShader)
        
        for m in self.mesh_object_list: 
            m.draw(self.meshShader)
        
        for m in self.tex_object_list: 
            m.draw(self.texShader)

        for m in self.skel_object_list: 
            if m.points is not None: 
                m.draw(self.colorShader)

    def draw_bg(self):
        if self.is_use_background_img:
            self.background.draw() 


    # left bottom corner is pixel (0,0) for pos. 
    def add_bg_text(self, text, pos, scale, color): 
        color_cv = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        pos_cv = (pos[0], g_mouseState["window_h"]-pos[1])
        self.background.add_bg_text(pos_cv, text, color_cv, fontsize=scale)

    def add_fg_img(self, img): 
        self.foreground.load_img(img) 

    def draw_fg_text(self, text, pos, scale, color): 
        self.textShader.use()
        projection = np.asarray(glm.ortho(0, 1920, 0, 1080), dtype=np.float32)
        self.textShader.set_mat4f("projection", projection.T) 
        self.textShader.set_int("text", 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        c = np.asarray(color, dtype=np.float32) 
        self.textShader.set_vec3f("textColor", color) 
        self.text_render.render_text(self.textShader, text, pos[0], pos[1], scale)

    def set_bg_color(self, color): 
        
        glClearBufferfv(GL_COLOR, 0, (color[0], color[1], color[2], 1.0))

    def readImage(self):
        data = glReadPixels(0, 0, g_mouseState["window_w"], g_mouseState["window_h"], GL_BGRA, GL_FLOAT, outputType=None)
        data = data.reshape(g_mouseState["window_h"], g_mouseState["window_w"], -1)
        data = np.flip(data, 0)
        img = (255 * data[..., :3]).astype(np.uint8)
        return img
    
    ''' C++ get image offscreen 
    // register data 
    GLuint m_renderbuffers[2];
	GLuint m_framebuffer;
    std::vector<cudaGraphicsResource_t> m_cuda_gl_resources;
	m_cuda_gl_resources.resize(1); 
    cudaArray_t m_colorArray;
	cudaGraphicsGLRegisterImage(&m_cuda_gl_resources[0], m_renderbuffers[0],
		GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly);
    // render 
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer); 
    draw() // draw function. 
    cudaGraphicsMapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]); 
	cudaGraphicsSubResourceGetMappedArray(&m_colorArray, m_cuda_gl_resources[0], 0, 0);
    cudaMemcpy2DFromArray(m_device_renderData, WINDOW_WIDTH * sizeof(float4),
		m_colorArray, 0, 0, WINDOW_WIDTH * sizeof(float4), WINDOW_HEIGHT, cudaMemcpyDeviceToDevice);
	cv::Mat img = extract_bgr_mat(m_device_renderData, WINDOW_WIDTH, WINDOW_HEIGHT); 
    cudaGraphicsUnmapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); 

    '''
    ### fail to achieve what we want 
    # def readImageOffscreen(self, width, height):
    #     ## offscreen render data 
    #     fbWidth, fbHeight = width, height
    #     # Setup framebuffer
    #     framebuffer = glGenFramebuffers (1)
    #     glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    #     # Setup colorbuffer
    #     colorbuffer = glGenRenderbuffers (1)
    #     glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer)
    #     glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, fbWidth, fbHeight)
    #     glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorbuffer) 
    #     # Setup depthbuffer
    #     depthbuffer = glGenRenderbuffers (1)
    #     glBindRenderbuffer(GL_RENDERBUFFER,depthbuffer)
    #     glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, fbWidth, fbHeight)
    #     glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer)

    #     # check status
    #     status = glCheckFramebufferStatus (GL_FRAMEBUFFER)
    #     if status != GL_FRAMEBUFFER_COMPLETE:  
    #         print("error in framebuffer activation")
    #     else: 
    #         pass 
    #     glViewport(0,0,fbWidth, fbHeight) 
    
    #     self.draw() 

    #     glReadBuffer(GL_COLOR_ATTACHMENT0)
    #     glPixelStorei(GL_PACK_ALIGNMENT, 1)
    #     data = glReadPixels (0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    #     image = Image.new("RGB", (width, height), (0, 0, 0))
    #     image.frombytes(data)
    #     image = image.transpose(Image.FLIP_TOP_BOTTOM)

    #     glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE) 
    #     glViewport(0,0,g_mouseState["window_w"],g_mouseState["window_h"])
    #     return np.asarray(image) * 255

    def startLoop(self): 
        while (glfwGetKey(self.window, GLFW_KEY_ESCAPE) != GLFW_PRESS and glfwWindowShouldClose(self.window) == 0):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.draw() 

            if g_mouseState["saveimg"]: 
                img = self.readImage() 
                while os.path.exists("saveimg{}.jpg".format(g_mouseState["saveimg_index"])): 
                    g_mouseState["saveimg_index"] += 1
                cv2.imwrite("saveimg{}.jpg".format(g_mouseState["saveimg_index"]), img) 
                g_mouseState["saveimg"] = False 
                g_mouseState["saveimg_index"] += 1
            glfwSwapBuffers(self.window)
            glfwPollEvents()

        glfwTerminate()

if __name__ == "__main__":
    a = MainRender()