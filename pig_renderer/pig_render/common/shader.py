from OpenGL.GL import *
import numpy as np 
from OpenGL.GL import glCreateShader

def compile_shader(shader_file_path, shader_type):
    ShaderId = glCreateShader(shader_type)
    shaderCode = ''
    with open(shader_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            shaderCode = shaderCode + line
    # print(shaderCode)
    # print(type(shaderCode))
    # Compile Shader
    print('Compiling shader : {}'.format(shader_file_path))
    glShaderSource(ShaderId, shaderCode)
    glCompileShader(ShaderId)

    # Check Shader
    # Result = glGetShaderiv(ShaderId, GL_COMPILE_STATUS)
    InfoLogLength = glGetShaderiv(ShaderId, GL_INFO_LOG_LENGTH)
    if (InfoLogLength > 0):
        VertexShaderErrorMessage = glGetShaderInfoLog(ShaderId)
        print(VertexShaderErrorMessage)
        raise SyntaxError('error from shader: {}'.format(shader_file_path))
    return ShaderId


def loadShaders(vertex_file_path, fragment_file_path):
    VertexShaderID = compile_shader(vertex_file_path, GL_VERTEX_SHADER)
    FragmentShaderID = compile_shader(fragment_file_path, GL_FRAGMENT_SHADER)

    # Link the program
    print('Linking program')
    ProgramID = glCreateProgram()
    glAttachShader(ProgramID, VertexShaderID)
    glAttachShader(ProgramID, FragmentShaderID)
    glLinkProgram(ProgramID)

    # Check the program
    # Result = glGetProgramiv(ProgramID, GL_LINK_STATUS)
    InfoLogLength = glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH)
    if (InfoLogLength > 0):
        ProgramErrorMessage = glGetProgramInfoLog(ProgramID)
        print(ProgramErrorMessage)
        raise RuntimeError('program error')

    glDetachShader(ProgramID, VertexShaderID)
    glDetachShader(ProgramID, FragmentShaderID)

    glDeleteShader(VertexShaderID)
    glDeleteShader(FragmentShaderID)

    return ProgramID


class SimpleShader(object): 
    def __init__(self, vertexShaderPath, fragmentShaderPath, tcsPath=None, tesPath=None, geometryShaderPath=None): 
        self.programID = 0 
        self.programID = loadShaders(vertexShaderPath, fragmentShaderPath)

    def use(self): 
        glUseProgram(self.programID) 

    def set_bool(self, name, value): 
        glUniform1i(glGetUniformLocation(self.programID, name),int(value))
    
    def set_int(self, name, value): 
        glUniform1i(glGetUniformLocation(self.programID, name),int(value))
    
    def set_float(self, name, value): 
        glUniform1f(glGetUniformLocation(self.programID, name), float(value))

    def set_vec2f(self, name, value): 
        glUniform2fv(glGetUniformLocation(self.programID, name), 
            1, value)

    def set_vec2f_xy(self, name, x,y):
        glUniform2f(glGetUniformLocation(self.programID, name), 
            float(x), float(y))

    def set_vec3f(self, name, value): 
        glUniform3fv(glGetUniformLocation(self.programID, name), 
            1, value)

    def set_vec3f_xyz(self, name, x,y,z): 
        glUniform3f(glGetUniformLocation(self.programID, name), 
            float(x), float(y), float(z))

    
    def set_vec4f(self, name, value): 
        glUniform4fv(glGetUniformLocation(self.programID, name), 
            1, value)

    def set_vec4f_xyzw(self, name, x,y,z,w): 
        glUniform4f(glGetUniformLocation(self.programID, name), 
            float(x), float(y), float(z), float(w))

    def set_mat2f(self, name, value): 
        glUniformMatrix2fv(glGetUniformLocation(self.programID, name), 
            1, GL_FALSE, value)

    def set_mat3f(self, name, value): 
        glUniformMatrix3fv(glGetUniformLocation(self.programID, name), 
            1, GL_FALSE, value)
    
    def set_mat4f(self, name, value): 
        assert value.shape==(4,4) and value.dtype==np.float32, \
            "require shape (4,4) get {}, require type np.float32 get {}".format(value.shape, value.dtype)
        glUniformMatrix4fv(glGetUniformLocation(self.programID, name), 
            1, GL_FALSE, value)
