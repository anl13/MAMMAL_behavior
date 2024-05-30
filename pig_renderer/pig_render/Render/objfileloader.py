# objfileloader compatible with numpy 

# modified from  OBJFileLoader 
# see link: https://www.pygame.org/wiki/OBJFileLoader
# An Liang 2018 Jan 28th 

# Last modified by AN Liang 2021.12.06
# Feature: Add vertex color split. 

import numpy as np 
import os 
def MTL(filename):
    contents = {}
    mtl = None
    if not os.path.exists(filename): 
        print("warning: no mtl file ", filename, " was found. ") 
        return contents
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.faces_vert = [] 
        self.faces_tex = [] 
        self.colors = [] 
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = np.asarray( list(map(float, values[1:4])) ) 
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
                if len(values) == 7: 
                    c = np.asarray(list(map(float, values[4:7])) )
                    self.colors.append(c) 

            elif values[0] == 'vn':
                v = np.asarray( list(map(float, values[1:4])) ) 
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(np.asarray( list(map(float, values[1:3]))) ) 
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]) - 1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]) - 1)
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]) - 1)
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
                self.faces_vert.append(face) 
                self.faces_tex.append(texcoords)

        self.vertices = np.asarray(self.vertices) 
        self.normals = np.asarray(self.normals) 
        self.texcoords = np.asarray(self.texcoords) 
        self.faces_vert = np.asarray(self.faces_vert, np.int) 
        self.faces_tex  = np.asarray(self.faces_tex, np.int)
        self.colors = np.asarray(self.colors) / 255

    
if __name__ == "__main__": 
    smpl = OBJ("reduced_pig.obj")
    from IPython import embed; embed() 