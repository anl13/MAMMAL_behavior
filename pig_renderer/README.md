# Introduction 
This is a tiny renderder used in paper _Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL_. 
This renderer could render textured mesh, monocolor mesh, vertex-colored mesh, and provides a solution for skeleton rendering. 
This project is modified from https://github.com/Dw1010/pyRender. 

## Usage 
After cloning the repository, put `pig_render/` to your own workspace, and follow `demo_gl.py` to see how to use the functions. 

## Dependencies 
You should install glfw, OpenGL, glm by 
```shell
pip install PyOpenGL
pip install pyGLM
pip install glfw
pip install freetype-py 
```
Note that, you should `pip install pyGLM` for `glm` lib not `pip install glm`. 

## Citation
If you use these datasets in your research, please cite the paper

```BibTex
@article{MAMMAL, 
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    booktitle = {},
    month = {July},
    year = {2022}
}
```
