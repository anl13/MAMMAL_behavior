# MAMMAL_behavior
This project shows how the behavior analysis was performed in the [MAMMAL](https://github.com/anl13/MAMMAL_core). 
It was tested on Windows 10 with anaconda 4.12.0, python 3.7.9. 

## Create Environment
1. Install anaconda. 
2. Create virtual environment, as 
```shell
conda create -n MAMMAL1 python=3.7.9
conda activate MAMMAL1
```
3. Install packages. 
```shell
pip install -r requirements.txt 
```

## Prepare Data
Download `data.zip` from [Google Drive](https://drive.google.com/file/d/1IsMo9StrDKh2gbrn0J8YRFQKKZCiFpbG/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1dYr_5_QLpYqomlW0ApQ-vA) (extract code: jvc9) and unzip it under the main folder as  
```
`-- MAMMAL_behavior
    |-- data/
    |-- fonts/
    |-- nm_results/
    |-- ...
```
The `data` folder contains necessary processed data to run the code. 

Download pig\_render from https://github.com/anl13/pig_renderer. Put it to current folder like 
```
`-- MAMMAL_behavior
    |-- data/
    |-- fonts/
    |-- nm_results/
    |-- pig_render/
        |-- Render/
        |-- MainRender.py
        |-- ...
    |-- ...
```

## Animal-Scene Interaction Measurement
```shell
python nm_drink.py
```
This file read data from `data/drink_data/` folder, and write figures to `nm_results/scene_behavior/` folder. To help understand, we also provide motion capture result videos in `nm_results/scene_behaivor/` used for eat and drink judgement. 

## Behavior Space Discovery
```shell
python nm_cluster.py
```
This file read behavior data from `data/clips/` folder (44 clips in total) and cluster them using tsne. It is worth mentioning that, due to the randomness of tsne, we may not get the exactly same results through two runs. However, you can still get some meaningful results from this file. 

This file may take about 4 minutes to perform the data processing and clustering. After finishing the process, it will write middle data and density images to `nm_results/individual_behavior/` folder, and present a renderer to show cluster peaks. Ususally, with the pre-defined parameters, it will generate 38~44 cluster areas after watershed algorithm, and the pose at the density peak of each area will be shown. You can use LEFT or RIGHT buttom to browse them. The eight distinct postures are manually determined during the browsing. It may not be the best clustering result due to the insufficiency of data used, but the results are still meanningful. Just enjoy browsing them! 

## Social Behavior Recognition
```
python nm_social.py
```
This file will read `data/batch5_nm/` data sequence (1000 frames) and write video and figure result to `nm_results/social_behavior/`. Please check the code for more information. 

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