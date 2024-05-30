from asyncore import write
from IPython.terminal.embed import embed
import numpy as np
from numpy.lib.function_base import interp 
import scipy.cluster.vq as vq 
import matplotlib.pyplot as plt 
import pylab
from scipy.ndimage.measurements import label
from scipy.spatial import distance
from skimage import feature 
from sklearn import manifold, datasets 
import pickle
from sklearn import neighbors
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity 
from bodymodel_np import BodyModelNumpy
import cv2 
from attributes import * 
from tqdm import tqdm 
import matplotlib as mpl 
from sklearn.decomposition import PCA 
import pandas as pd 
from sklearn import manifold 
import mpl_toolkits.axisartist as axisartist
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D
from sklearn.model_selection import GridSearchCV 
from scipy import ndimage as ndi 
from skimage.segmentation import watershed 
from skimage.feature import peak_local_max
from OpenGL.GL import *
import glfw
from glfw.GLFW import *
from numpy.lib.twodim_base import fliplr
from pig_renderer.pig_render.common import *
from pig_renderer.pig_render.Render import *
from pig_renderer.pig_render import MainRender
from pig_renderer.pig_render.Render import OBJ
from bodymodel_np import BodyModelNumpy
from time import time
from time import sleep
import os 
import json 

class CluterEngine(object): 
    def __init__(self, result_folder = None):
        self.bm = BodyModelNumpy() 
        self.visualize_gt_label = False 
        if result_folder is not None: 
            if not os.path.exists(result_folder): 
                os.makedirs(result_folder) 
        self.result_folder = result_folder

    def read_clip(self, clip_index): 
        self.clip_index = clip_index 
        data_file = "data/clips/{}/data.pkl".format(clip_index) 
        with open(data_file, 'rb') as f: 
            self.data = pickle.load(f) 

        
    def load_all_clips(self):
        all_features_type2 = [] 
        all_features_index = [] 
        self.all_data = [] 

        for clip_index in range(44):
            data_path = "data/clips/{}/data.pkl".format(clip_index)
            with open(data_path, "rb") as f: 
                data = pickle.load(f) 
            self.all_data.append(data) 
            feature_path = "data/clips/{}/features_type2.pkl".format(clip_index) 
            with open(feature_path, 'rb') as f: 
                features_type2 = pickle.load(f) 
            all_features_type2.append(features_type2)
            feature_index = np.zeros([features_type2.shape[0],2])
            feature_index[:,0] = clip_index 
            feature_index[:,1] = np.arange(features_type2.shape[0])
            all_features_index.append(feature_index)
        self.features_type2 = np.concatenate(all_features_type2, axis=0)
        self.features_index = np.concatenate(all_features_index, axis=0) 
        with open(self.result_folder + "/clip_features_type2.pkl" , 'wb') as f: 
            pickle.dump(self.features_type2, f) # [4*N, featureN ]
        with open(self.result_folder + "/clip_feature_index.pkl", 'wb') as f: 
            pickle.dump(self.features_index, f) 
        with open(self.result_folder + "/all_data.pkl", 'wb') as f: 
            pickle.dump(self.all_data, f) 
        print("feature type2 size : ", self.features_type2.shape)
    
    def build_feature_type2(self):
        joint_speed_sample = [15,16,17,7,8,9,56,57,58,40,41,42,2] # 13 
        surface_speed_sample = [ 
            g_noseid,g_tailid,g_lefteyeid, g_righteyeid # 4
        ]
        deformed_rot_ids = [0, 2, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23] # 22 
        selected_joints = [15,16,17,7,8,9,56,57,58,40,41,42]
        features = [] 
        N = self.data["joints62"].shape[0] 
        for k in tqdm(range(N)): 
            J_origin = self.data["joints62"][k]
            poseparam = self.data["rots"][k].copy()
            t = self.data["trans"][k].copy()
            s = self.data["scale"]
            R0,_ = cv2.Rodrigues(np.asarray([0,0,poseparam[0,0]]))
            T0 = np.asarray([t[0],t[1],0])
            invR = R0.transpose() 
            invT = -invR.dot(T0) 
            t[0:2] = 0 
            t[2] = t[2] / s 
            poseparam[0,0] = 0 
            J_norm = (J_origin.dot(invR.T) + invT) / s 
            S_origin = self.data["surface_points"][k]
            S_norm = (S_origin.dot(invR.T) + invT) / s
            S_vvec_origin = self.data["surface_speed_vec"][k]
            S_vvec_norm = (S_vvec_origin@invR.T) / s 
            J_vvec_origin = self.data["joint62_speeds"][k]
            J_vvec_norm = (J_vvec_origin@invR.T) / s
            skel = self.data["skel"][k].copy() 
            skel_norm = (skel.dot(invR.T) + invT) / s 
            aa = self.data["rots"][k].copy()

            root_height = J_norm[0,2] # (1) height of body root 
            body_height = J_norm[2,2] # (1) height of body center 
            head_vec = S_norm[g_noseid] - J_norm[23]
            head_vec = head_vec / np.linalg.norm(head_vec) # (3) head direct, normalized vector 
            body_angle = self.data["body_angle"][k].copy() # (1) body angle 
            body_angle = body_angle / 60 
            x_rot = poseparam[0,2] / np.math.pi * 2 # (1) body roll. use to decern left/right lay 

            ## dynamic state 
            selected_rots = aa[deformed_rot_ids] / np.math.pi *2 
            selected_j_v = J_vvec_norm[joint_speed_sample]
            selected_s_v = S_vvec_norm[surface_speed_sample]
            selected_j_v[selected_j_v < 0.03] = 0 
            selected_s_v[selected_s_v < 0.03] = 0

            F =  selected_j_v.ravel().tolist() \
                + selected_s_v.ravel().tolist() \
                + [root_height, body_height,body_angle,x_rot] \
                + skel_norm.ravel().tolist() \
                + selected_rots.ravel().tolist()
            F = np.asarray(F) 
            features.append(F)
        self.features_type2 = np.asarray(features)
        print(self.features_type2.shape)

        data_folder = "data/clips/{}/".format(self.clip_index)
        with open(data_folder + "/features_type2.pkl", 'wb') as f: 
            pickle.dump(self.features_type2, f) 

    def feature_pca(self, n=16, feature_name="clip_features"): 
        with open(self.result_folder + "/{}.pkl".format(feature_name), 'rb') as f: 
            features = pickle.load(f) 
        pca = PCA(n_components=n)
        pca.fit(features)
        print("pca explained variance ratio: ", pca.explained_variance_ratio_)
        ratio = np.asarray(pca.explained_variance_ratio_)
        print("pca sum: ", ratio.sum())
        self.transformed_data = pca.transform(features)
        with open(self.result_folder + "/feature_pca.pkl", 'wb') as f:
            pickle.dump(self.transformed_data, f) 
        # from IPython import embed; embed()

        
    def cluster(self, type="tsne", n_neigh=50): 
        with open(self.result_folder + "/feature_pca.pkl", 'rb') as f:
            self.transformed_data = pickle.load(f) 
        data = self.transformed_data
        a = time() 

        if type=="umap":
            embedding = umap.UMAP(n_neighbors=n_neigh, min_dist=0.1,n_components=2,
                metric="euclidean").fit_transform(data)
        elif type == "tsne":
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=None,
                perplexity=n_neigh,
                metric="euclidean")
            embedding = tsne.fit_transform(data) 
        else: 
            print("please choose tsne or umap!")
            exit()

        b = time() 
        print("time elapsed: ", b-a, " seconds")

        with open(self.result_folder + "/embedding.pkl", 'wb') as f: 
            pickle.dump(embedding, f)
        fig = plt.figure(figsize=(5,5)) 
        ax = fig.gca()
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False) 
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        ax.scatter(embedding[:,0], embedding[:,1], s=0.1)
        plt.savefig(self.result_folder + "/" + type + ".png", dpi=1000, bbox_inches="tight")

    def density(self, bandwidth=0.02):
        with open(self.result_folder + "/embedding.pkl", 'rb') as f: 
            embedding = pickle.load(f) 
        # normalize embedding 
        M = embedding.max(axis=0)
        m = embedding.min(axis=0) 
        embedding = (embedding - m) / (M-m) * 0.9 + 0.05

        a =  time()
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(embedding)
        b = time()
        print("finish fitting, elapse ", b-a, " seconds")
        

        xx, yy = np.mgrid[0:1:0.005, 0:1:0.005]
        xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
        z = np.exp(kde.score_samples(xy_sample))
        zz = np.reshape(z, xx.shape)
        # zz = zz / embedding.shape[0]
        c = time() 
        print(np.argmax(zz))
        print(zz.max())
        print("sample elapse ", c-b, " seconds")
        plt.cla()
        fig = plt.figure(1, figsize=(5,5)) 
        
        zz_max = zz.max() 
        zz_min = zz_max * 0.05
        zz_thresh = zz_max
        zz[zz > zz_thresh] = zz_thresh 
        zz[zz < zz_min] = zz_min
        zz = (zz - zz_min) / (zz_thresh - zz_min)
        scale = 255
        zz = zz * scale 
        with open(self.result_folder + "/kde-density.pkl", 'wb') as f: 
            pickle.dump((zz.T)[::-1,:],f) 
        zz = zz.astype(np.uint8) 
        H,W = zz.shape

        zz = zz.T
        zz = cv2.flip(zz, flipCode=0)
        cv2.imwrite(self.result_folder + "/kde-density.png", zz)

        colormap_jet = np.loadtxt("colormaps/jet.txt").astype(np.uint8)
        colormap_jet = colormap_jet[:,(2,1,0)]
        zz = zz.reshape([-1])
        zz_jet = colormap_jet[zz].reshape([H,W,3])

        cv2.imwrite(self.result_folder + "/kde-jet.png", zz_jet)
        # from IPython import embed; embed() 

        ## change to white background and 804 * 804 size for paper. 
        zz_jet_expand = np.zeros([804,804,3],dtype=np.uint8) + 255 
        zz_jet_copy = zz_jet.copy() 
        zz = zz.reshape([200,200])
        zz_jet_copy[zz == 0] = np.asarray([255,255,255],dtype=np.uint8) 

        zz_jet_expand[2:802,2:802,:] = cv2.resize(zz_jet_copy, (800,800)) 
        cv2.imwrite(self.result_folder + "/kde-jet-expand.png", zz_jet_expand) 

    
    def watershed(self):
        zz = cv2.imread(self.result_folder + "/kde-density.png",cv2.IMREAD_GRAYSCALE) 
        # with open(self.result_folder+ "/kde-density.pkl", 'rb') as f:
        #     zz = pickle.load(f) 
        # zz  = zz * 255
        coords = peak_local_max(zz, 3) 
        mask = np.zeros(zz.shape, dtype=bool) 
        for coord in coords:
            x,y = coord 
            if zz[x,y] < 100: 
                continue  
            mask[tuple(coord.T)] = True 
        markers, _ = ndi.label(mask) 
        labels = watershed(255-zz, markers)

        zz_mask = np.zeros(zz.shape, np.uint8) 
        zz_mask[zz == 0] = 255
        dist = cv2.distanceTransform(src = zz_mask, distanceType=cv2.DIST_L2, maskSize=0)
        color_table= np.loadtxt("colormaps/rapidtable.txt") 
        label_pseudo = color_table[labels].astype(np.uint8) 
        zz_mask = dist < 10 
        label_pseudo[~zz_mask] = np.asarray([255,255,255])
        print(labels.max())
        labels[~zz_mask] = 0
        
        cv2.imwrite(self.result_folder + "/labels.png", labels) 
        cv2.imwrite(self.result_folder + "/labels_pseudo.png", label_pseudo)

        colormap_vis = np.zeros([1000, 1000, 3], dtype=np.uint8)
        for i in range(10):
            for k in range(10):
                colormap_vis[100*i:100*(i+1), 100*k:100*(k+1),:] = color_table[i*10+k]
        cv2.imwrite(self.result_folder + "/colortable.png", colormap_vis)


    def edge_overlap(self): 
        labels = cv2.imread(self.result_folder + "/labels.png",cv2.IMREAD_GRAYSCALE)
        bg_id = labels[0,0]
        H,W = labels.shape
        print("bground:", bg_id)
        max_class = labels.max()
        print(labels.min())
        print(labels.max())
        labels = cv2.resize(labels, (W*4, H*4),interpolation=cv2.INTER_NEAREST)

        contours = []
        for k in range(max_class):
            tmp = labels.copy()
            tmp[labels==k+1] = 255
            tmp[tmp<255] = 0 
            c, hierarchy = cv2.findContours(tmp,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = np.asarray(c)
            contours.append(c)
        tmp = labels.copy() 
        
        ## draw contours
        H,W = labels.shape
        img = np.zeros((H, W), dtype=np.uint8) + 255
        for k in range(max_class):
            if k+1 == bg_id: 
                continue 
            else: 
                cv2.drawContours(img, contours[k], -1, 0,1)
                cv2.drawContours(labels, contours[k], -1, 255, 1)
        cv2.imwrite(self.result_folder + "/contour.png", img)
        cv2.imwrite(self.result_folder + "/label+contour.png", labels)
        with open(self.result_folder + "/contours.pkl", 'wb') as f: 
            pickle.dump(contours, f) 
        # calculate block center and draw label on it 
        for k in range(max_class): 
            if k+1 == bg_id: 
                cv2.putText(img, str(k), (5,20), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            else: 
                current_contours = contours[k]
                pos = current_contours.squeeze().mean(axis=0) 
                cv2.putText(img, str(k+1), pos.astype(np.int).tolist(), cv2.FONT_HERSHEY_COMPLEX, 1, 0)
        cv2.imwrite(self.result_folder + "/label+number.png", img)
        # from IPython import embed; embed()

    #### without manually checking the semantic label of each area, it is impossible to 
    #### draw hierarchical edges on the area. 
    # def hierarchical_edge(self): 
    #     labels = cv2.imread(self.result_folder + "/labels.png",cv2.IMREAD_GRAYSCALE)
    #     bg_id = labels[0,0]
    #     H,W = labels.shape
    #     max_class = labels.max()
    #     labels = cv2.resize(labels, (W*4, H*4),interpolation=cv2.INTER_NEAREST)

    #     ### each mini-area 
    #     contours = []
    #     for k in range(max_class):
    #         tmp = labels.copy()
    #         tmp[labels==k+1] = 255
    #         tmp[tmp<255] = 0 
    #         c, hierarchy = cv2.findContours(tmp,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         c = np.asarray(c)
    #         contours.append(c) 

    #     ### each middle area 
    #     color_table = np.loadtxt("colormaps/rapidtable.txt")
    #     color_table = color_table[65:]
    #     # color_table = color_table * 1.3
    #     # color_table += np.asarray([50,50,50])
    #     color_table[color_table > 255] = 255
    #     label_cluster = [ 
    #         [9,14,20,23,28,33,38], # (1) left lay 
    #         [7, 10,19],               # (2) right lay 
    #         [13, 16, 17],          # (3) belly lay 
    #         [35],                  # (4) lean 
    #         # [15,21,22,24,26],      # (5) bow left 
    #         # [25, 29,30,37, 6, 12,  11, 18],  # (6) bow right 
    #         [15,21,22,24,26,25, 29,30,37, 6, 12,  11, 18], # bow
    #         [3,4,2,8,27,31,32,34,36,40],     # (7) stand 
    #         [39],                            # (8) scratch 
    #         [1,5]                            # (9) sit 
    #     ]
    #     middle_contours = [] 
    #     middle_labels = labels.copy()
    #     for k, indices in enumerate(label_cluster): 
    #         tmp = np.zeros(labels.shape, dtype=labels.dtype)
    #         for index in indices: 
    #             tmp[labels==index] = 255
    #             middle_labels[labels==index] = k + 1 
            
    #         c, h = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         c = np.asarray(c)
    #         middle_contours.append(c) 
    #     label_pseudo = color_table[middle_labels].astype(np.uint8) 
    #     label_pseudo[middle_labels==0] = np.asarray([255,255,255])
    #     ### whole area 
    #     tmp = labels.copy()
    #     tmp[labels>0] = 255 
    #     c, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     out_contour = np.asarray(c) 

    #     ## draw contours: sub area 
    #     H,W = labels.shape
    #     img = np.zeros((H, W), dtype=np.uint8) + 255
    #     for k in range(max_class):
    #         if k+1 == bg_id: 
    #             continue 
    #         else: 
    #             cv2.drawContours(img, contours[k], -1, 150, 2)
    #             cv2.drawContours(label_pseudo, contours[k], -1, (150, 150, 150), 2)
    #     ## draw: middle area 
    #     for k, c in enumerate(middle_contours): 
    #         cv2.drawContours(img, c, -1, 0, 2)
    #         cv2.drawContours(label_pseudo, c, -1, (0,0,0), 2)

    #     ## draw: whole area 
    #     cv2.drawContours(img, out_contour, -1, 0, 3)
    #     cv2.drawContours(label_pseudo, out_contour, -1, (0,0,0), 3)

    #     cv2.imwrite(self.result_folder + "/contour.png", img)
    #     cv2.imwrite(self.result_folder + "/label+colortab.png", label_pseudo)

    #     ## draw for better edge, for paper use. 
    #     label_expand = np.zeros([H+4, W+4,3], dtype=np.uint8) + 255 
    #     label_expand[2:2+H,2:2+W] = label_pseudo
    #     cv2.drawContours(label_expand, out_contour+2, -1, (0,0,0), 3)
    #     cv2.imwrite(self.result_folder + "/label+colortab_expand.png", label_expand)

    def visualize_features(self, avg_width=1, write_obj=False):
        labels = cv2.imread(self.result_folder + "/labels.png", cv2.IMREAD_GRAYSCALE)
        with open(self.result_folder + "/kde-density.pkl", 'rb') as f: 
            density = pickle.load(f) 
        with open(self.result_folder + "/embedding.pkl", 'rb') as f: 
            embedding = pickle.load(f)
        with open(self.result_folder + "/clip_feature_index.pkl", 'rb') as f: 
            clip_feature_index = pickle.load(f) 
        with open(self.result_folder + "/all_data.pkl", 'rb') as f: 
            all_data = pickle.load(f) 

        M = embedding.max(axis=0)
        m = embedding.min(axis=0) 
        embedding = (embedding - m) / (M-m) * 0.9 + 0.05
        H, W = labels.shape
        embed_xy = embedding * np.asarray([H,W])
        embed_xy = embed_xy.astype(np.int32)
        embed_xy[:,1] = H-1-embed_xy[:,1]
        embed_label = labels[embed_xy[:,0], embed_xy[:,1]] 
        embed_density = density[embed_xy[:,0], embed_xy[:,1]]
        max_class = labels.max() 
        bg_id = labels[0,0]

        visual_features = [] 
        for k in range(max_class):
            if k+1==bg_id: 
                continue 
            index = (embed_label == k+1)
            
            current_density = embed_density[index].copy() 
            current_N = current_density.shape[0]
            current_feature_index =  clip_feature_index[index]
            print("label ", k+1, "  N: ", current_N)
            if current_N < avg_width: 
                continue 
            current_peak = current_density.argmax() 
            peak_clip_index = int(current_feature_index[current_peak][0])
            peak_frame_index = int(current_feature_index[current_peak][1])

            if avg_width == 1: 
                used_ids = np.asarray([current_peak])
            else: 
                used_ids = (np.random.rand(avg_width-1) * current_N).astype(np.int32)
                used_ids = np.asarray(used_ids.tolist() + [current_peak], dtype=np.int32)
            mean_rots = np.zeros([62,3]) 
            mean_t = np.zeros([3])
            total_w = 0 
            for i in range(used_ids.shape[0]):
                used_id = used_ids[i] 
                feature_id = current_feature_index[used_id]
                clip_index = int(feature_id[0]) 
                frame_index = int(feature_id[1])
                
                w = current_density[used_id] + 0.001
                t = all_data[clip_index]["trans"][frame_index]
                s = all_data[clip_index]["scale"]
                poseparam = all_data[clip_index]["rots"][frame_index]
                poseparam[0,0] = 0 
                t[0:2] = 0 
                t[2] /= s 
                mean_rots += poseparam * w 
                total_w += w 
                mean_t += t * w
            mean_rots /= total_w 
            mean_t /= total_w
        
            visual_features.append([k+1,mean_rots, mean_t, peak_clip_index, peak_frame_index]) 
        bones = [] 
        parents = self.bm.parents 
        for k in range(62): 
            if parents[k] >= 0: 
                bones.append([k,parents[k]])

        renderer = MainRender() 
        mesh = ColorRender() 
        mesh.load_data_basic(vertex = self.bm.vertices, face = self.bm.faces)
        mesh.set_color(np.ones(3))
        mesh.bind_arrays()
        renderer.color_object_list.append(mesh)
        renderer.build_floor()
        window = renderer.window
        
        a = time() 
        total_count = 0 
        k = 0 
        all_feature_num = len(visual_features)

        if write_obj: 
            if not os.path.exists(self.result_folder + "/objs/"): 
                os.makedirs(self.result_folder + "/objs/")
        while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS and glfwWindowShouldClose(window) == 0):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST) 
            renderer.set_bg_color((1,1,1))
            
            featureid, rot, trans, clip_index, frame_index = visual_features[k]
            V,J = self.bm.forward(rot, trans, 1)
         
            if write_obj: 
                self.bm.write_obj(self.result_folder+ "/objs/{}.obj".format(k))
            
            renderer.color_object_list[0].load_data_basic(vertex = V, face=self.bm.faces)
            renderer.color_object_list[0].bind_arrays()

            renderer.draw() 
            
            b = time() - a # seconds
            total_count += 1
            fps = (total_count) / (b + 0.0001) 
            info = "render fps: %2.2f"%(fps)
            renderer.draw_fg_text(info, (0, 300), 1, (0,0,0))
            info = "class {}".format(featureid)
            renderer.draw_fg_text(info, (0, 240), 1, (0,0,0))
            info = "clip index = {}, frame {}".format(clip_index, frame_index)
            renderer.draw_fg_text(info, (0, 180), 1, (0,0,0))
            info = "press LEFT or RIGHT to browse"
            renderer.draw_fg_text(info, (0, 120), 1, (0,0,0))
            if g_mouseState["key_left"]:
                if k > 0:  
                    k -= 1 
                if k == 0: 
                    k = all_feature_num - 1 
                g_mouseState["key_left"] = False 
            if g_mouseState["key_right"]: 
                if k < all_feature_num-1: 
                    k += 1 
                else: 
                    k = 0
                g_mouseState["key_right"] = False 

            if g_mouseState["saveimg"]: 
                img = renderer.readImage() 
                cv2.imwrite("saveimg{}.jpg".format(g_mouseState["saveimg_index"]), img) 
                g_mouseState["saveimg"] = False 
                g_mouseState["saveimg_index"] += 1
            glfwSwapBuffers(window)
            glfwPollEvents()

        glfwTerminate()

    def check_seq(self): 
        with open(self.result_folder + "/clip_feature_index.pkl",'rb') as f: 
            features_index = pickle.load(f) 
        with open(self.result_folder + "/all_data.pkl", 'rb') as f: 
            all_data = pickle.load(f) 
        bones = [] 
        parents = self.bm.parents 
        for k in range(62): 
            if parents[k] >= 0: 
                bones.append([k,parents[k]])

        renderer = MainRender() 
        # for k in range(1): 
        #     skel = SkelRender(jointnum=62, topo=bones)
        #     skel.bind_arrays()
        #     renderer.skel_object_list.append(skel) 
        mesh = ColorRender() 
        mesh.load_data_basic(vertex = self.bm.vertices, face = self.bm.faces)
        mesh.set_color(np.ones(3))
        mesh.bind_arrays()
        renderer.color_object_list.append(mesh)
        renderer.build_floor()
        window = renderer.window
        
        a = time() 
        total_count = 0 
        k = 0 

        max_n = features_index.shape[0]
        while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS and glfwWindowShouldClose(window) == 0):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST) 
            renderer.set_bg_color((1,1,1))
            
            feature = features_index[k]
            clip_index = int(feature[0])
            frame_index = int(feature[1])
            t = all_data[clip_index]["trans"][frame_index]
            s = all_data[clip_index]["scale"]
            poseparam = all_data[clip_index]["rots"][frame_index]
            V,J = self.bm.forward(poseparam, trans=t, scale=s)
            
            # renderer.skel_object_list[0].set_joint_position(J)
            # color = color1[0]
            # renderer.skel_object_list[0].set_joint_bone_color(color, color * 0.6)
            renderer.color_object_list[0].load_data_basic(vertex = V, face=self.bm.faces)
            renderer.color_object_list[0].bind_arrays()
            
            renderer.draw() 
            
            b = time() - a # seconds
            total_count += 1
            fps = (total_count) / (b + 0.0001) 
            info = "frame %d, render fps: %2.2f"%(k, fps)
            renderer.draw_fg_text(info, (0, 300), 1, (0,0,0))

            info = "clip {}, frame {}".format(clip_index, frame_index) 
            renderer.draw_fg_text(info, (0, 200), 1, (0,0,0))

            if g_mouseState["key_left"]:
                if k > 0:  
                    k -= 1 
                if k == 0: 
                    k = max_n - 1 
                g_mouseState["key_left"] = False 
            if g_mouseState["key_right"]: 
                if k < max_n-1: 
                    k += 1 
                else: 
                    k = 0
                g_mouseState["key_right"] = False 
            
            if g_mouseState["saveimg"]: 
                img = renderer.readImage() 
                cv2.imwrite("saveimg{}.jpg".format(g_mouseState["saveimg_index"]), img) 
                g_mouseState["saveimg"] = False 
                g_mouseState["saveimg_index"] += 1
            glfwSwapBuffers(window)
            glfwPollEvents()

        glfwTerminate()


if __name__ == "__main__":
    '''
    STEP1: 
    build feature for each clip. 
    It takes about 20 seconds. 
    '''
    print("...reading all clips and build features...")
    for clip_index in range(44):
        engin = CluterEngine()
        engin.read_clip(clip_index) 
        engin.build_feature_type2()

    '''
    STEP2: 
    PCA, tsne, density estimation, watershed and edge visualize. 
    It takes about 2 minutes. 
    '''
    params = { 
        "folder": "results/individual_behavior", 
        "pca_n": 16, 
        "feature_name": "clip_features_type2", 
        "bandwidth": 0.03, 
        "cluster_method" : "tsne", 
        "n_neigh": 80
    }

    engin = CluterEngine(result_folder=params['folder']) 
    engin.load_all_clips() 
    engin.feature_pca(n=params['pca_n'], feature_name=params['feature_name']) 
    engin.cluster(type=params['cluster_method'], n_neigh=params['n_neigh'])
    engin.density(bandwidth=params['bandwidth'])
    engin.watershed()
    engin.edge_overlap()

    '''
    STEP3: 
    uncomment this line to visualize pose at density peaks. 
    set write_obj to True will write every density peak model into 'nm_results/individual_behavior/objs/'
    '''
    engin.visualize_features(avg_width=1, write_obj=False)
    
    '''
    STEP4:
    uncomment this line to visualize clip sequence data.
    press LEFT and RIGHT to browse the data. 
    '''
    # engin.check_seq()   
