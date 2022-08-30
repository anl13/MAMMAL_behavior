import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R
import pickle 
from bodymodel_np import BodyModelNumpy 
from sklearn.decomposition import PCA 
import cv2
from tqdm import tqdm  
from PIL import ImageFont, ImageDraw, Image 
from videoio import VideoWriter 
import matplotlib as mpl 
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D

def check_drink(skel, criterion):
    target1 = np.asarray([-0.73902, 0.811039, 0.240614])
    target2 = np.asarray([-0.303331, 0.811231, 0.321842])
    nose = skel[0]
    dist1 = np.linalg.norm(target1 - nose) 
    dist2 = np.linalg.norm(target2 - nose)
    if dist1 < criterion : 
        return 1 
    if dist2 < criterion : 
        return 1
    return 0

def check_eat(skel):
    y_lim = 0.7
    x_left = 0.02
    x_right = 0.64
    nose = skel[0]
    if nose[0] > x_left and nose[0] < x_right and nose[1] > y_lim and nose[2] < 0.2: 
        return 1
    return 0

def check_window(window, index):
    if window[0:index].sum() >=1 and window[index:].sum() >= 1:
        return 1 
    else: 
        return 0

def accum_assend(seq):
    length = seq.shape[0]
    count = 0
    for i in range(length-1):
        if seq[i] == 0 and seq[i+1] ==1: 
            count += 1 
    return count 

def check_drink_seq():
    for pig in range(4):
        with open("data/drink_data/1005/pig_{}_joints19.pkl".format(pig), 'rb') as f: 
            data = pickle.load(f)
        length = len(data) 
        drink_state = np.zeros(length)
        criterion = 0.12
        for i in tqdm(range(length)):
            drink_state[i] = check_drink(data[i], criterion)

        output = np.zeros(length) 
        for i in tqdm(range(length)):
            left = int(max(i-50, 0))
            right = int(min(i+50, length))
            if drink_state[i] == 1: 
                if drink_state[left:right].sum() < 10: 
                    output[i] = 0
                output[i] = 1
            else:
                output[i] = check_window(drink_state[left:right], i-left)
        total_times = accum_assend(output) 
        print("total_drink time : ", output.sum() / 25.0, "  s")
        print("total drink count: ", total_times)
        with open("data/drink_data/1005/drink_{}.pkl".format(pig), 'wb') as f: 
            pickle.dump( output,f)

def check_eat_seq():
    with open("data/drink_data/0705_10/skel19.pkl", 'rb') as f: 
        skel = pickle.load(f)

    for pid in range(4):
        length = 1000 
        drink_state = np.zeros(length)
        criterion = 0.12
        for i in tqdm(range(length)):
            drink_state[i] = check_eat(skel[pid, i])

        output = np.zeros(length) 
        for i in tqdm(range(length)):
            left = int(max(i-50, 0))
            right = int(min(i+50, length))
            if drink_state[i] == 1: 
                if drink_state[left:right].sum() < 10: 
                    output[i] = 0
                output[i] = 1
            else:
                output[i] = check_window(drink_state[left:right], i-left)
        with open("data/drink_data/0705_10/eat_{}.pkl".format(pid), 'wb') as f: 
            pickle.dump(output,f)

font_list= [
    "arial.ttf"
]
font = ImageFont.truetype("fonts/"+font_list[0], 40)  
colors = [
    (173, 203, 227),
    (254, 138, 113), 
    (246, 205, 97), 
    (26, 153, 0)
]
def draw_drink_label(img, index, posindex, label="drink"):
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    draw.rectangle(( 45,20+posindex * 40, 155,20+posindex * 40 + 45 ), fill='white', outline=colors[index], width=1)
    draw.text(( 50,20 + posindex * 40), label, font=font, fill=colors[index], align='center')
    return np.array(image)

## 2022.07.26
## draw new fig, smaller. 
def eat_supp_fig_new(): 
    all_drinks = []
    for k in range(4): 
        with open("data/drink_data/0705_10/eat_{}.pkl".format(k), 'rb') as f: 
            drink_A = pickle.load(f)
            all_drinks.append(drink_A)
         
    mpl.rc('font', family='Arial')
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    fig = plt.figure(figsize=(1.5,1)) 
    # pignames = [2,1,3,0]
    figcolors = [
        (173, 203, 227),
        (254, 138, 113), 
        (246, 205, 97), 
        (26, 153, 0)
    ]
    xs = np.arange(1000) 
    frameindex=999
    plt.clf()

    for pid in range(4): 
        label = all_drinks[pid]
        N = label.shape[0] 
        width = []
        width_label = []
        tmp_state = label[0] 
        tmp_w = 0
        for k in range(N):
            if label[k] == tmp_state: 
                tmp_w += 1
                if k == N-1: 
                    width_label.append(tmp_state)
                    width.append(tmp_w) 
            else: 
                width.append(tmp_w)
                width_label.append(tmp_state) 
                tmp_state = label[k]
                tmp_w = 1
        width = np.asarray(width) 

        ax = fig.add_subplot(4,1,pid+1)
        c = [
            np.ones(3), 
            np.zeros(3) + 0.5, 
            np.asarray(figcolors[pid]) / 255
        ]
        for k in range(width.shape[0]):
            if k == 0: 
                ax.barh(0.5, width[k], height=1, left=0, facecolor=c[int(width_label[k])], edgecolor=(0,0,0), linewidth=0.2, linestyle='--')
            ax.barh(0.5, width[k], height=1, left=width[0:k].sum(),facecolor=c[int(width_label[k])], edgecolor=(0,0,0), linewidth=0.2, linestyle='--') 
        ax.set_ylim(0,1.1)
        # ax.set_xlim(0,1000)
        ax.set_xticks([0,500,1000], [0,20,40], fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel("Pig{}".format(pid+1), fontsize=7, rotation=0, color=c[2])
        # ax.scatter([frameindex], [all_drinks[k][frameindex]], s=15, color=c, edgecolors=c * 0.5)
        if pid == 0: 
            color1 = [
                np.ones(3), 
                np.zeros(3) + 0.5
            ]
            legends = ["Not Eat", "Eat"]
            patches = [] 
            for legend_id in range(2): 
                patch1 = mpatches.Patch(facecolor=color1[legend_id], label=legends[legend_id], linewidth=0.2,
                    linestyle='--',edgecolor=(0,0,0)) 
                patches.append(patch1)
            ax.legend(handles=patches, bbox_to_anchor=(-0.1,1.1), loc=3, borderaxespad=0, fontsize=6, ncol=3, frameon=False)
        if pid < 3:
            for line in ["right", "top"]: 
                ax.spines[line].set_visible(False)
                ax.set_xticks([])
        else: 
            for line in ["right", "top"]: 
                ax.spines[line].set_visible(False) 
        for line in ["bottom", "left", "right"]: 
            ax.spines[line].set_linewidth(0.5)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)
        ax.plot(1, 0, ">k", markersize=0.8, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.xlabel("Time (s)", fontsize=7)
    plt.savefig("nm_results/scene_behavior/eat.png", dpi=1000, bbox_inches="tight")
    plt.savefig("nm_results/scene_behavior/eat.svg", dpi=1000, bbox_inches="tight")



def drink_supp_fig_new(): 
    all_drinks = []
    for k in range(4): 
        with open("data/drink_data/1005/drink_{}.pkl".format(k), 'rb') as f: 
            drink_A = pickle.load(f)
            all_drinks.append(drink_A[2450:3450])
         
    mpl.rc('font', family='Arial')
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    fig = plt.figure(figsize=(1.5,1)) 
    # pignames = [2,1,3,0]
    pignames = [2,1,3,0]
    figcolors = [
        (173, 203, 227),
        (254, 138, 113), 
        (246, 205, 97), 
        (26, 153, 0)
    ]
    xs = np.arange(1000) 
    frameindex=999
    plt.clf()

    for pid in range(4): 
        label = all_drinks[pid]
        N = label.shape[0] 
        width = []
        width_label = []
        tmp_state = label[0] 
        tmp_w = 0
        for k in range(N):
            if label[k] == tmp_state: 
                tmp_w += 1
                if k == N-1: 
                    width_label.append(tmp_state)
                    width.append(tmp_w) 
            else: 
                width.append(tmp_w)
                width_label.append(tmp_state) 
                tmp_state = label[k]
                tmp_w = 1
        width = np.asarray(width) 

        ax = fig.add_subplot(4,1,pid+1)
        c = [
            np.ones(3), 
            np.zeros(3) + 0.5, 
            np.asarray(figcolors[pignames[pid]]) / 255
        ]
        for k in range(width.shape[0]):
            if k == 0: 
                ax.barh(0.5, width[k], height=1, left=0, facecolor=c[int(width_label[k])], edgecolor=(0,0,0), linewidth=0.2, linestyle='--')
            ax.barh(0.5, width[k], height=1, left=width[0:k].sum(), facecolor=c[int(width_label[k])], edgecolor=(0,0,0), linewidth=0.2, linestyle='--') 
        ax.set_ylim(0,1.1)
        # ax.set_xlim(0,1000)
        ax.set_xticks([0,500,1000], [0,20,40], fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel("Pig{}".format(pid+1), fontsize=7, rotation=0, color=c[2])
        # ax.scatter([frameindex], [all_drinks[k][frameindex]], s=15, color=c, edgecolors=c * 0.5)
        if pid == 0: 
            color1 = [
                np.ones(3), 
                np.zeros(3) + 0.5
            ]
            legends = ["Not Drink", "Drink"]
            patches = [] 
            for legend_id in range(2): 
                patch1 = mpatches.Patch(facecolor=color1[legend_id], label=legends[legend_id], linewidth=0.2,
                    linestyle='--',edgecolor=(0,0,0)) 
                patches.append(patch1)
            ax.legend(handles=patches, bbox_to_anchor=(-0.1,1.1), loc=3, borderaxespad=0, fontsize=6, ncol=3, frameon=False)
        if pid < 3:
            for line in ["right", "top"]: 
                ax.spines[line].set_visible(False)
                ax.set_xticks([])
        else: 
            for line in ["right", "top"]: 
                ax.spines[line].set_visible(False) 
        for line in ["bottom", "left", "right"]: 
            ax.spines[line].set_linewidth(0.5)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)
        ax.plot(1, 0, ">k", markersize=0.8, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.xlabel("Time (s)", fontsize=7)
    plt.savefig("nm_results/scene_behavior/drink.png", dpi=1000, bbox_inches="tight")
    plt.savefig("nm_results/scene_behavior/drink.svg", dpi=1000, bbox_inches="tight")

if __name__ == "__main__":
    check_eat_seq() 
    check_drink_seq() 
    drink_supp_fig_new() 
    eat_supp_fig_new()