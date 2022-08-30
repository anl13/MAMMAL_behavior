import numpy as np 
import pickle 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm   
from bodymodel_np import BodyModelNumpy
from tqdm import tqdm 
from pig_render.Render import OBJ
import mpl_toolkits.axisartist as axisartist
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Patch 
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull 
from shapely.geometry import Polygon
import cv2 
from videoio import VideoWriter
from PIL import ImageFont, ImageDraw, Image 

font_list= [
    "arial.ttf"
]
font = ImageFont.truetype("fonts/"+font_list[0], 20)  
colors = [
    (173, 203, 227),
    (254, 138, 113), 
    (246, 205, 97), 
    (26, 153, 0)
]
g_bodypart = np.loadtxt("PIG_model/other_data/reduced_parts.txt", dtype=np.int64).squeeze()
g_body_part_names = [ 
    "NOT body", # 0
    "Main body",  # 1
    "Head", 
    "Left Ear",
    "Right Ear",
    "Left Front Leg", 
    "Right Front Leg", 
    "Left Hind Leg", 
    "Right Hind Leg", 
    "Tail"
]

def draw_background_box(img): 
    img = img[:,:,:]
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 170, 140),  fill=None,  outline=colors[0][::-1], width=2)
    draw.rectangle((40, 140, 170, 260), fill=None, outline=colors[1][::-1], width=2)
    draw.rectangle((40, 260, 170, 380), fill=None, outline=colors[2][::-1], width=2)
    draw.rectangle((40, 380, 170, 500), fill=None, outline=colors[3][::-1], width=2)
    return np.array(image)
    
def draw_label(img, labels):
    img = img[:,:,(2,1,0)]
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)

    for k in range(4): 
        for index, labeldata in enumerate(labels[k]):
            label, pidA, pidB = labeldata 
            W,H = font.getsize(label) 
            halfW = int(W/2)
            
            draw.rectangle((  45,           20+index * 40 + pidA * 120, 45 + halfW+5,   20 + index * 40 + pidA * 120+ H + 3 ), fill=colors[pidA], outline=colors[pidA], width=1)
            draw.rectangle((  45 + halfW+2, 20+index * 40 + pidA * 120, 45 + W + 10,    20 + index * 40 + pidA * 120+ H + 3 ), fill=colors[pidB], outline=colors[pidB], width=1)

            draw.text(( 50,20 + index * 40 + pidA * 120), label, font=font, fill=(0,0,0), align='center')
    return np.array(image)

figcolors = [
    (173, 203, 227),
    (254, 138, 113), 
    (246, 205, 97), 
    (26, 153, 0)
]

def load_related_data():
    folder = "data/batch5_nm/"
    with open(folder + "/pig_62_jsmth.pkl", 'rb') as f: 
        joints62 = pickle.load(f)
        joints62 = np.asarray(joints62) # [4, N, 62, 3]
    with open(folder + "/pig_angle.pkl", 'rb') as f: 
        body_angle = pickle.load(f) # [4, N]
    with open(folder + "/pig_center_speed.pkl", 'rb') as f: 
        center_speed = pickle.load(f) # [4, N]
    with open(folder + "/head_dists.pkl", 'rb') as f: 
        head_dists = pickle.load(f) 
        head_dists = np.asarray(head_dists) # [N, 12, 3]
    with open(folder + "/surface_dists.pkl", 'rb') as f: 
        surface_dists = pickle.load(f)
        surface_dists = np.asarray(surface_dists) # [N, 6, 3]
    with open(folder + "/surface_speed.pkl", 'rb') as f: 
        surface_speed = pickle.load(f) 
        surface_speed = np.asarray(surface_speed) # [4, N, 546]
    with open(folder + "/surface_speedvec.pkl",'rb') as f: 
        speedvec = pickle.load(f) # [4, N, 546, 3]
    with open(folder + "/surface.pkl", 'rb') as f: 
        surface = pickle.load(f) 
    with open(folder + "/head_dists_speed.pkl", 'rb') as f: 
        head_dist_speed = pickle.load(f) # [N, 12]

    return joints62, body_angle, center_speed, head_dists, surface_dists, surface_speed, speedvec, head_dist_speed, surface

# 0: out of social range 
# 1: group2
# 2: head-to-head
# 3: head-to-tail 
# 4: head_to_leg 
# 5: head-to-body
# 6: approach 
# 7: leave 
# 8: mount 
# 9: lean mount
def dyadic_behavior(head_dists, surface_speed_A, surface_dist, 
        body_angle_A, body_angle_B, surface_A, surface_B, current_shift):
    
    cur_surf_dist = surface_dist[current_shift]
    cur_surf_A = surface_A[current_shift, :, :]
    cur_surf_B = surface_B[current_shift, :, :]
    V_A_xy = cur_surf_A[:,0:2]
    V_B_xy = cur_surf_B[:,0:2]
    hull_A = ConvexHull(V_A_xy) 
    hull_B = ConvexHull(V_B_xy) 
    pA = Polygon(V_A_xy[hull_A.vertices])
    pB = Polygon(V_B_xy[hull_B.vertices])
    inter = pA.intersection(pB).area
    union = pA.area + pB.area - inter
    iou = 0. if union == 0 else inter/union
    mean_surf_dist = np.mean(surface_dist[:,0])
    mean_head_dist = np.mean(head_dists[:,0]) 
    head_src_id = int(head_dists[current_shift,1])
    head_tgt_id = int(head_dists[current_shift,2])

    if iou > 0.15 and body_angle_A[current_shift] > 20: 
        if body_angle_B[current_shift] > 20: 
            return 9 ## lean  
        else: 
            return 8 ## mount 
    if head_dists[-1,0] < 0.05 and head_dists[0,0] > 0.2: 
        if np.mean(surface_speed_A[:,262]) > 0.05:
            return 6 # approach 
    if head_dists[0,0] < 0.05 and head_dists[-1,0] > 0.2:
        if np.mean(surface_speed_A[:,262]) > 0.05:  
            return 7 # leave

    if mean_head_dist < 0.05 and g_bodypart[head_tgt_id] in [2,3,4]: 
        return 2 
 
    # if mean_head_dist < 0.05 and np.linalg.norm(cur_surf_B[head_tgt_id,:] - cur_surf_B[120,:]) < 0.1:
    if mean_head_dist < 0.05 and g_bodypart[head_tgt_id] in [9]:
        return 3 ## head-to-tail 
    if mean_head_dist < 0.05 and g_bodypart[head_tgt_id] in [5,6,7,8]: 
        return 4 ## head to leg
    if mean_head_dist < 0.05: 
        return 5 
    if mean_surf_dist < 0.05: 
        return 1 # group2 
    else: 
        return 0 # out of social range 

def transfer_head_dist_to_matrix(head_dist): 
    N = head_dist.shape[0]
    names_index = [
        [0,1], [0,2],[0,3], [1,0],[1,2], [1,3], 
        [2,0], [2,1], [2,3], [3,0], [3,1], [3,2]
    ]
    mat = np.zeros((N,4,4,3))
    for k in range(12): 
        a,b = names_index[k]
        mat[:,a,b] = head_dist[:,k]
    return mat 

def transfer_surface_dist_to_matrix(surface_dist): 
    N = surface_dist.shape[0]
    names_index_noorder = [
        [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
    ]
    mat = np.zeros((N,4,4,3))
    for k in range(6): 
        a,b = names_index_noorder[k]
        mat[:,a,b] = surface_dist[:,k]
        mat[:,b,a] = surface_dist[:,k]
    return mat 

def behavior_analysis():
    joints62, body_angle, center_speed, head_dists, \
        surface_dists, surface_speed, speedvec, \
        head_dist_speed, surface = load_related_data()
    head_dist_mat = transfer_head_dist_to_matrix(head_dists) 
    surface_dist_mat = transfer_surface_dist_to_matrix(surface_dists) 

    N = 1000
    folder = "data/batch5_nm/"
    names_index = [
        [0,1], [0,2],[0,3], [1,0],[1,2], [1,3], 
        [2,0], [2,1], [2,3], [3,0], [3,1], [3,2]
    ]
    state_dyadic = np.zeros((N,4,4))

    for i in tqdm(range(0, N)):
        startid = i - 12 
        endid = i + 13 
        if startid < 0: 
            startid = 0 
        if endid >= N: 
            endid = N
        current_shift = i - startid 

        for A,B in names_index: 
            # from A to B 
            state = dyadic_behavior(head_dist_mat[startid:endid, A, B], 
                surface_speed[A, startid:endid,:], 
                surface_dist_mat[startid:endid,A,B],
                body_angle[A,startid:endid], body_angle[B, startid:endid], 
                surface[A, startid:endid], surface[B, startid:endid], current_shift
                )
            state_dyadic[i,A,B] = state 
    with open(folder + "/nm_social2.pkl",'wb') as f: 
        pickle.dump(state_dyadic, f) 


def draw_behavior(): 
    cap  = cv2.VideoCapture("nm_results/social_behavior/batch5_rend.mp4") 
    with open("data/batch5_nm/nm_social2.pkl", 'rb') as f: 
        behavior = pickle.load(f) 
    behavior_names = [ 
        "out of social range", ## not used. 
        "in social range",  ## not used 
        "head - head",
        "head - tail",
        "head - limb",
        "head - body",
        "approach",
        "leave",
        "mount",
        "lean mount"  ## not used .
    ]
    writer = VideoWriter("nm_results/social_behavior/batch5_behavior.mp4", (960, 540), fps=25)

    for i in tqdm(range(1000)): 
        _, img = cap.read()
        labels = [[], [], [], []] 
        for A in range(4):
            for B in range(4):
                behav_id = int(behavior[i,A,B])
                if behav_id < 2: 
                    continue
                bname = behavior_names[behav_id]
                # pos = [100, 50 + k * 20]
                labels[A].append([bname, A, B])
                # img = cv2.putText(img, bname, pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, figcolors[A][::-1])
        
        img = draw_background_box(img)
        out = draw_label(img, labels)
               
        writer.write(out) 
    writer.close() 

def plot_behavior_chart(): 
    colorsetf = rgb = cm.get_cmap(plt.get_cmap('Accent'))(np.linspace(0.0,1.0,10))[:,:3]
    with open("data/batch5_nm/nm_social2.pkl", 'rb') as f: 
        label = pickle.load(f) 
    label_13 = label[300:650,1,3]
    N = label_13.shape[0] 
    width = []
    width_label = []
    tmp_state = label_13[0] 
    tmp_w = 0
    for k in range(N):
        if label_13[k] == tmp_state: 
            tmp_w += 1
            if k == N-1: 
                width_label.append(tmp_state)
                width.append(tmp_w) 
        else: 
            width.append(tmp_w)
            width_label.append(tmp_state) 
            tmp_state = label_13[k]
            tmp_w = 1
    width = np.asarray(width) 
    
    mpl.rc('font', family='Arial')
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    fig = plt.figure(figsize=(4,0.3))
    ax = fig.add_subplot(1,1,1)
    for k in range(width.shape[0]):
        if k == 0: 
            ax.barh(1, width[k], height=2, left=0, color=colorsetf[int(width_label[k])])
        ax.barh(1, width[k], height=2, left=width[0:k].sum(), color=colorsetf[int(width_label[k])]) 
    # ax.scatter([368-300, 584 - 300, 634-300],[0,0,0],marker='o',s=1,color='k')
    # ax.plot(xs,data[:,i,0], color=np.zeros(3), linewidth=0.5)
    ax.set_ylim(0,2.1)
    for line in ["right", "top"]: 
            ax.spines[line].set_visible(False) 
    for line in ["bottom", "left"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.plot(1, 0, ">k", markersize=0.8, transform=ax.get_yaxis_transform(), clip_on=False)
    # ax.plot(0, 2, "^k", markersize=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
    plt.xticks([0, 100,200,300], [12,16,20,24], fontsize=7)
    plt.title("Social behaviors from one pig to another", fontsize=7) 
    plt.text(-35, -1.5, "Time", fontsize=7)    
    plt.text(350, -1.5, " (s)", fontsize=7)
    plt.yticks([]) 

    # plt.show()
    plt.savefig("nm_results/social_behavior/Fig.2g.png", dpi=1000, bbox_inches="tight")
    plt.savefig("nm_results/social_behavior/Fig.2g.svg", dpi=1000, bbox_inches="tight")

if __name__ == "__main__":

    behavior_analysis() 

    draw_behavior()

    plot_behavior_chart()