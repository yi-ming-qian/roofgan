import numpy as np
import cv2
import imageio
import glob
import math
from tqdm import tqdm
import multiprocessing
import argparse

def comp_corner_distance(cor1,num1,cor2,num2,max_dist):
    diff = np.repeat(cor1, num2, axis=0) - np.tile(cor2, (num1, 1))
    diff_norm = np.linalg.norm(diff, axis=-1).reshape(num1, num2)
    minid = np.argmin(diff_norm, axis=1)
    cloest_error = diff_norm[np.arange(num1), minid]
    cor2_flag = np.full(num2, -1, dtype=np.int)
    # exclusion
    for i in range(num1):
        j = minid[i]
        if cor2_flag[j]<0:
            cor2_flag[j] = i
            continue
        if cloest_error[i]<=cloest_error[cor2_flag[j]]:
            cloest_error[cor2_flag[j]] = max_dist
            cor2_flag[j] = i
        else:
            cloest_error[i] = max_dist
    return np.mean(cloest_error)

def comp_roof_distance(height1, corners1, height2, corners2):
    pixel_step = 0.5
    max_dist = np.linalg.norm(np.array([64*pixel_step,64*pixel_step,np.amax(height1)]), axis=-1)

    roof_dist = 0.
    roof_num = 0
    face_num1, face_num2 = len(corners1), len(corners2)
    face_dists = np.full(face_num1,max_dist+1)
    face_minids = np.full(face_num1, -1, dtype=np.int)
    for f1, pix1 in enumerate(corners1): # corners1 is list
        num1 = pix1.shape[0]
        if num1==0:
            face_dists[f1] = max_dist
            face_minids[f1] = -1
            continue
        cor1 = np.concatenate([pix1*pixel_step, height1[pix1[:,0], pix1[:,1]].reshape(-1,1)], 1)
        for f2, pix2 in enumerate(corners2): # corners2 is list
            num2 = pix2.shape[0]
            if num2==0:
                continue
            cor2 = np.concatenate([pix2*pixel_step, height2[pix2[:,0], pix2[:,1]].reshape(-1,1)], 1)

            tmp_dist = comp_corner_distance(cor1,num1,cor2,num2,max_dist) + comp_corner_distance(cor2,num2,cor1,num1,max_dist)
            
            if tmp_dist<face_dists[f1]:
                face_dists[f1] = tmp_dist
                face_minids[f1] = f2

    face2_flag = np.full(face_num2, -1, dtype=np.int)   
    for i in range(face_num1):
        j = face_minids[i]
        if face2_flag[j]<0:
            face2_flag[j] = i
            continue
        if face_dists[i]<=face_dists[face2_flag[j]]:
            face_dists[face2_flag[j]] = max_dist
            face2_flag[j] = i
        else:
            face_dists[i] = max_dist
    return np.mean(face_dists)

def get_corners(facemasks, idx):
    all_masks = []
    for mask in facemasks:
        num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
        for i in range(1,num_labels):
            all_masks.append(labels_im==i)
    all_masks = np.asarray(all_masks)
    # sorting
    areas = np.sum(np.sum(all_masks, axis=-1), axis=-1)
    all_masks = all_masks[np.argsort(-areas)]

    face_label = np.full((64,64), -1, dtype=np.int)
    corner_label = np.full((64,64), -1, dtype=np.int)
    for i, mask in enumerate(all_masks):
        face_label[mask] = i
    all_contours = []
    all_corners0, all_corners1 = [], []
    for i, mask in enumerate(all_masks):
        _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        corners0, corners1 = [], []
        for c in contours:
            p = c[:,0,:]
            all_contours.append(p)
            pnum = p.shape[0]
            for k in range(pnum):
                nextk = (k+1)%pnum
                if p[k,1]==p[nextk,1] and abs(p[k,0]-p[nextk,0])>2:
                    if p[k,0]>p[nextk,0]:
                        corners0.append(p[nextk,:])
                        corners0.append(p[k,:])
                    else:
                        corners0.append(p[k,:])
                        corners0.append(p[nextk,:])
                elif p[k,0]==p[nextk,0] and abs(p[k,1]-p[nextk,1])>2:
                    if p[k,1]>p[nextk,1]:
                        corners1.append(p[nextk,:])
                        corners1.append(p[k,:])
                    else:
                        corners1.append(p[k,:])
                        corners1.append(p[nextk,:])

        corners0 = np.asarray(corners0)
        corners1 = np.asarray(corners1)

        if corners0.shape[0]!=0:
            corner_label[corners0[:,1], corners0[:,0]] = i
            all_corners0.append(corners0)
        if corners1.shape[0]!=0:
            corner_label[corners1[:,1], corners1[:,0]] = i
            all_corners1.append(corners1)
    all_contours = np.concatenate(all_contours,0)
    if len(all_corners0) == 0:
        all_corners0 = np.array([])
    else:
        all_corners0 = np.concatenate(all_corners0,0)
    if len(all_corners1) == 0:
        all_corners1 = np.array([])
    else:
        all_corners1 = np.concatenate(all_corners1,0)

    for k in range(0, all_corners0.shape[0], 2):
        y = all_corners0[k,1]
        for x in [all_corners0[k,0]-1, all_corners0[k+1,0]+1]:
            flag = corner_check(x, y, face_label, corner_label, winsize=2)
            if flag == False:
                flag = corner_check(x, y-1, face_label, corner_label, winsize=2)
            if flag == False:
                flag = corner_check(x, y+1, face_label, corner_label, winsize=2)

    for k in range(0, all_corners1.shape[0], 2):
        x = all_corners1[k,0]
        for y in [all_corners1[k,1]-1, all_corners1[k+1,1]+1]:
            flag = corner_check(x, y, face_label, corner_label, winsize=2)
            if flag == False:
                flag = corner_check(x-1, y, face_label, corner_label, winsize=2)
            if flag == False:
                flag = corner_check(x+1, y, face_label, corner_label, winsize=2)

    all_corners = []
    for i in range(all_masks.shape[0]):
        ylist, xlist = np.nonzero(corner_label == i)
        all_corners.append(np.concatenate([ylist.reshape(-1,1),xlist.reshape(-1,1)],axis=1))
    
    return all_masks, all_corners
    # visualization
    img = np.ones((64,64,3))
    corner_img = np.zeros((64,64,3))
    corner_gifs = []
    base_gif = (face_label>=0).astype(np.uint8)*125
    for i, mask in enumerate(all_masks):
        tmp_color = np.random.random(3)*255
        img[mask,:] = tmp_color
        corner_img[mask,:] = tmp_color
        corner_img[all_corners[i][:,0], all_corners[i][:,1],:] = 255-tmp_color
        tmp_img = base_gif.copy()
        tmp_img[mask] = 255
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
        tmp_img[all_corners[i][:,0], all_corners[i][:,1],:] = np.array([0,255,0])
        corner_gifs.append(tmp_img)

    #cv2.imwrite(f"./experiments/proj_log_temp/pqnet-PartNet-Lamp/results/fake_z_ckpt100000_num20000-mesh-p1_graph/fid_image/{idx}.png", np.concatenate([img],axis=1))
    # if idx==518:
    #     imageio.mimsave(f"./experiments/{idx}.gif", corner_gifs, duration=1)

    
def corner_check(x, y, face_label, corner_label, winsize=1):
    if x<0 or x>=64 or y<0 or y>=64 or face_label[y,x]<0:
        return False
    tmp_win = local_window(face_label, x, y, winsize)
    if np.sum(tmp_win==face_label[y,x])==(6+(winsize-1)*9):
        return False
    tmp_win = local_window(corner_label, x, y, winsize)
    if np.sum(tmp_win==face_label[y,x])==0:
        corner_label[y,x] = face_label[y,x]
        return True
    return False
def local_window(m, x, y, winsize=1):
    xmin = max(0,x-winsize)
    xmax = min(64,x+winsize+1)
    ymin = max(0,y-winsize)
    ymax = min(64,y+winsize+1)
    return m[ymin:ymax, xmin:xmax]

def load_data(path, num, postfix):
    all_heights, all_corners, all_nblocks = [], [], []
    for i in range(num):
        data = np.load(path+f"{i}{postfix}.npz")
        height, masks, num_blocks = data["height_map"], data["face_masks"], data["num_blocks"][0]
        if masks.shape[0]==0:
            all_heights.append(height)
            all_corners.append([np.array([])])
            all_nblocks.append(num_blocks)
            print("zero masks in generation")
            continue
        masks, corners = get_corners(masks, i)
        all_heights.append(height)
        all_corners.append(corners)
        all_nblocks.append(num_blocks)
    return all_heights, all_corners, all_nblocks

def my_weight(x):
    return 1/2+1/(1+math.exp(-x))

def eval_MMD(target_heights, target_corners, target_nblocks, target_num, source_heights, source_corners, source_nblocks, source_num):
    mutual = True
    dists = np.full((target_num, 2), 1e8)
    scaled_dists = np.full((target_num, 2),1e8)
    for i in tqdm(range(target_num)):
        t_height, t_corners, t_nblocks = target_heights[i], target_corners[i], target_nblocks[i]
        for j in range(source_num):
            s_height, s_corners, s_nblocks = source_heights[j], source_corners[j], source_nblocks[j]
            dist = comp_roof_distance(t_height, t_corners, s_height, s_corners)
            if mutual:
                dist += comp_roof_distance(s_height, s_corners, t_height, t_corners)

            if dist<dists[i,0]:
                dists[i,0] = dist
                dists[i,1] = j
            weight = my_weight(abs(t_nblocks-s_nblocks))
            dist *= weight
            if dist<scaled_dists[i,0]:
                scaled_dists[i,0] = dist
                scaled_dists[i,1] = j
        if debug:
            print(i ,dists[i], scaled_dists[i])
    
    return dists, scaled_dists

def get_dist_average(all_dists):
    target_num = all_dists.shape[1]
    minid = np.argmin(all_dists[:,:,0], axis=0)
    dist_mat = all_dists[minid,np.arange(target_num),:]
    return np.mean(dist_mat[:,0]), dist_mat

def MMD_processor(augid, target_path, target_num, source_heights, source_corners, source_nblocks, source_num):
    target_heights, target_corners, target_nblocks = load_data(target_path, target_num, f"-{augid}_gt")
    dist_mat, scaled_dist_mat = eval_MMD(target_heights, target_corners, target_nblocks, target_num, 
        source_heights, source_corners, source_nblocks, source_num)
    return dist_mat, scaled_dist_mat

def evaluate(args):
    global debug
    debug = False
    # parameter    
    method= args.method
    exclude = args.exclude
    gnum = args.gnum # generated sample number
    tasks = args.tasks
    ##########################
    # is_gan = method!="pqnet"
    base_path = "../experiments/results/"

    if exclude==0:
        target_path = base_path + "gt_test/test64/"
        source_path=base_path+ f"{method}/"
    else:
        target_path = base_path + f"gt_test/{exclude}/"
        source_path=base_path+ f"{method}-{exclude}/"
                    
    target_num = 64#len(glob.glob(target_path+"*_gt.png"))
    source_num = gnum

    fp = open(f"{method}_exclude{exclude}.txt", "w")
    fp.write("target:\t"+target_path+"\n")
    fp.write("source:\t"+source_path+"\n")
    fp.write("target num:\t"+str(target_num)+"\n"+"source num:\t"+str(source_num)+"\n")
    fp.write("method:\t"+method+"\n")
    fp.write("exclude:\t"+str(exclude)+"\n")
    fp.write("result:\tRMMD\tscaled_RMMD\n") # please ignore scaled RMMD
    for ret in tasks:
        source_heights, source_corners, source_nblocks = load_data(source_path, source_num, "_"+ret)
        ################ MMD
        pool = multiprocessing.Pool(8)
        results = []
        for augid in range(8):
            results.append(pool.apply_async(MMD_processor,(augid, target_path, target_num, 
                source_heights, source_corners, source_nblocks, source_num)))
        pool.close()
        pool.join()
        pool.terminate()

        all_dists, all_scaled_dists= [], []
        for result in results:
            dist_mat, scaled_dist_mat = result.get()
            all_dists.append(dist_mat)
            all_scaled_dists.append(scaled_dist_mat)

        all_dists = np.stack(all_dists)
        all_scaled_dists = np.stack(all_scaled_dists)
        # find the min from all augmented cases
        dist_MMD, dist_mat = get_dist_average(all_dists)
        scaled_dist_MMD, scaled_dist_mat = get_dist_average(all_scaled_dists)
        print(dist_MMD, scaled_dist_MMD)
        fp.write(f"{ret}:\t{dist_MMD/4}\t{scaled_dist_MMD/4}\t")
        #np.save(source_path+f"dist_t{target_num}_s{source_num}_{ret}.npz", dist_mat)
        #np.save(source_path+f"scaled_dist_t{target_num}_s{source_num}_{ret}.npz", scaled_dist_mat)

    fp.close()

def main():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--exclude', type=int, default=0)
    parser.add_argument('--gnum', type=int, default=1000)
    parser.add_argument('--tasks', type=str, default="raw,thres,graph")
    args = parser.parse_args()
    args.tasks = [i for i in args.tasks.split(',')]
    evaluate(args)

if __name__ == '__main__':
    main()