import numpy as np
import math
import cv2
import imageio

H_GABLE = 0
V_GABLE = 1
H_HIP = 2
V_HIP = 3
pixel_step = 0.5
IMG_SIZE = 64
EPISON = 1e-6
T_angle, T_boundary, T_middle, T_height = 0.94, 1.1, 4., 0.07#0.94, 1.+1e-4, 4., 0.07

class House(object):
    def __init__(self, block_types, founda_positions, founda_height, roof_angles):
        self.block_types = block_types
        self.num_blocks = len(block_types)
        if len(founda_positions.shape)==2:
            self.founda_positions = founda_positions
        elif len(founda_positions.shape)==3:
            self.founda_probs = founda_positions
            self.probmask_to_bb()
        else:
            raise NotImplementedError
        self.founda_height = founda_height
        self.roof_angles = roof_angles # cos value
        self.remove_invalid_blocks()
        self.angle_to_normal()
        self.normal_to_heightratio()
        self.roof_graph = np.full_like(self.founda_positions, -1, dtype=np.int)
    
    def remove_invalid_blocks(self):
        valid_flags = np.full(self.num_blocks, True)
        for i in range(self.num_blocks):
            ymin, ymax, xmin, xmax = self.founda_positions[i,:]
            if ymax-ymin<1e-4 or xmax-xmin<1e-4:
                valid_flags[i] = False
        self.block_types = self.block_types[valid_flags]
        self.num_blocks = np.sum(valid_flags)
        self.founda_positions = self.founda_positions[valid_flags, :]
        self.roof_angles = self.roof_angles[valid_flags, :]
        self.valid_block_flags = valid_flags

    def probmask_to_bb(self):
        founda_mask = self.founda_probs>0.5
        self.founda_positions = np.zeros((self.num_blocks, 4))
        for i in range(self.num_blocks):
            num_labels, labels_im = cv2.connectedComponents(founda_mask[i].astype(np.uint8))
            if num_labels==1:
                self.founda_positions[i,:] = 0
                continue
            max_area=-1.
            for j in range(1,num_labels):
                tmp_flag = labels_im==j
                tmp_area = np.sum(tmp_flag)
                if tmp_area>max_area:
                    max_area=tmp_area
                    max_mask = tmp_flag
            ylist, xlist = np.nonzero(max_mask)
            if ylist.shape[0]==0:
                ylist=np.array([32])
            if xlist.shape[0]==0:
                xlist=np.array([32])
            self.founda_positions[i,0] = np.amin(ylist)
            self.founda_positions[i,1] = np.amax(ylist)
            self.founda_positions[i,2] = np.amin(xlist)
            self.founda_positions[i,3] = np.amax(xlist)
    def save_raw_founda_prob(self, filename):
        tmp_gifs = []
        for i in range(self.num_blocks):
            tmp_img = np.zeros((64,64,3))
            tmp_img[:,:,1] = self.founda_probs[i]
            tmp_gifs.append((tmp_img*255).astype(np.uint8))
        imageio.mimsave(filename, tmp_gifs, duration=1)

    def angle_to_normal(self):
        roof_normals = []
        for i, btype in enumerate(self.block_types):
            roof_normal = np.zeros((4,3))
            tb_angle_cos, lr_angle_cos = self.roof_angles[i,:]
            tb_angle_sin = math.sqrt(abs(1.-tb_angle_cos**2))
            lr_angle_sin = math.sqrt(abs(1.-lr_angle_cos**2))

            if btype == H_GABLE:
                roof_normal[0,:] = [0., -tb_angle_sin, tb_angle_cos]
                roof_normal[1,:] = [0., tb_angle_sin, tb_angle_cos]
            elif btype == V_GABLE:
                roof_normal[2,:] = [-lr_angle_sin, 0., lr_angle_cos]
                roof_normal[3,:] = [lr_angle_sin, 0., lr_angle_cos]
            else:
                roof_normal[0,:] = [0., -tb_angle_sin, tb_angle_cos]
                roof_normal[1,:] = [0., tb_angle_sin, tb_angle_cos]
                roof_normal[2,:] = [-lr_angle_sin, 0., lr_angle_cos]
                roof_normal[3,:] = [lr_angle_sin, 0., lr_angle_cos]
            roof_normals.append(roof_normal)
        self.roof_normals = np.asarray(roof_normals)

    def normal_to_heightratio(self):
        roof_heights = np.zeros(self.block_types.shape)
        roof_ratios = np.zeros(self.block_types.shape)
        for i, btype in enumerate(self.block_types):
            ymin = self.founda_positions[i,0]
            ymax = self.founda_positions[i,1]
            xmin = self.founda_positions[i,2]
            xmax = self.founda_positions[i,3]
            ymiddle = (ymin+ymax)/2.0
            xmiddle = (xmin+xmax)/2.0
            point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
            if btype == H_GABLE:
                roof_heights[i] = update_height(self.roof_normals[i,0], point, xmin, ymiddle)
            elif btype == V_GABLE:
                roof_heights[i] = update_height(self.roof_normals[i,2], point, xmiddle, ymin)
            elif btype == H_HIP:
                roof_heights[i] = update_height(self.roof_normals[i,0], point, xmin, ymiddle)
                xtmp = update_ratio_h(self.roof_normals[i,2], point, xmin, roof_heights[i])
                roof_ratios[i] = max(0,min(0.499,(xtmp-xmin)/(xmax-xmin)))
            elif btype == V_HIP:
                roof_heights[i] = update_height(self.roof_normals[i,2], point, xmiddle, ymin)
                ytmp = update_ratio_v(self.roof_normals[i,0], point, xmin, roof_heights[i])
                roof_ratios[i] = max(0,min(0.499,(ytmp-ymin)/(ymax-ymin)))
            else:
                print('block type not supported')
                exit()
        self.roof_heights = roof_heights
        self.roof_ratios = roof_ratios

    def sequential_snap(self):
        coli_label = np.full((self.num_blocks, self.num_blocks, 6), False, dtype=np.bool)
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if i==j:
                    continue
                for f in range(4):
                    if abs(self.founda_positions[i,f]-self.founda_positions[j,f])<T_boundary:
                        coli_label[i,j,f] = True
                for f in range(2):
                    if np.dot(self.roof_normals[i,f], self.roof_normals[j,f])>T_angle:
                        coli_label[i,j,f+4] = True
        self.graph_snap(coli_label)
    def graph_snap(self, coli_label):
        # coli_label: blocknum x blocknum x 3 x4
        if coli_label.shape[0] != self.num_blocks:
            coli_label = coli_label[self.valid_block_flags,:,:]
            coli_label = coli_label[:,self.valid_block_flags,:]
        # sort
        to_rel_num, block_area = np.zeros(self.num_blocks), np.zeros(self.num_blocks)
        for i in range(self.num_blocks):
            to_rel_num[i] = np.sum(coli_label[:,i,:4])
            block_area[i] = abs((self.founda_positions[i,1]-self.founda_positions[i,0])*(self.founda_positions[i,3]-self.founda_positions[i,2]))
        block_order = np.lexsort((to_rel_num, block_area))[::-1]

        roof_indicators = np.full_like(self.founda_positions, True, dtype=np.bool)
        roof_updated = np.full_like(self.founda_positions, False, dtype=np.bool)
        founda_updated = np.full_like(self.founda_positions, False, dtype=np.bool)
        for i, btype in enumerate(self.block_types):
            if btype==H_GABLE:
                roof_indicators[i,2] = False
                roof_indicators[i,3] = False
            elif btype==V_GABLE:
                roof_indicators[i,0] = False
                roof_indicators[i,1] = False

        middles = np.full_like(self.founda_positions, -1.)
        for i, btype in enumerate(self.block_types):
            if btype%2==0:
                middles[i,0] = (self.founda_positions[i,0]+self.founda_positions[i,1])/2.
                middles[i,1] = middles[i,0]
            else:
                middles[i,2] = (self.founda_positions[i,2]+self.founda_positions[i,3])/2.
                middles[i,3] = middles[i,2]
        for i in block_order:
            btype = self.block_types[i]
            for f in range(4):
                symid = symmetry_faceid(f)
                for j in block_order:
                    if founda_updated[i,f] == True or j==i:
                        break
                    mid_pos = (self.founda_positions[j,f] + self.founda_positions[j,symid])/2.
                    if coli_label[i,j,f] or abs(self.founda_positions[i,f]-self.founda_positions[j,f])<T_boundary:
                    
                        gap = self.founda_positions[j,f]-self.founda_positions[i,f]
                        self.founda_positions[i,f] = self.founda_positions[j,f]
                        
                        founda_updated[i,f] = True
                        if abs(self.roof_heights[i]-self.roof_heights[j])/self.roof_heights[i] < T_height:
                            self.roof_heights[i] = self.roof_heights[j]
                    elif middles[j,f]>0.0 and abs(self.founda_positions[i,f]-middles[j,f])<T_middle and ((btype+self.block_types[j])%2)!=0:
                        gap = mid_pos - self.founda_positions[i,f]
                        self.founda_positions[i,f] = mid_pos
                        
                        founda_updated[i,f] = True
                        if abs(self.roof_heights[i]-self.roof_heights[j])/self.roof_heights[i] < T_height:
                            self.roof_heights[i] = self.roof_heights[j]
                    if roof_indicators[j,f]==False or roof_indicators[i,f]==False:
                        continue
                
                    if (np.dot(self.roof_normals[i,f],self.roof_normals[j,f])>T_angle or coli_label[i,j,int(f/2)+4]) and abs(self.founda_positions[i,f]-self.founda_positions[j,f])<T_boundary:
                        self.roof_normals[i,f] = self.roof_normals[j,f].copy()
                        self.roof_normals[i,symid] = self.roof_normals[j,symid].copy()
                        roof_updated[i,f] = True
                        roof_updated[i,symid] = True
                        self.roof_graph[i,f] = self.roof_graph[j,f] if self.roof_graph[j,f]>=0 else j*4+f
            self.normal_to_foundaratio(i, btype, founda_updated)

    def normal_to_foundaratio(self, i, btype, founda_updated):
        ymin = self.founda_positions[i,0]
        ymax = self.founda_positions[i,1]
        xmin = self.founda_positions[i,2]
        xmax = self.founda_positions[i,3]
        ymiddle = (ymin+ymax)/2.
        xmiddle = (xmin+xmax)/2.

        if btype == H_GABLE:
            if founda_updated[i,0]==True and founda_updated[i,1]==True:
                pass
                #print(str(H_GABLE)+": both top and bottom boundaries are fixed")
            elif founda_updated[i,0]==True and founda_updated[i,1]==False:
                point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
                ymiddle = update_founda_h(self.roof_normals[i,0], point, xmin, self.roof_heights[i])
                self.founda_positions[i,1] = 2.0*ymiddle-ymin
            else:
                point = np.array([xmin*pixel_step, ymax*pixel_step, self.founda_height])
                ymiddle = update_founda_h(self.roof_normals[i,1], point, xmin, self.roof_heights[i])
                self.founda_positions[i,0] = 2.0*ymiddle-ymax

        elif btype == V_GABLE:
            if founda_updated[i,2]==True and founda_updated[i,3]==True:
                pass
                #print(str(V_GABLE)+": both left and right boundaries are fixed")
            elif founda_updated[i,2]==True and founda_updated[i,3]==False:
                point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
                xmiddle = update_founda_v(self.roof_normals[i,2], point, ymin, self.roof_heights[i])
                self.founda_positions[i,3] = 2.0*xmiddle-xmin
            else:
                point = np.array([xmax*pixel_step, ymin*pixel_step, self.founda_height])
                xmiddle = update_founda_v(self.roof_normals[i,3], point, ymin, self.roof_heights[i])
                self.founda_positions[i,2] = 2.0*xmiddle-xmax
        elif btype == H_HIP:
            if founda_updated[i,0]==True and founda_updated[i,1]==True:
                pass
                #print(str(H_HIP)+": both top and bottom boundaries are fixed")
            elif founda_updated[i,0]==True and founda_updated[i,1]==False:
                point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
                ymiddle = update_founda_h(self.roof_normals[i,0], point, xmin, self.roof_heights[i])
                self.founda_positions[i,1] = 2.0*ymiddle-ymin
            else:
                point = np.array([xmin*pixel_step, ymax*pixel_step, self.founda_height])
                ymiddle = update_founda_h(self.roof_normals[i,1], point, xmin, self.roof_heights[i])
                self.founda_positions[i,0] = 2.0*ymiddle-ymax
                
            point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
            xtmp = update_ratio_h(self.roof_normals[i,2], point, ymin, self.roof_heights[i])
            self.roof_ratios[i] = max(0,min(0.499,(xtmp-xmin)/(xmax-xmin)))
        elif btype == V_HIP:
            if founda_updated[i,2]==True and founda_updated[i,3]==True:
                pass
                #print(str(V_HIP)+": both left and right boundaries are fixed")
            elif founda_updated[i,2]==True and founda_updated[i,3]==False:
                point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
                xmiddle = update_founda_v(self.roof_normals[i,2], point, ymin, self.roof_heights[i])
                self.founda_positions[i,3] = 2.0*xmiddle-xmin
            else:
                point = np.array([xmax*pixel_step, ymin*pixel_step, self.founda_height])
                xmiddle = update_founda_v(self.roof_normals[i,3], point, ymin, self.roof_heights[i])
                self.founda_positions[i,2] = 2.0*xmiddle-xmax
                
            point = np.array([xmin*pixel_step, ymin*pixel_step, self.founda_height])
            ytmp = update_ratio_v(self.roof_normals[i,0], point, xmin, self.roof_heights[i])
            self.roof_ratios[i] = max(0,min(0.499,(ytmp-ymin)/(ymax-ymin)))

    def save_to_mesh(self, filename):
        max_height = np.amax(self.roof_heights)
        max_height = max(self.founda_height, max_height)
        vertices = ""
        faces = ""
        for i, btype in enumerate(self.block_types):  
            ymin = self.founda_positions[i,0]
            ymax = self.founda_positions[i,1]
            xmin = self.founda_positions[i,2]
            xmax = self.founda_positions[i,3]

            rgb_g = get_obj_color(0., max_height, 0.)
            rgb_f = get_obj_color(self.founda_height, max_height, 0.)
            vertices += v_string(xmin*pixel_step, (IMG_SIZE-ymin)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                    + v_string(xmin*pixel_step, (IMG_SIZE-ymin)*pixel_step, self.founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                    + v_string(xmax*pixel_step, (IMG_SIZE-ymin)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                    + v_string(xmax*pixel_step, (IMG_SIZE-ymin)*pixel_step, self.founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                    + v_string(xmax*pixel_step, (IMG_SIZE-ymax)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                    + v_string(xmax*pixel_step, (IMG_SIZE-ymax)*pixel_step, self.founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                    + v_string(xmin*pixel_step, (IMG_SIZE-ymax)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                    + v_string(xmin*pixel_step, (IMG_SIZE-ymax)*pixel_step, self.founda_height, rgb_f[0], rgb_f[1], rgb_f[2])
            ymiddle = (ymin+ymax)/2.0
            xmiddle = (xmin+xmax)/2.0
            xgap = (xmax-xmin)*self.roof_ratios[i]
            ygap = (ymax-ymin)*self.roof_ratios[i]
            fstart = 10*i
            rgb_r = get_obj_color(self.roof_heights[i], max_height, 0.)
            if btype == H_GABLE: #horizontal gable
                vertices += v_string(xmin*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                         + v_string(xmax*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2])
                faces += f_string(fstart, 1,2,4,3) \
                      + f_string(fstart, 3,4,6,5) \
                      + f_string(fstart, 5,6,8,7) \
                      + f_string(fstart, 1,7,8,2) \
                      + f_string(fstart, 2,9,10,4) \
                      + f_string(fstart, 9,8,6,10) \
                      + f_string3(fstart, 2,8,9) \
                      + f_string3(fstart, 10,6,4)
            elif btype == V_GABLE: #vertical gable
                vertices += v_string(xmiddle*pixel_step, (IMG_SIZE-ymin)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                         + v_string(xmiddle*pixel_step, (IMG_SIZE-ymax)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2])
                faces += f_string(fstart, 1,2,4,3) \
                      + f_string(fstart, 3,4,6,5) \
                      + f_string(fstart, 5,6,8,7) \
                      +f_string(fstart, 1,7,8,2) \
                      + f_string3(fstart, 2,9,4) \
                      + f_string3(fstart, 10,8,6) \
                      + f_string(fstart, 2,8,10,9) \
                      + f_string(fstart, 9,10,6,4)
            elif btype == H_HIP: #horizontal hip
                vertices += v_string((xmin+xgap)*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                         + v_string((xmax-xgap)*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) 
                faces += f_string(fstart, 1,2,4,3) \
                      + f_string(fstart, 3,4,6,5) \
                      + f_string(fstart, 5,6,8,7) \
                      + f_string(fstart, 1,7,8,2) \
                      + f_string(fstart, 2,9,10,4) \
                      + f_string(fstart, 9,8,6,10) \
                      + f_string3(fstart, 2,8,9) \
                      + f_string3(fstart, 10,6,4)
            elif btype == V_HIP: #vertical hip
                vertices += v_string(xmiddle*pixel_step, (IMG_SIZE-(ymin+ygap))*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                         + v_string(xmiddle*pixel_step, (IMG_SIZE-(ymax-ygap))*pixel_step, self.roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) 
                faces += f_string(fstart, 1,2,4,3) \
                      + f_string(fstart, 3,4,6,5) \
                      + f_string(fstart, 5,6,8,7) \
                      + f_string(fstart, 1,7,8,2) \
                      + f_string3(fstart, 2,9,4) \
                      + f_string3(fstart, 10,8,6) \
                      + f_string(fstart, 2,8,10,9) \
                      + f_string(fstart, 9,10,6,4)
            else:
                raise NotImplementedError
        with open(filename, 'w') as f:
            f.write(vertices)
            f.write(faces)
    def rasterize_hgable(self, idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal):
        pixelnum = len(ys)
        ymin = founda_pos[0]
        ymax = founda_pos[1]
        xmin = founda_pos[2]
        xmax = founda_pos[3]
        ymiddle = (ymin+ymax)/2.
        # top
        height1 = (ys-ymin)/(ymiddle-ymin)*(roof_height-founda_height) + founda_height
        # bottom
        height2 = (ys-ymax)/(ymiddle-ymax)*(roof_height-founda_height) + founda_height
        heights = np.stack([height1,height2],1)
        minid = np.argmin(heights, axis=1)
        final_height = heights[np.arange(pixelnum),minid]
        final_normal = roof_normal[minid,:]

        tmp = (ys<ymin+EPISON) | (ys>ymax-EPISON) | (xs<xmin+EPISON) | (xs>xmax-EPISON)
        final_height[tmp] = founda_height
        final_normal[tmp,:] = 1.0

        face_colors = np.arange(4) + idx*4
        face_label = face_colors[minid]
        face_label[tmp] = -1
        return final_height, final_normal, face_label

    def rasterize_vgable(self, idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal):
        pixelnum = len(ys)
        ymin = founda_pos[0]
        ymax = founda_pos[1]
        xmin = founda_pos[2]
        xmax = founda_pos[3]
        xmiddle = (xmin+xmax)/2.
        # left
        height1 = (xs-xmin)/(xmiddle-xmin)*(roof_height-founda_height) + founda_height
        # right
        height2 = (xs-xmax)/(xmiddle-xmax)*(roof_height-founda_height) + founda_height
        heights = np.stack([height1,height2],1)
        minid = np.argmin(heights, axis=1)
        final_height = heights[np.arange(pixelnum),minid]
        final_normal = roof_normal[minid+2,:]

        tmp = (ys<ymin+EPISON) | (ys>ymax-EPISON) | (xs<xmin+EPISON) | (xs>xmax-EPISON)
        final_height[tmp] = founda_height
        final_normal[tmp,:] = 1.0

        face_colors = np.arange(4) + idx*4
        face_label = face_colors[minid+2]
        face_label[tmp] = -1
        return final_height, final_normal, face_label

    def rasterize_hhip(self, idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal, roof_ratio):
        pixelnum = len(ys)
        ymin = founda_pos[0]
        ymax = founda_pos[1]
        xmin = founda_pos[2]
        xmax = founda_pos[3]
        xmiddle = (xmin+xmax)/2.
        ymiddle = (ymin+ymax)/2.
        height_top = (ys-ymin)/(ymiddle-ymin)*(roof_height-founda_height) + founda_height
        height_bottom = (ys-ymax)/(ymiddle-ymax)*(roof_height-founda_height) + founda_height
        
        gap = roof_ratio*(xmax-xmin)
        left = min(xmin+gap, xmiddle)
        right = max(xmax-gap, xmiddle)
        height_left = (xs-xmin)/(left-xmin)*(roof_height-founda_height) + founda_height
        height_right = (xs-xmax)/(right-xmax)*(roof_height-founda_height) + founda_height
        heights = np.stack([height_top,height_bottom,height_left,height_right],1)
        minid = np.argmin(heights, axis=1)
        final_height = heights[np.arange(pixelnum),minid]
        final_normal = roof_normal[minid,:]

        tmp = (ys<ymin+EPISON) | (ys>ymax-EPISON) | (xs<xmin+EPISON) | (xs>xmax-EPISON)
        final_height[tmp] = founda_height
        final_normal[tmp,:] = 1.0

        face_colors = np.arange(4) + idx*4
        face_label = face_colors[minid]
        face_label[tmp] = -1
        return final_height, final_normal, face_label

    def rasterize_vhip(self, idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal, roof_ratio):
        pixelnum = len(ys)
        ymin = founda_pos[0]
        ymax = founda_pos[1]
        xmin = founda_pos[2]
        xmax = founda_pos[3]
        xmiddle = (xmin+xmax)/2.
        ymiddle = (ymin+ymax)/2.
        height_left = (xs-xmin)/(xmiddle-xmin)*(roof_height-founda_height) + founda_height
        height_right = (xs-xmax)/(xmiddle-xmax)*(roof_height-founda_height) + founda_height

        gap = roof_ratio*(ymax-ymin)
        top = min(ymin+gap, ymiddle)
        bottom = max(ymax-gap, ymiddle)
        height_top = (ys-ymin)/(top-ymin)*(roof_height-founda_height) + founda_height
        height_bottom = (ys-ymax)/(bottom-ymax)*(roof_height-founda_height) + founda_height
        heights = np.stack([height_top,height_bottom,height_left,height_right],1)
        minid = np.argmin(heights, axis=1)
        final_height = heights[np.arange(pixelnum),minid]
        final_normal = roof_normal[minid,:]

        tmp = (ys<ymin+EPISON) | (ys>ymax-EPISON) | (xs<xmin+EPISON) | (xs>xmax-EPISON)
        final_height[tmp] = founda_height
        final_normal[tmp,:] = 1.0

        face_colors = np.arange(4) + idx*4
        face_label = face_colors[minid]
        face_label[tmp] = -1
        return final_height, final_normal, face_label

    def rasterize_block(self, i):
        tmp = np.arange(64)
        ys = np.repeat(tmp, 64)
        xs = np.tile(tmp, 64)

        roof_angle_map = np.zeros((64,64))
        btype = self.block_types[i]
        if btype == H_GABLE:
            height_tmp, normal_tmp, label_tmp = self.rasterize_hgable(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i])
        elif btype == V_GABLE:
            height_tmp, normal_tmp, label_tmp = self.rasterize_vgable(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i])
        elif btype == H_HIP:
            height_tmp, normal_tmp, label_tmp = self.rasterize_hhip(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i], self.roof_ratios[i])
        elif btype == V_HIP:
            height_tmp, normal_tmp, label_tmp = self.rasterize_vhip(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i], self.roof_ratios[i])
        else:
            raise NotImplementedError
        label_tmp = label_tmp - i*4
        label_tmp = label_tmp.reshape(64,64)
        tb_mask = np.logical_or(label_tmp==0, label_tmp==1)
        lr_mask = np.logical_or(label_tmp==2, label_tmp==3)
        roof_angle_map[tb_mask] = self.roof_angles[i,0]
        roof_angle_map[lr_mask] = self.roof_angles[i,1]
        label_onehot = np.zeros((64,64,3))
        label_onehot[tb_mask,0] = 1
        label_onehot[lr_mask,1] = 1
        label_onehot[label_tmp<0,2] = 1
        return label_onehot, roof_angle_map

        colors=np.array([0,0,1,1,0,0,1,0,1,0,1,0]).reshape(4,3)
        a = np.zeros((64,64,3))
        for j in range(4):
            a[label_tmp==j] = colors[j,:]
        return label_onehot, roof_angle_map, a*255

    def rasterize_house(self, scale_flag=True):
        if scale_flag:
            self.scaling_house()
        tmp = np.arange(64)
        ys = np.repeat(tmp, 64)
        xs = np.tile(tmp, 64)

        height_map, normal_map, label_map = [], [], []
        for i, btype in enumerate(self.block_types):
            if btype == H_GABLE:
                height_tmp, normal_tmp, label_tmp = self.rasterize_hgable(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i])
            elif btype == V_GABLE:
                height_tmp, normal_tmp, label_tmp = self.rasterize_vgable(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i])
            elif btype == H_HIP:
                height_tmp, normal_tmp, label_tmp = self.rasterize_hhip(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i], self.roof_ratios[i])
            elif btype == V_HIP:
                height_tmp, normal_tmp, label_tmp = self.rasterize_vhip(i, ys, xs, self.founda_positions[i], self.founda_height, self.roof_heights[i], self.roof_normals[i], self.roof_ratios[i])
            else:
                raise NotImplementedError
            height_map.append(height_tmp)
            normal_map.append(normal_tmp)
            label_map.append(label_tmp)
        
        
        height_map = np.stack(height_map)
        normal_map = np.stack(normal_map)
        label_map = np.stack(label_map)


        minid = np.argmax(height_map, axis=0)
        final_height = height_map[minid,np.arange(4096)]-self.founda_height
        final_height = final_height.reshape(64,64)
        final_normal = normal_map[minid,np.arange(4096),:]
        final_normal = (final_normal+1.)/2.
        final_normal = final_normal.reshape(64,64,3)*255.
        final_normal = final_normal[:, :, ::-1]
        
        face_labelmap = label_map[minid,np.arange(4096)]
        face_labelmap = self.combine_faces(face_labelmap, self.roof_graph)
        face_color = self.face_label2color(face_labelmap)
        face_color = face_color.reshape(64,64,3)*255.
        return final_height, self.face_label2mask(face_labelmap.reshape(64,64)), face_color, final_normal

    def scaling_house(self):
        max_len = 63.5
        top_bound = np.amin(self.founda_positions[:,0])
        left_bound = np.amin(self.founda_positions[:,2])
        bottom_bound = np.amax(self.founda_positions[:,1])
        right_bound = np.amax(self.founda_positions[:,3])
        if (bottom_bound-top_bound) > (right_bound-left_bound):
            self.founda_positions[:,0:2] = (self.founda_positions[:,0:2]-top_bound)/(bottom_bound-top_bound)*max_len
            scaled_len = (right_bound-left_bound)*max_len/(bottom_bound-top_bound)
            self.founda_positions[:,2:4] = (self.founda_positions[:,2:4]-left_bound)/(right_bound-left_bound)*scaled_len+(max_len-scaled_len)/2.0
        else:
            self.founda_positions[:,2:4] = (self.founda_positions[:,2:4]-left_bound)/(right_bound-left_bound)*max_len
            scaled_len = (bottom_bound-top_bound)*max_len/(right_bound-left_bound)
            self.founda_positions[:,0:2] = (self.founda_positions[:,0:2]-top_bound)/(bottom_bound-top_bound)*scaled_len+(max_len-scaled_len)/2.0
        self.normal_to_heightratio()

    def build_roof_graph(self):
        roof_graph = np.full_like(self.founda_positions, -1, dtype=np.int)
        for i in range(1, self.num_blocks):
            for f in range(4):
                for j in range(i):
                    if np.dot(self.roof_normals[i,f],self.roof_normals[j,f])>T_angle and abs(self.founda_positions[i,f]-self.founda_positions[j,f])<T_boundary:
                        roof_graph[i,f] = j*4+f
                        break
        return roof_graph

    def combine_faces(self, face_label, roof_graph):
        face_label_new = face_label.copy()
        for i in range(0, roof_graph.shape[0]):
            for f in range(4):
                if roof_graph[i,f]>=0:
                   tmp = face_label == (i*4+f)
                   face_label_new[tmp] = roof_graph[i,f]
        return face_label_new

    def face_label2color(self, face_label):
        maxid = np.amax(face_label)
        colorsets = np.random.random((maxid+2,3))
        colorsets[-1,:] = 1.
        return colorsets[face_label,:]
    def face_label2mask(self, face_label):
        ulabels = np.unique(face_label)
        masks = []
        for i in ulabels:
            if i<0:
                continue
            masks.append(face_label==i)
        return np.asarray(masks)

# utility functions
def update_height(normal, pt, xi, yi):
    offset = np.dot(normal, pt)
    return (offset-normal[0]*xi*pixel_step-normal[1]*yi*pixel_step)/normal[2]
def update_founda_h(normal, pt, xi, height):
    offset = np.dot(normal, pt)
    return (offset-normal[0]*xi*pixel_step-normal[2]*height)/normal[1]/pixel_step
def update_founda_v(normal, pt, yi, height):
    offset = np.dot(normal, pt)
    return (offset-normal[1]*yi*pixel_step-normal[2]*height)/normal[0]/pixel_step
def update_ratio_h(normal, pt, yi, height):
    offset = np.dot(normal, pt)
    return (offset-normal[1]*yi*pixel_step-normal[2]*height)/normal[0]/pixel_step
def update_ratio_v(normal, pt, xi, height):
    offset = np.dot(normal, pt)
    return (offset-normal[0]*xi*pixel_step-normal[2]*height)/normal[1]/pixel_step
def symmetry_faceid(i):
    if i==0:
        return 1
    elif i==1:
        return 0
    elif i==2:
        return 3
    elif i==3:
        return 2
    else:
        raise NotImplementedError
def v_string(x, y, z, r, g, b):
    return "v "+str(x)+" "+str(y)+" "+str(z)+" "+str(r)+" "+str(g)+" "+str(b)+"\n"
def f_string3(s, f1, f2, f3):
    return "f "+str(f1+s)+" "+str(f2+s)+" "+str(f3+s)+"\n"
def f_string(s, f1, f2, f3, f4):
    return "f "+str(f1+s)+" "+str(f2+s)+" "+str(f3+s)+" "+str(f4+s)+"\n"
def get_obj_color(val, max_val, min_val):
    val = max(val, EPISON)
    idx = int((val-min_val)/(max_val-min_val+1e-4)*255.)*3
    # if len(obj_colormap[idx:idx+3])<3:
    #     print("idx= "+str(idx))
    #     print(val, max_val, min_val)
    return obj_colormap[idx:idx+3]
obj_colormap = np.array(
    [0,0,0.5156,
    0,0,0.5312,
    0,0,0.5469,
    0,0,0.5625,
    0,0,0.5781,
    0,0,0.5938,
    0,0,0.6094,
    0,0,0.625,
    0,0,0.6406,
    0,0,0.6562,
    0,0,0.6719,
    0,0,0.6875,
    0,0,0.7031,
    0,0,0.7188,
    0,0,0.7344,
    0,0,0.75,
    0,0,0.7656,
    0,0,0.7812,
    0,0,0.7969,
    0,0,0.8125,
    0,0,0.8281,
    0,0,0.8438,
    0,0,0.8594,
    0,0,0.875,
    0,0,0.8906,
    0,0,0.9062,
    0,0,0.9219,
    0,0,0.9375,
    0,0,0.9531,
    0,0,0.9688,
    0,0,0.9844,
    0,0,1,
    0,0.0156,1,
    0,0.0312,1,
    0,0.0469,1,
    0,0.0625,1,
    0,0.0781,1,
    0,0.0938,1,
    0,0.1094,1,
    0,0.125,1,
    0,0.1406,1,
    0,0.1562,1,
    0,0.1719,1,
    0,0.1875,1,
    0,0.2031,1,
    0,0.2188,1,
    0,0.2344,1,
    0,0.25,1,
    0,0.2656,1,
    0,0.2812,1,
    0,0.2969,1,
    0,0.3125,1,
    0,0.3281,1,
    0,0.3438,1,
    0,0.3594,1,
    0,0.375,1,
    0,0.3906,1,
    0,0.4062,1,
    0,0.4219,1,
    0,0.4375,1,
    0,0.4531,1,
    0,0.4688,1,
    0,0.4844,1,
    0,0.5,1,
    0,0.5156,1,
    0,0.5312,1,
    0,0.5469,1,
    0,0.5625,1,
    0,0.5781,1,
    0,0.5938,1,
    0,0.6094,1,
    0,0.625,1,
    0,0.6406,1,
    0,0.6562,1,
    0,0.6719,1,
    0,0.6875,1,
    0,0.7031,1,
    0,0.7188,1,
    0,0.7344,1,
    0,0.75,1,
    0,0.7656,1,
    0,0.7812,1,
    0,0.7969,1,
    0,0.8125,1,
    0,0.8281,1,
    0,0.8438,1,
    0,0.8594,1,
    0,0.875,1,
    0,0.8906,1,
    0,0.9062,1,
    0,0.9219,1,
    0,0.9375,1,
    0,0.9531,1,
    0,0.9688,1,
    0,0.9844,1,
    0,1,1,
    0.0156,1,0.9844,
    0.0312,1,0.9688,
    0.0469,1,0.9531,
    0.0625,1,0.9375,
    0.0781,1,0.9219,
    0.0938,1,0.9062,
    0.1094,1,0.8906,
    0.125,1,0.875,
    0.1406,1,0.8594,
    0.1562,1,0.8438,
    0.1719,1,0.8281,
    0.1875,1,0.8125,
    0.2031,1,0.7969,
    0.2188,1,0.7812,
    0.2344,1,0.7656,
    0.25,1,0.75,
    0.2656,1,0.7344,
    0.2812,1,0.7188,
    0.2969,1,0.7031,
    0.3125,1,0.6875,
    0.3281,1,0.6719,
    0.3438,1,0.6562,
    0.3594,1,0.6406,
    0.375,1,0.625,
    0.3906,1,0.6094,
    0.4062,1,0.5938,
    0.4219,1,0.5781,
    0.4375,1,0.5625,
    0.4531,1,0.5469,
    0.4688,1,0.5312,
    0.4844,1,0.5156,
    0.5,1,0.5,
    0.5156,1,0.4844,
    0.5312,1,0.4688,
    0.5469,1,0.4531,
    0.5625,1,0.4375,
    0.5781,1,0.4219,
    0.5938,1,0.4062,
    0.6094,1,0.3906,
    0.625,1,0.375,
    0.6406,1,0.3594,
    0.6562,1,0.3438,
    0.6719,1,0.3281,
    0.6875,1,0.3125,
    0.7031,1,0.2969,
    0.7188,1,0.2812,
    0.7344,1,0.2656,
    0.75,1,0.25,
    0.7656,1,0.2344,
    0.7812,1,0.2188,
    0.7969,1,0.2031,
    0.8125,1,0.1875,
    0.8281,1,0.1719,
    0.8438,1,0.1562,
    0.8594,1,0.1406,
    0.875,1,0.125,
    0.8906,1,0.1094,
    0.9062,1,0.0938,
    0.9219,1,0.0781,
    0.9375,1,0.0625,
    0.9531,1,0.0469,
    0.9688,1,0.0312,
    0.9844,1,0.0156,
    1,1,0,
    1,0.9844,0,
    1,0.9688,0,
    1,0.9531,0,
    1,0.9375,0,
    1,0.9219,0,
    1,0.9062,0,
    1,0.8906,0,
    1,0.875,0,
    1,0.8594,0,
    1,0.8438,0,
    1,0.8281,0,
    1,0.8125,0,
    1,0.7969,0,
    1,0.7812,0,
    1,0.7656,0,
    1,0.75,0,
    1,0.7344,0,
    1,0.7188,0,
    1,0.7031,0,
    1,0.6875,0,
    1,0.6719,0,
    1,0.6562,0,
    1,0.6406,0,
    1,0.625,0,
    1,0.6094,0,
    1,0.5938,0,
    1,0.5781,0,
    1,0.5625,0,
    1,0.5469,0,
    1,0.5312,0,
    1,0.5156,0,
    1,0.5,0,
    1,0.4844,0,
    1,0.4688,0,
    1,0.4531,0,
    1,0.4375,0,
    1,0.4219,0,
    1,0.4062,0,
    1,0.3906,0,
    1,0.375,0,
    1,0.3594,0,
    1,0.3438,0,
    1,0.3281,0,
    1,0.3125,0,
    1,0.2969,0,
    1,0.2812,0,
    1,0.2656,0,
    1,0.25,0,
    1,0.2344,0,
    1,0.2188,0,
    1,0.2031,0,
    1,0.1875,0,
    1,0.1719,0,
    1,0.1562,0,
    1,0.1406,0,
    1,0.125,0,
    1,0.1094,0,
    1,0.0938,0,
    1,0.0781,0,
    1,0.0625,0,
    1,0.0469,0,
    1,0.0312,0,
    1,0.0156,0,
    1,0,0,
    0.9844,0,0,
    0.9688,0,0,
    0.9531,0,0,
    0.9375,0,0,
    0.9219,0,0,
    0.9062,0,0,
    0.8906,0,0,
    0.875,0,0,
    0.8594,0,0,
    0.8438,0,0,
    0.8281,0,0,
    0.8125,0,0,
    0.7969,0,0,
    0.7812,0,0,
    0.7656,0,0,
    0.75,0,0,
    0.7344,0,0,
    0.7188,0,0,
    0.7031,0,0,
    0.6875,0,0,
    0.6719,0,0,
    0.6562,0,0,
    0.6406,0,0,
    0.625,0,0,
    0.6094,0,0,
    0.5938,0,0,
    0.5781,0,0,
    0.5625,0,0,
    0.5469,0,0,
    0.5312,0,0,
    0.5156,0,0,
    0.5,0,0])