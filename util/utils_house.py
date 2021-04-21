import numpy as np
import math

H_GABLE = 0
V_GABLE = 1
H_HIP = 2
V_HIP = 3
pixel_step = 0.5
IMG_SIZE = 64

def angle_to_normal(block_types, roof_angles):
    roof_normals = []
    for i, btype in enumerate(block_types):
        roof_normal = np.zeros((4,3))
        tb_angle = roof_angles[i,0] # top bottom
        tb_angle_sin, tb_angle_cos = math.sin(tb_angle), math.cos(tb_angle)
        lr_angle = roof_angles[i,1] # left right
        lr_angle_sin, lr_angle_cos = math.sin(lr_angle), math.cos(lr_angle)
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
    return np.asarray(roof_normals)

def normal_to_heightratio(block_types, founda_positions, founda_height, roof_normals):
    roof_heights = np.zeros(block_types.shape)
    roof_ratios = np.zeros(block_types.shape)
    for i, btype in enumerate(block_types):
        ymin = founda_positions[i,0]
        ymax = founda_positions[i,1]
        xmin = founda_positions[i,2]
        xmax = founda_positions[i,3]
        ymiddle = (ymin+ymax)/2.0
        xmiddle = (xmin+xmax)/2.0
        point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
        if btype == H_GABLE:
            roof_heights[i] = update_height(roof_normals[i,0], point, xmin, ymiddle)
        elif btype == V_GABLE:
            roof_heights[i] = update_height(roof_normals[i,2], point, xmiddle, ymin)
        elif btype == H_HIP:
            roof_heights[i] = update_height(roof_normals[i,0], point, xmin, ymiddle)
            xtmp = update_ratio_h(roof_normals[i,2], point, xmin, roof_heights[i])
            roof_ratios[i] = (xtmp-xmin)/(xmax-xmin)
        elif btype == V_HIP:
            roof_heights[i] = update_height(roof_normals[i,2], point, xmiddle, ymin)
            ytmp = update_ratio_v(roof_normals[i,0], point, xmin, roof_heights[i])
            roof_ratios[i] = (ytmp-ymin)/(ymax-ymin)
        else:
            print('block type not supported')
            exit()
    return roof_heights, roof_ratios

def sequential_snap(block_types, founda_positions, founda_height, roof_normals, roof_heights, roof_ratios):
    T_angle, T_boundary, T_middle, T_height = 0.94, 1., 2., 0.07
    roof_indicators = np.full_like(founda_positions, True, dtype=np.bool)
    roof_updated = np.full_like(founda_positions, False, dtype=np.bool)
    founda_updated = np.full_like(founda_positions, False, dtype=np.bool)
    for i, btype in enumerate(block_types):
        if btype==H_GABLE:
            roof_indicators[i,2] = False
            roof_indicators[i,3] = False
        elif btype==V_GABLE:
            roof_indicators[i,0] = False
            roof_indicators[i,1] = False

    middles = np.full_like(founda_positions, -1.)
    for i, btype in enumerate(block_types):
        if btype%2==0:
            middles[i,0] = (founda_positions[i,0]+founda_positions[i,1])/2.
            middles[i,1] = middles[i,0]
        else:
            middles[i,2] = (founda_positions[i,2]+founda_positions[i,3])/2.
            middles[i,3] = middles[i,2]
    roof_graph = np.full_like(founda_positions, -1, dtype=np.int)
    for i, btype in enumerate(block_types):
        for f in range(4):
            symid = symmetry_faceid(f)
            for j in range(i):
                mid_pos = (founda_positions[j,f] + founda_positions[j,symid])/2.
                if abs(founda_positions[i,f]-founda_positions[j,f])<T_boundary:
                    founda_positions[i,f] = founda_positions[j,f]
                    founda_updated[i,f] = True
                    if abs(roof_heights[i]-roof_heights[j])/roof_heights[i] < T_height:
                        roof_heights[i] = roof_heights[j]
                elif middles[j,f]>0.0 and abs(founda_positions[i,f]-middles[j,f])<T_middle and ((btype+block_types[j])%2)!=0:
                    founda_positions[i,f] = mid_pos
                    founda_updated[i,f] = True
                    if abs(roof_heights[i]-roof_heights[j])/roof_heights[i] < T_height:
                        roof_heights[i] = roof_heights[j]
                if roof_indicators[j,f]==False or roof_indicators[i,f]==False:
                    continue
            
                if np.dot(roof_normals[i,f],roof_normals[j,f])>T_angle and abs(founda_positions[i,f]-founda_positions[j,f])<T_boundary:
                    roof_normals[i,f] = roof_normals[j,f].copy()
                    founda_positions[i,f] = founda_positions[j,f]
                    roof_normals[i,symid] = roof_normals[j,symid].copy()
                    roof_updated[i,f] = True
                    roof_updated[i,symid] = True
                    roof_graph[i,f] = j*4+f
                    #roof_graph[i,symid] = j*4+symid
        normal_to_foundaratio(i, btype, founda_positions, founda_updated, founda_height, roof_normals, roof_heights, roof_ratios)
    return roof_graph

def normal_to_foundaratio(i, btype, founda_positions, founda_updated, founda_height, roof_normals, roof_heights, roof_ratios):
    ymin = founda_positions[i,0]
    ymax = founda_positions[i,1]
    xmin = founda_positions[i,2]
    xmax = founda_positions[i,3]
    ymiddle = (ymin+ymax)/2.
    xmiddle = (xmin+xmax)/2.

    if btype == H_GABLE:
        if founda_updated[i,0]==True and founda_updated[i,1]==True:
            print(str(H_GABLE)+": both top and bottom boundaries are fixed")
        elif founda_updated[i,0]==True and founda_updated[i,1]==False:
            point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
            ymiddle = update_founda_h(roof_normals[i,0], point, xmin, roof_heights[i])
            founda_positions[i,1] = 2.0*ymiddle-ymin
        else:
            point = np.array([xmin*pixel_step, ymax*pixel_step, founda_height])
            ymiddle = update_founda_h(roof_normals[i,1], point, xmin, roof_heights[i])
            founda_positions[i,0] = 2.0*ymiddle-ymax

    elif btype == V_GABLE:
        if founda_updated[i,2]==True and founda_updated[i,3]==True:
            print(str(V_GABLE)+": both left and right boundaries are fixed")
        elif founda_updated[i,2]==True and founda_updated[i,3]==False:
            point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
            xmiddle = update_founda_v(roof_normals[i,2], point, ymin, roof_heights[i])
            founda_positions[i,3] = 2.0*xmiddle-xmin
        else:
            point = np.array([xmax*pixel_step, ymin*pixel_step, founda_height])
            xmiddle = update_founda_v(roof_normals[i,3], point, ymin, roof_heights[i])
            founda_positions[i,2] = 2.0*xmiddle-xmax
    elif btype == H_HIP:
        if founda_updated[i,0]==True and founda_updated[i,1]==True:
            print(str(H_HIP)+": both top and bottom boundaries are fixed")
        elif founda_updated[i,0]==True and founda_updated[i,1]==False:
            point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
            ymiddle = update_founda_h(roof_normals[i,0], point, xmin, roof_heights[i])
            founda_positions[i,1] = 2.0*ymiddle-ymin
        else:
            point = np.array([xmin*pixel_step, ymax*pixel_step, founda_height])
            ymiddle = update_founda_h(roof_normals[i,1], point, xmin, roof_heights[i])
            founda_positions[i,0] = 2.0*ymiddle-ymax
            
        point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
        xtmp = update_ratio_h(roof_normals[i,2], point, ymin, roof_heights[i])
        roof_ratios[i] = (xtmp-xmin)/(xmax-xmin)
    elif btype == V_HIP:
        if founda_updated[i,2]==True and founda_updated[i,3]==True:
            print(str(V_HIP)+": both left and right boundaries are fixed")
        elif founda_updated[i,2]==True and founda_updated[i,3]==False:
              point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
              xmiddle = update_founda_v(roof_normals[i,2], point, ymin, roof_heights[i])
              founda_positions[i,3] = 2.0*xmiddle-xmin
        else:
            point = np.array([xmax*pixel_step, ymin*pixel_step, founda_height])
            xmiddle = update_founda_v(roof_normals[i,3], point, ymin, roof_heights[i])
            founda_positions[i,2] = 2.0*xmiddle-xmax
            
        point = np.array([xmin*pixel_step, ymin*pixel_step, founda_height])
        ytmp = update_ratio_v(roof_normals[i,0], point, xmin, roof_heights[i])
        roof_ratios[i] = (ytmp-ymin)/(ymax-ymin)

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
def save_to_mesh(filename, block_types, founda_positions, founda_height, roof_heights, roof_ratios):
    max_height = np.amax(roof_heights)
    max_height = max(founda_height, max_height)
    vertices = ""
    faces = ""
    for i, btype in enumerate(block_types):  
        ymin = founda_positions[i,0]
        ymax = founda_positions[i,1]
        xmin = founda_positions[i,2]
        xmax = founda_positions[i,3]

        rgb_g = get_obj_color(0., max_height, 0.)
        rgb_f = get_obj_color(founda_height, max_height, 0.)
        vertices += v_string(xmin*pixel_step, (IMG_SIZE-ymin)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                + v_string(xmin*pixel_step, (IMG_SIZE-ymin)*pixel_step, founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                + v_string(xmax*pixel_step, (IMG_SIZE-ymin)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                + v_string(xmax*pixel_step, (IMG_SIZE-ymin)*pixel_step, founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                + v_string(xmax*pixel_step, (IMG_SIZE-ymax)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                + v_string(xmax*pixel_step, (IMG_SIZE-ymax)*pixel_step, founda_height, rgb_f[0], rgb_f[1], rgb_f[2]) \
                + v_string(xmin*pixel_step, (IMG_SIZE-ymax)*pixel_step, 0., rgb_g[0], rgb_g[1], rgb_g[2]) \
                + v_string(xmin*pixel_step, (IMG_SIZE-ymax)*pixel_step, founda_height, rgb_f[0], rgb_f[1], rgb_f[2])
        ymiddle = (ymin+ymax)/2.0
        xmiddle = (xmin+xmax)/2.0
        xgap = (xmax-xmin)*roof_ratios[i]
        ygap = (ymax-ymin)*roof_ratios[i]
        fstart = 10*i
        rgb_r = get_obj_color(roof_heights[i], max_height, 0.)
        if btype == H_GABLE: #horizontal gable
            vertices += v_string(xmin*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                     + v_string(xmax*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2])
            faces += f_string(fstart, 1,2,4,3) \
                  + f_string(fstart, 3,4,6,5) \
                  + f_string(fstart, 5,6,8,7) \
                  + f_string(fstart, 1,7,8,2) \
                  + f_string(fstart, 2,9,10,4) \
                  + f_string(fstart, 9,8,6,10) \
                  + f_string3(fstart, 2,8,9) \
                  + f_string3(fstart, 10,6,4)
        elif btype == V_GABLE: #vertical gable
            vertices += v_string(xmiddle*pixel_step, (IMG_SIZE-ymin)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                     + v_string(xmiddle*pixel_step, (IMG_SIZE-ymax)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2])
            faces += f_string(fstart, 1,2,4,3) \
                  + f_string(fstart, 3,4,6,5) \
                  + f_string(fstart, 5,6,8,7) \
                  +f_string(fstart, 1,7,8,2) \
                  + f_string3(fstart, 2,9,4) \
                  + f_string3(fstart, 10,8,6) \
                  + f_string(fstart, 2,8,10,9) \
                  + f_string(fstart, 9,10,6,4)
        elif btype == H_HIP: #horizontal hip
            vertices += v_string((xmin+xgap)*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                     + v_string((xmax-xgap)*pixel_step, (IMG_SIZE-ymiddle)*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) 
            faces += f_string(fstart, 1,2,4,3) \
                  + f_string(fstart, 3,4,6,5) \
                  + f_string(fstart, 5,6,8,7) \
                  + f_string(fstart, 1,7,8,2) \
                  + f_string(fstart, 2,9,10,4) \
                  + f_string(fstart, 9,8,6,10) \
                  + f_string3(fstart, 2,8,9) \
                  + f_string3(fstart, 10,6,4)
        elif btype == V_HIP: #vertical hip
            vertices += v_string(xmiddle*pixel_step, (IMG_SIZE-(ymin+ygap))*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) \
                     + v_string(xmiddle*pixel_step, (IMG_SIZE-(ymax-ygap))*pixel_step, roof_heights[i], rgb_r[0], rgb_r[1], rgb_r[2]) 
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

def v_string(x, y, z, r, g, b):
    return "v "+str(x)+" "+str(y)+" "+str(z)+" "+str(r)+" "+str(g)+" "+str(b)+"\n"
def f_string3(s, f1, f2, f3):
    return "f "+str(f1+s)+" "+str(f2+s)+" "+str(f3+s)+"\n"
def f_string(s, f1, f2, f3, f4):
    return "f "+str(f1+s)+" "+str(f2+s)+" "+str(f3+s)+" "+str(f4+s)+"\n"
def get_obj_color(val, max_val, min_val):
    val = max(val, 1e-4)
    idx = int((val-min_val)/(max_val-min_val)*255.)*3
    # if len(obj_colormap[idx:idx+3])<3:
    #     print("idx= "+str(idx))
    #     print(val, max_val, min_val)
    return obj_colormap[idx:idx+3]

def rasterize_hgable(idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal):
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

    tmp = (ys<ymin) | (ys>ymax) | (xs<xmin) | (xs>xmax)
    final_height[tmp] = founda_height
    final_normal[tmp,:] = 1.0

    face_colors = np.arange(4) + idx*4
    face_label = face_colors[minid]
    face_label[tmp] = -1
    return final_height, final_normal, face_label

def rasterize_vgable(idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal):
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

    tmp = (ys<ymin) | (ys>ymax) | (xs<xmin) | (xs>xmax)
    final_height[tmp] = founda_height
    final_normal[tmp,:] = 1.0

    face_colors = np.arange(4) + idx*4
    face_label = face_colors[minid+2]
    face_label[tmp] = -1
    return final_height, final_normal, face_label

def rasterize_hhip(idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal, roof_ratio):
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

    tmp = (ys<ymin) | (ys>ymax) | (xs<xmin) | (xs>xmax)
    final_height[tmp] = founda_height
    final_normal[tmp,:] = 1.0

    face_colors = np.arange(4) + idx*4
    face_label = face_colors[minid]
    face_label[tmp] = -1
    return final_height, final_normal, face_label

def rasterize_vhip(idx, ys, xs, founda_pos, founda_height, roof_height, roof_normal, roof_ratio):
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

    tmp = (ys<ymin) | (ys>ymax) | (xs<xmin) | (xs>xmax)
    final_height[tmp] = founda_height
    final_normal[tmp,:] = 1.0

    face_colors = np.arange(4) + idx*4
    face_label = face_colors[minid]
    face_label[tmp] = -1
    return final_height, final_normal, face_label


def rasterize_house(block_types, founda_positions, founda_height, roof_heights, roof_normals, roof_ratios, roof_graph):
    tmp = np.arange(64)
    ys = np.repeat(tmp, 64)
    xs = np.tile(tmp, 64)

    height_map, normal_map, label_map = [], [], []
    for i, btype in enumerate(block_types):
        if btype == H_GABLE:
            height_tmp, normal_tmp, label_tmp = rasterize_hgable(i, ys, xs, founda_positions[i], founda_height, roof_heights[i], roof_normals[i])
        elif btype == V_GABLE:
            height_tmp, normal_tmp, label_tmp = rasterize_vgable(i, ys, xs, founda_positions[i], founda_height, roof_heights[i], roof_normals[i])
        elif btype == H_HIP:
            height_tmp, normal_tmp, label_tmp = rasterize_hhip(i, ys, xs, founda_positions[i], founda_height, roof_heights[i], roof_normals[i], roof_ratios[i])
        elif btype == V_HIP:
            height_tmp, normal_tmp, label_tmp = rasterize_vhip(i, ys, xs, founda_positions[i], founda_height, roof_heights[i], roof_normals[i], roof_ratios[i])
        else:
            raise NotImplementedError
        height_map.append(height_tmp)
        normal_map.append(normal_tmp)
        label_map.append(label_tmp)
    
    
    height_map = np.stack(height_map)
    normal_map = np.stack(normal_map)
    label_map = np.stack(label_map)


    minid = np.argmax(height_map, axis=0)
    final_normal = normal_map[minid,np.arange(4096),:]
    final_normal[:2,:] = (final_normal[:2,:]+1.)/2.
    final_normal = final_normal.reshape(64,64,3)*255

    face_label = label_map[minid,np.arange(4096)]  
    face_label = combine_faces(face_label, roof_graph)  
    face_color = face_label2color(face_label)
    face_color = face_color.reshape(64,64,3)*255.
    return face_color

def combine_faces(face_label, roof_graph):
    face_label_new = face_label.copy()
    for i in range(1, roof_graph.shape[0]):
        for f in range(4):
            if roof_graph[i,f]>=0:
               tmp = face_label == (i*4+f)
               face_label_new[tmp] = roof_graph[i,f]
    return face_label_new

def face_label2color(face_label):
    maxid = np.amax(face_label)
    colorsets = np.random.random((maxid+2,3))
    colorsets[-1,:] = 1.
    return colorsets[face_label,:]
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
