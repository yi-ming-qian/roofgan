import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

max_flag = True

# Conv modules
class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))
class TranConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(TranConvReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))

class ConvMPN(nn.Module):
    def __init__(self, in_channels, edgeg_flag=False):
        super(ConvMPN, self).__init__()
        self.edgeg_flag = edgeg_flag
        self.main = nn.Sequential(
            ConvReLU(in_channels*2, in_channels*2, 3, 1, 1),
            ConvReLU(in_channels*2, in_channels, 3, 1, 1),
            ConvReLU(in_channels, in_channels, 3, 1, 1)
            )

    def aggregate(self, x, edge, n):
        if self.edgeg_flag and edge.size(0)==0:
            out = torch.zeros_like(x)
            return out
        if self.edgeg_flag:
            x_n = x[edge]
            x_n = x_n.view(int(n*(n-1)/2), (n-2)*2, x.size(1), x.size(2), x.size(3))
        else:
            x_n = x[edge]
            x_n = x_n.view(n, n-1, x.size(1), x.size(2), x.size(3))
        if max_flag:
            out = torch.max(x_n,1)[0]
        else:
            out = torch.mean(x_n,1)
        return out

    def forward(self, feats, num_blocks, blockg_edge=None, edgeg_edge=None):
        # feature aggregation
        feats_agg = []
        batch_size = len(num_blocks)
        offset1 = 0
        for i in range(batch_size):
            n = num_blocks[i].item()
            if self.edgeg_flag:
                t1 = int(n*(n-1)/2)
                feat = feats[offset1:offset1+t1]
                offset1 += t1

                edge = edgeg_edge[n]
            else:
                feat = feats[offset1:offset1+n]
                offset1 += n

                edge = blockg_edge[n]
            feats_agg.append(self.aggregate(feat, edge, n))
        feats_agg = torch.cat(feats_agg)
        return self.main(torch.cat([feats, feats_agg], 1))



class Generator(nn.Module):

    def __init__(self, n_dim):
        super(Generator, self).__init__()
        self.linear_map = nn.Linear(n_dim, 1024)
        #self.encoder = nn.Sequential(ConvReLU(16,32,3,1,1))

        self.conv_mpn_1 = ConvMPN(16)
        self.upsample_1 = TranConvReLU(16,16,4,2,1)
        self.conv_mpn_2 = ConvMPN(16)
        self.upsample_2 = TranConvReLU(16,16,4,2,1)

        self.onehot = nn.Sequential(
            ConvReLU(16, 256, 3, 1, 1),
            ConvReLU(256, 128, 3, 1, 1),
            nn.Conv2d(128, 3, 3, 1, 1)
            )
        self.angle =nn.Sequential(
            ConvReLU(16, 256, 3, 1, 1),
            ConvReLU(256, 128, 3, 1, 1),
            nn.Conv2d(128, 1, 3, 1, 1)
            )
        self.softmax = nn.Softmax(dim=1)

        # relation
        self.coli_decoder = nn.Sequential(
            ConvReLU(32, 48, 4, 2, 1),
            ConvReLU(48, 64, 4, 2, 1),
            ConvReLU(64, 128, 4, 2, 1),
            ConvReLU(128, 256, 4, 2, 1),
            nn.Conv2d(256, 512, 4, 2, 1)
            )
        self.coli_fc = nn.Linear(512, 6)

        # differential snapping
        rows = torch.arange(32, dtype=torch.float)/31*2-1
        self.rows = rows.unsqueeze(-1).repeat(1,32).cuda()
        columns = torch.arange(32, dtype=torch.float)/31*2-1
        self.columns = columns.unsqueeze(0).repeat(32,1).cuda()
        self.top_filter = torch.FloatTensor([[0, -1, 0],[0, 1, 0],[0, 0, 0]]).view(1,1,3,3).cuda()
        self.bottom_filter = torch.FloatTensor([[0, 0, 0],[0, 1, 0],[0, -1, 0]]).view(1,1,3,3).cuda()
        self.left_filter = torch.FloatTensor([[0, 0, 0],[-1, 1, 0],[0, 0, 0]]).view(1,1,3,3).cuda()
        self.right_filter = torch.FloatTensor([[0, 0, 0],[0, 1, -1],[0, 0, 0]]).view(1,1,3,3).cuda()

        self.blockg_edges, self.blockg_to_edgegs, self.edgeg_nodes, _ = prepare_graph()


    def colinear(self, block_feats, num_blocks):
        batch_size = len(num_blocks)
        offset1 = 0
        edge_feats = []
        for b in range(batch_size):
            n = num_blocks[b].item()
            feat = block_feats[offset1:offset1+n]
            offset1 += n

            edge = self.edgeg_nodes[n]

            left = feat[edge[:,0]]
            right = feat[edge[:,1]]
            edge_feats.append(torch.cat([left, right], 1))
        edge_feats = torch.cat(edge_feats)
        edge_feats = self.coli_decoder(edge_feats)
        edge_feats = edge_feats.squeeze(-1).squeeze(-1)
        edge_feats = self.coli_fc(edge_feats)
        edge_feats = torch.sigmoid(edge_feats)
        return edge_feats

    def get_boundary(self, mask):
        t_weight = F.relu(F.conv2d(mask, self.top_filter, padding=1))
        t_sum = torch.sum(t_weight*self.rows, dim=(1,2,3))
        t_num = torch.sum(t_weight, dim=(1,2,3))
        top_bounds = t_sum/(t_num+1e-6)# zero division?

        t_weight = F.relu(F.conv2d(mask, self.bottom_filter, padding=1))
        t_sum = torch.sum(t_weight*self.rows, dim=(1,2,3))
        t_num = torch.sum(t_weight, dim=(1,2,3))
        bottom_bounds = t_sum/(t_num+1e-6)

        t_weight = F.relu(F.conv2d(mask, self.left_filter, padding=1))
        t_sum = torch.sum(t_weight*self.columns, dim=(1,2,3))
        t_num = torch.sum(t_weight, dim=(1,2,3))
        left_bounds = t_sum/(t_num+1e-6)

        t_weight = F.relu(F.conv2d(mask, self.right_filter, padding=1))
        t_sum = torch.sum(t_weight*self.columns, dim=(1,2,3))
        t_num = torch.sum(t_weight, dim=(1,2,3))
        right_bounds = t_sum/(t_num+1e-6)
        return torch.stack([top_bounds, bottom_bounds, left_bounds, right_bounds], -1)

    def get_affine(self, new_bound, old_bound):
        d1 = old_bound[:,(3,1)] - old_bound[:,(2,0)]
        d2 = new_bound[:,(3,1)] - new_bound[:,(2,0)]
        scale_xy = d1/(d2+1e-6)
        trans_xy = old_bound[:,(2,0)] - scale_xy*new_bound[:,(2,0)]
        affine_mat = torch.zeros(scale_xy.size(0),2,3).cuda()
        affine_mat[:,:,2] = trans_xy
        affine_mat[:,(0,1),(0,1)] = scale_xy
        return affine_mat

    def ds(self, onehots, angles, colinears, num_blocks):
        mask = 1-onehots[:,2:3,:,:]
        bounds = self.get_boundary(mask)
        batch_size = len(num_blocks)
        offset1, offset2 = 0, 0
        affine_mats = []
        for b in range(batch_size): # for each house
            n = num_blocks[b].item() # n blocks
            t_bound = bounds[offset1:offset1+n]
            offset1 += n
            t = int(n*(n-1)/2)
            colinear = F.relu(colinears[offset2:offset2+t]*2-1)
            offset2 += t

            blockg_edge = self.blockg_edges[n] # nx(n-1)
            other_bounds = t_bound[blockg_edge] # all the other bounds
            other_bounds = other_bounds.view(n, n-1, other_bounds.size(1))
            #disp = other_bounds - t_bound.unsqueeze(1)
            colinear_weights = colinear[self.blockg_to_edgegs[n]]
            colinear_weights = colinear_weights.view(n,n-1,colinear_weights.size(1))

            t_sum = other_bounds*colinear_weights
            t_sum = torch.sum(t_sum, dim=1)
            t_num = torch.sum(colinear_weights, dim=1)
            new_bound = (t_bound + t_sum)/(t_num+1)
            # print((t_bound+1)/2*31)
            # print((new_bound+1)/2*31)
            
            affine_mat = self.get_affine(new_bound, t_bound)
            affine_mats.append(affine_mat)
        affine_mats = torch.cat(affine_mats)
        grid = F.affine_grid(affine_mats, onehots.size())

        mask = F.grid_sample(mask, grid, padding_mode="zeros", mode="bilinear")
        onehots23 = 1-mask
        onehots02 = F.grid_sample(onehots[:,0:2,:,:], grid, padding_mode="zeros", mode="bilinear")
        onehots_ds = torch.cat([onehots02, onehots23],1)
        angles_ds = F.grid_sample(angles, grid, padding_mode="zeros", mode="bilinear")

        return onehots_ds, angles_ds


    def forward(self, noise, num_blocks):
        x = self.linear_map(noise)
        x = x.view(-1, 16, 8, 8)

        x = self.conv_mpn_1(x, num_blocks, blockg_edge=self.blockg_edges)
        x = self.upsample_1(x)
        x = self.conv_mpn_2(x, num_blocks, blockg_edge=self.blockg_edges)
        x = self.upsample_2(x)

        onehot = self.onehot(x)
        onehot = self.softmax(onehot)
        angle = self.angle(x)
        angle = torch.sigmoid(angle)
        colinear = self.colinear(x, num_blocks)
        onehot_ds, angle_ds = self.ds(onehot, angle, colinear[:,:4], num_blocks)

        return onehot_ds, angle_ds, colinear, onehot, angle


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.onehot = nn.Sequential(
            ConvReLU(6, 16, 3, 1, 1),
            ConvReLU(16, 16, 3, 1, 1),
            ConvReLU(16, 16, 3, 1, 1)
            )
        self.angle = nn.Sequential(
            ConvReLU(2, 16, 3, 1, 1),
            ConvReLU(16, 16, 3, 1, 1),
            ConvReLU(16, 16, 3, 1, 1)
            )
        d1_chn = 32
        self.d1_conv_mpn_1 = ConvMPN(d1_chn)
        self.d1_downsample_1 = ConvReLU(d1_chn,d1_chn,3,2,1)
        self.d1_conv_mpn_2 = ConvMPN(d1_chn)
        self.d1_downsample_2 = ConvReLU(d1_chn,d1_chn,3,2,1)
        self.d1_decoder = nn.Sequential(
            ConvReLU(d1_chn, 256, 3, 2, 1),
            ConvReLU(256, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 2, 1)
            )
        self.d1_fc = nn.Linear(128, 1)

        self.d2_coli = nn.Linear(6, 4*32*32)
        self.d2_decoder = nn.Sequential(
            ConvReLU(26, 64, 4, 2, 1),
            ConvReLU(64, 128, 4, 2, 1),
            ConvReLU(128, 256, 4, 2, 1),
            ConvReLU(256, 256, 4, 2, 1),
            nn.Conv2d(256, 128, 4, 2, 1)
            )
        self.d2_fc = nn.Linear(128, 1)

        self.blockg_edges, _, self.edgeg_nodes, _ = prepare_graph()


    def D1(self, x, num_blocks):
        x = self.d1_conv_mpn_1(x, num_blocks, blockg_edge=self.blockg_edges)
        x = self.d1_downsample_1(x)
        x = self.d1_conv_mpn_2(x, num_blocks, blockg_edge=self.blockg_edges)
        x = self.d1_downsample_2(x)
        x = self.d1_decoder(x)
        x = x.squeeze(-1).squeeze(-1)
        batch_size = len(num_blocks)
        output = []
        offset = 0
        for i in range(batch_size):
            n = num_blocks[i]
            z = x[offset:offset+n]
            offset += n
            if max_flag:
                output.append(torch.max(z,0,keepdim=True)[0])
            else:
                output.append(torch.mean(z,0,keepdim=True))
        output = torch.cat(output)
        return self.d1_fc(output)

    def D2(self, x, y, num_blocks):
        # group together
        batch_size = len(num_blocks)
        offset1, offset2 = 0, 0
        coli_feats = []
        for i in range(batch_size):
            n = num_blocks[i].item()
            feat = x[offset1:offset1+n]
            offset1 += n

            t = int(n*(n-1)/2)
            edge = self.edgeg_nodes[n]
            rela = y[offset2:offset2+t]
            offset2 += t

            left = feat[edge[:,0]]
            right = feat[edge[:,1]]
            coli_feats.append(torch.cat([left, right, rela], 1))

        coli_feats = torch.cat(coli_feats)
        coli_feats = self.d2_decoder(coli_feats)
        coli_feats = coli_feats.squeeze(-1).squeeze(-1)
        output = self.d2_fc(coli_feats)
        return output
    
    def forward(self, roof_onehots, roof_angles, colinear_ohots, num_blocks, roof_onehots_no, roof_angles_no):
        onehot_feats = self.onehot(torch.cat([roof_onehots, roof_onehots_no],1))
        angle_feats = self.angle(torch.cat([roof_angles, roof_angles_no],1))
        x = torch.cat([onehot_feats,angle_feats], 1)
        y = colinear_ohots.unsqueeze(2).unsqueeze(3).repeat(1,1,32,32)
        z = self.d2_coli(colinear_ohots).view(colinear_ohots.size(0),4,32,32)
        y = torch.cat([z,y],1)
        output1 = self.D1(x, num_blocks)
        x2 = torch.cat([roof_onehots, roof_angles, roof_onehots_no, roof_angles_no],1)
        output2 = self.D2(x2, y, num_blocks)
        batch_size = len(num_blocks)
        return output1+torch.mean(output2)*batch_size#, torch.mean(output2)

        

def prepare_graph(max_n=7):
    blockg_edges, edgeg_nodes, edgeg_edges, blockg_to_edgegs = [], [], [], []
    for num_blocks in range(max_n):
        if num_blocks<2:
            blockg_edges.append([])
            edgeg_nodes.append([])
            edgeg_edges.append([])
            blockg_to_edgegs.append([])
            continue
        blockg_edge, edgeg_node = [], []
        for i in range(num_blocks):
            tmp = np.arange(num_blocks)
            blockg_edge.append(tmp[tmp!=i])
        blockg_edge = np.concatenate(blockg_edge)
        blockg_edges.append(torch.LongTensor(blockg_edge))

        for i in range(1, num_blocks):
            for j in range(i):
                edgeg_node.append([i,j])
        edgeg_node = np.asarray(edgeg_node)
        edgeg_nodes.append(torch.LongTensor(edgeg_node))

        blockg_to_edgeg = []
        for i in range(num_blocks):
            flag = np.logical_or(edgeg_node[:,0]==i, edgeg_node[:,1]==i)
            tmp = np.arange(edgeg_node.shape[0])
            blockg_to_edgeg.append(tmp[flag])
        blockg_to_edgeg = np.concatenate(blockg_to_edgeg)
        blockg_to_edgegs.append(torch.LongTensor(blockg_to_edgeg))

        edgeg_edge = []
        for i in range(edgeg_node.shape[0]):
            a, b = edgeg_node[i,:]
            flag = np.logical_or.reduce((edgeg_node[:,0]==a, edgeg_node[:,0]==b, edgeg_node[:,1]==a, edgeg_node[:,1]==b))
            tmp = np.arange(edgeg_node.shape[0])
            edgeg_edge.append(tmp[np.logical_and(flag, tmp!=i)])
        edgeg_edge = np.concatenate(edgeg_edge)
        edgeg_edges.append(torch.LongTensor(edgeg_edge))
    return blockg_edges, blockg_to_edgegs, edgeg_nodes, edgeg_edges
