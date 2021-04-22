import os
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from networks import get_network
from util.utils import TrainClock, cycle, ensure_dir
import cv2
import imageio
from dataset.house import House
# below is for testing
from dataset import HouseDataset, pad_collate_fn_for_dict_house
from torch.utils.data import DataLoader


class WGANAgant(object):
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()

        self.batch_size = config.batch_size
        self.n_iters = config.n_iters
        self.critic_iters = config.critic_iters
        self.save_frequency = config.save_frequency
        self.gp_lambda = config.gp_lambda
        self.n_dim = config.n_dim
        self.parallel = config.parallel 

        # build netD and netG
        self.build_net(config)

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))

    def build_net(self, config):
        self.netD = get_network('D', config).cuda()
        self.netG = get_network('G', config).cuda()

    def eval(self):
        self.netD.eval()
        self.netG.eval()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config.lr, betas=(config.beta1, 0.9))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=config.lr, betas=(config.beta1, 0.9))

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.step))
            print("Saving checkpoint epoch {}...".format(self.clock.step))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, save_path)

        self.netG.cuda()
        self.netD.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    def calc_gradient_penalty(self, netD, r_onehots, r_angles, r_colinear, f_onehots, f_angles, f_colinear, num_blocks, f_onehots_no, f_angles_no):
        alpha = torch.rand(r_onehots.size(0), 1, 1, 1).cuda()
        # ip means interpolates
        alpha_onehot = alpha.expand(r_onehots.size())
        ip_onehot = alpha_onehot * r_onehots.detach() + ((1 - alpha_onehot) * f_onehots.detach())
        ip_onehot.requires_grad_(True)
        ip_onehot_no = alpha_onehot * r_onehots.detach() + ((1 - alpha_onehot) * f_onehots_no.detach())
        ip_onehot_no.requires_grad_(True)

        alpha_angle = alpha.expand(r_angles.size())
        ip_angle = alpha_angle * r_angles.detach() + ((1-alpha_angle)*f_angles.detach())
        ip_angle.requires_grad_(True)
        ip_angle_no = alpha_angle * r_angles.detach() + ((1-alpha_angle)*f_angles_no.detach())
        ip_angle_no.requires_grad_(True)

        alpha = torch.rand(r_colinear.size(0),1).cuda()
        alpha = alpha.expand(r_colinear.size())
        ip_colinear = alpha * r_colinear.detach() + ((1 - alpha) * f_colinear.detach())
        ip_colinear.requires_grad_(True)
        disc_ip = netD(ip_onehot, ip_angle, ip_colinear, num_blocks, ip_onehot_no, ip_angle_no)
        ######################################
        gradients = autograd.grad(outputs=disc_ip, inputs=[ip_onehot,ip_angle],
                                  grad_outputs=torch.ones(disc_ip.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        gradients = torch.cat(gradients,1)
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty1 = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        ######################################
        gradients = autograd.grad(outputs=disc_ip, inputs=[ip_onehot_no,ip_angle_no],
                                  grad_outputs=torch.ones(disc_ip.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        gradients = torch.cat(gradients,1)
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty3 = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        ######################################
        gradients = autograd.grad(outputs=disc_ip, inputs=ip_colinear,
                                  grad_outputs=torch.ones(disc_ip.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0] #attentation [0]
        #gradients = torch.cat(gradients,1)
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty2 = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty1+gradient_penalty2+gradient_penalty3

    def train(self, dataloader):
        """training process"""
        data = cycle(dataloader)

        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda()
        mone = mone.cuda()

        pbar = tqdm(range(self.n_iters))
        for iteration in pbar:
            ############################
            # (1) Update D network
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True

            for iter_d in range(self.critic_iters):
                real_data = next(data)
                # parse data
                r_roof_onehots = real_data["roof_onehot_maps"].cuda()
                r_roof_angles = real_data["roof_angle_maps"].cuda()
                r_coli_onehots = real_data["colinear_onehots"].cuda()
                num_blocks = real_data["num_blocks"]
                r_roof_onehots.requires_grad_(True)
                r_roof_angles.requires_grad_(True)
                r_coli_onehots.requires_grad_(True)
                self.netD.zero_grad()

                # train with real
                D_real = self.netD(r_roof_onehots, r_roof_angles, r_coli_onehots, num_blocks, r_roof_onehots, r_roof_angles)
                D_real = D_real.mean()
                D_real.backward(mone)

                # train with fake
                noise = torch.randn(r_roof_onehots.size(0), self.n_dim).cuda()
                f_roof_onehots, f_roof_angles, f_coli_onehots, f_roof_onehots_no, f_roof_angles_no = self.netG(noise, num_blocks)
                D_fake = self.netD(f_roof_onehots.detach(), f_roof_angles.detach(), f_coli_onehots.detach(), num_blocks, 
                    f_roof_onehots_no.detach(), f_roof_angles_no.detach())
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = self.calc_gradient_penalty(self.netD, r_roof_onehots, r_roof_angles, r_coli_onehots,
                                    f_roof_onehots, f_roof_angles, f_coli_onehots, num_blocks, f_roof_onehots_no, f_roof_angles_no)
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                self.optimizerD.step()

            # if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation
            self.netG.zero_grad()

            noise = torch.randn(r_roof_onehots.size(0), self.n_dim)
            noise = noise.cuda()
            noise.requires_grad_(True)

            f_roof_onehots, f_roof_angles, f_coli_onehots, f_roof_onehots_no, f_roof_angles_no = self.netG(noise, num_blocks)
            G = self.netD(f_roof_onehots, f_roof_angles, f_coli_onehots, num_blocks, f_roof_onehots_no, f_roof_angles_no)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            self.optimizerG.step()

            # Write logs and save samples
            pbar.set_postfix({"D_loss": D_cost.item(), "G_loss": G_cost.item()})
            self.train_tb.add_scalars("loss", {"D_loss": D_cost.item(), "G_loss": G_cost.item()}, global_step=iteration)
            self.train_tb.add_scalar("wasserstein distance", Wasserstein_D.item(), global_step=iteration)

            # save model
            self.clock.tick()
            if self.clock.step % self.save_frequency == 0:
                self.save_ckpt()

    def generate(self, config):
        # np.random.seed(0)
        # torch.manual_seed(0)
        
        self.eval()
        save_dir = os.path.join(config.exp_dir, "results/ckpt-{}-num-{}".format(config.ckpt, config.n_samples))
        ensure_dir(save_dir)

        distribution = np.array([121,231,55,19])
        distribution = distribution/np.sum(distribution)
        n_blocks = np.random.choice(np.arange(2,6), config.n_samples, p=distribution)
        blockg_edges, edgeg_nodes, edgeg_edges = self.prepare_input(7)

        for i in tqdm(range(config.n_samples)):
            #print(i)
            valid = False
            while valid==False:
                block_num = n_blocks[i] if config.exclude==0 else config.exclude#np.random.randint(2,high=7)
                noise = torch.randn(block_num, self.n_dim).cuda()
                blockg_edge = blockg_edges[block_num]
                edgeg_node = edgeg_nodes[block_num]
                edgeg_edge = edgeg_edges[block_num]
                with torch.no_grad():
                    f_roof_onehots, f_roof_angles, f_coli_onehots, _, _ = self.netG(noise, torch.LongTensor([block_num]))
                    f_roof_onehots = f_roof_onehots.cpu().numpy()
                    f_roof_angles = f_roof_angles.squeeze(1).cpu().numpy()
                    f_coli_onehots = f_coli_onehots.cpu().numpy()
                    valid = self.parse_block(f_roof_onehots, f_roof_angles, f_coli_onehots, edgeg_node.numpy(), save_dir+f"/{i}", i)


    def parse_block(self, roof_onehots_map1, roof_angles_map1, coli_onehots, edgeg_node, save_dir, fileid):
        
        block_num = roof_onehots_map1.shape[0]
        roof_onehots_map, roof_angles_map = np.zeros((block_num, 3, 64, 64)), np.zeros((block_num, 64, 64))
        for i in range(block_num):
            ttmp = cv2.resize(roof_onehots_map1[i].transpose((1,2,0)), (64,64))
            roof_onehots_map[i] = ttmp.transpose((2,0,1))
            roof_angles_map[i] = cv2.resize(roof_angles_map1[i], (64,64))

        block_types, founda_masks, roof_angles = [], [], []
        tmp_gifs = []
        for i in range(block_num):
            roof_labels = np.argmax(roof_onehots_map[i], axis=0)
            roof_colors = np.zeros((64,64,3))
            for j in range(3):
                roof_colors[roof_labels==j,:] = color_set[j,:]
            roof_colors = np.concatenate([roof_colors, get_number_img(i)],1)
            tmp_gifs.append(roof_colors.astype(np.uint8))

            tb_mask = roof_labels==0
            tb_area = np.sum(tb_mask)
            lr_mask = roof_labels==1
            lr_area = np.sum(lr_mask)
            tmp_mask = roof_labels!=2
            founda_area = np.sum(tmp_mask)
            if founda_area<10:
                founda_masks.append(np.zeros(tmp_mask.shape))
                block_types.append(-1)
                roof_angles.append([0., 0.])
                continue
            founda_masks.append(tmp_mask.astype(np.float))
            if lr_area/founda_area<0.05:
                tmp_type = 0
            elif tb_area/founda_area<0.05:
                tmp_type=1
            elif tb_area>lr_area:
                tmp_type=2
            else:
                tmp_type=3
            block_types.append(tmp_type)
            tb_angle, lr_angle = 0., 0.
            tb_angle_m = roof_angles_map[i]#*roof_onehots_map[i,0]
            lr_angle_m = roof_angles_map[i]#*roof_onehots_map[i,1]
            if tmp_type==0:
                tb_angle = np.mean(tb_angle_m[tb_mask])
            elif tmp_type==1:
                lr_angle = np.mean(lr_angle_m[lr_mask])
            else:
                tb_angle = np.mean(tb_angle_m[tb_mask])
                lr_angle = np.mean(lr_angle_m[lr_mask])
            roof_angles.append([tb_angle, lr_angle])

        #imageio.mimsave(save_dir+"_blocks.gif", tmp_gifs, duration=1)
        block_types = np.asarray(block_types)
        founda_masks = np.asarray(founda_masks)
        roof_angles = np.asarray(roof_angles)

        house = House(block_types.copy(), founda_masks.copy(), 2., roof_angles.copy())
        if house.num_blocks<=1:
            return False
        n_blocks_save = np.array([house.num_blocks])
        ######### graph snapping
        coli_onehots = coli_onehots>0.5
        coli_binary = np.full((block_num, block_num, 6), False, dtype=np.bool)
        for k in range(edgeg_node.shape[0]):
            i, j = edgeg_node[k]
            coli_binary[i,j] = coli_onehots[k]
            coli_binary[j,i] = coli_onehots[k]
        
        house = House(block_types.copy(), founda_masks.copy(), 2., roof_angles.copy())
        house.graph_snap(coli_binary)
        height_map, face_masks, segment_img, normal_img = house.rasterize_house(scale_flag=True)
        cv2.imwrite(save_dir+"_graph.png", segment_img)
        cv2.imwrite(save_dir+"_normal_graph.png", normal_img)
        #np.savez(save_dir + f'_graph.npz', height_map=height_map, face_masks=face_masks, num_blocks=n_blocks_save)
        house.save_to_mesh(save_dir+"_graph.obj")
        
        return True
        
    def prepare_input(self, max_n):
        blockg_edges, edgeg_nodes, edgeg_edges = [], [], []
        for num_blocks in range(max_n):
            if num_blocks<2:
                blockg_edges.append([])
                edgeg_nodes.append([])
                edgeg_edges.append([])
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

            edgeg_edge = []
            for i in range(edgeg_node.shape[0]):
                a, b = edgeg_node[i,:]
                flag = np.logical_or.reduce((edgeg_node[:,0]==a, edgeg_node[:,0]==b, edgeg_node[:,1]==a, edgeg_node[:,1]==b))
                tmp = np.arange(edgeg_node.shape[0])
                edgeg_edge.append(tmp[np.logical_and(flag, tmp!=i)])
            edgeg_edge = np.concatenate(edgeg_edge)
            edgeg_edges.append(torch.LongTensor(edgeg_edge))
        return blockg_edges, edgeg_nodes, edgeg_edges

color_set = np.array(
    [255,0,0,
     0,255,0,
     0,0,0
    ]).reshape(-1,3)
def get_number_img(i):
    tmp = np.ones((64,64,3))*255
    cv2.putText(tmp, f'{i}', (10,53), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    return tmp
