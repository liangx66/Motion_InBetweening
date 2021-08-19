import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
sys.path.insert(0, os.path.dirname(__file__))
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from utils.LaFan1 import LaFan1
from utils.skeleton import Skeleton
from utils.interpolate import interpolate_local
from utils.functions import write_to_bvhfile
import numpy as np
import yaml
import random
import imageio
import matplotlib.pyplot as plt
from model import Encoder
from PIL import Image


def interpolation(X, Q, n_past=10, n_future=10, n_trans = 30):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape     numpy(Batchsize, Timesteps, Joints, 3)
    :param Q: Local quaternions array of shape     numpy(B, T, J, 4)
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :param n_trans:
    :return:  B, curr_window, xxx
    """
    batchsize = X.shape[0]  #  B

    # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
    curr_window = n_trans + n_past + n_future
    curr_x = X[:, :curr_window, ...]    # B, curr_window, J, 3
    curr_q = Q[:, :curr_window, ...]    # B, curr_window, J, 4
    gt_pose = np.concatenate([curr_x.reshape((batchsize, curr_window, -1)), curr_q.reshape((batchsize, curr_window, -1))], axis=2)  # [B, curr_window, J*3+J*4]

    # Interpolation pos/quats
    x, q = curr_x, curr_q    # x: B,curr_window,J,3       q: B, curr_window, J, 4
    inter_pos, inter_local_quats = interpolate_local(x, q, n_past, n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4
    inter_pos = inter_pos.numpy()
    inter_local_quats = inter_local_quats.numpy()
    trans_inter_pos = inter_pos[:, 1:-1, :, :]    #  B, n_trans, J, 3
    inter_local_quats = inter_local_quats[:, 1:-1, :, :]  #  B, n_trans, J, 4
    total_interp_positions = np.concatenate([curr_x[:, 0: n_past, ...], trans_inter_pos, curr_x[:, -n_future: , ...]], axis = 1)    # B, curr_window, J, 3
    total_interp_rotations = np.concatenate([curr_q[:, 0: n_past, ...], inter_local_quats, curr_q[:, -n_future: , ...]], axis = 1)  # B, curr_window, J, 4
    interp_pose = np.concatenate([total_interp_positions.reshape((batchsize, curr_window, -1)), total_interp_rotations.reshape((batchsize, curr_window, -1))], axis=2)  # [B, curr_window, xxx]
    return gt_pose, interp_pose

def plot_pose(pose, cur_frame, prefix):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    ax.cla()
    num_joint = pose.shape[0] // 3
    for i, p in enumerate(parents):
        if i > 0:
            ax.plot([pose[i, 0], pose[p, 0]],\
                    [pose[i, 2], pose[p, 2]],\
                    [pose[i, 1], pose[p, 1]], c='r')
            ax.plot([pose[i+num_joint, 0], pose[p+num_joint, 0]],\
                    [pose[i+num_joint, 2], pose[p+num_joint, 2]],\
                    [pose[i+num_joint, 1], pose[p+num_joint, 1]], c='b')
            ax.plot([pose[i+num_joint*2, 0], pose[p+num_joint*2, 0]],\
                    [pose[i+num_joint*2, 2], pose[p+num_joint*2, 2]],\
                    [pose[i+num_joint*2, 1], pose[p+num_joint*2, 1]], c='g')
    # ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1],c='b')
    # ax.scatter(pose[num_joint:num_joint*2, 0], pose[num_joint:num_joint*2, 2], pose[num_joint:num_joint*2, 1],c='b')
    # ax.scatter(pose[num_joint*2:num_joint*3, 0], pose[num_joint*2:num_joint*3, 2], pose[num_joint*2:num_joint*3, 1],c='g')
    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(prefix + '_' + str(cur_frame)+'.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    opt = yaml.load(open('config/test_config_bfaxia.yaml', 'r').read())
    model_dir =opt['test']['model_dir']

    random.seed(opt['test']['seed'])
    torch.manual_seed(opt['test']['seed'])
    if opt['test']['cuda']:
        torch.cuda.manual_seed(opt['test']['seed'])

    # ===================================放到GPU==================================
    device = torch.device('cuda' if opt['test']['cuda'] else 'cpu')
    print(device)

    ## initilize the skeleton ##
    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    skeleton_mocap.to(device)
    if opt['data']['data_set'] == "lafan":
        skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])

    ## load test data ##
    lafan_data_test = LaFan1(opt['data']['data_dir'], \
                             opt['data']['data_set'], \
                             seq_len = opt['model']['seq_length'], \
                              offset = opt['data']['offset'],\
                              train = False,
                              debug=opt['test']['debug'])
    x_mean = lafan_data_test.x_mean.to(device)
    x_std = lafan_data_test.x_std.to(device).view(1, 1, opt['model']['num_joints'], 3)
    print("test_positions.shape", lafan_data_test.data['X'].shape)
    print("test_rotations.shape", lafan_data_test.data['local_q'].shape)

    lafan_loader_test = DataLoader(lafan_data_test, \
                                    batch_size=opt['test']['batch_size'], \
                                    shuffle=False)

    ## initialize model and load parameters ##
    model = Encoder(device=device,
                    seq_len=opt['model']['seq_length'],
                    input_dim=opt['model']['input_dim'],
                    n_layers=opt['model']['n_layers'],
                    n_head=opt['model']['n_head'],
                    d_k=opt['model']['d_k'],
                    d_v=opt['model']['d_v'],
                    d_model=opt['model']['d_model'],
                    d_inner=opt['model']['d_inner'],
                    dropout=opt['test']['dropout'],
                    n_past=opt['model']['n_past'],
                    n_future=opt['model']['n_future'],
                    n_trans=opt['model']['n_trans'])

    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model loaded.')

    model = model.to(device)
    model.eval()

    loss_total_list = []

    for batch_i, batch_data in enumerate(lafan_loader_test):
        loss_pose = 0
        loss_quat = 0
        loss_position = 0

        pred_img_list = []
        gt_img_list = []

        with torch.no_grad():
            positions = batch_data['X']         # [B, F, J, 3]
            rotations = batch_data['local_q']   # [B, F, J, 4]

            # gt_pose numpy [B, F, dim] interp_pose numpy[B, F, dim]
            gt_pose, interp_pose = interpolation(positions,
                                                 rotations,
                                                 n_past=opt['model']['n_past'],
                                                 n_future=opt['model']['n_future'],
                                                 n_trans=opt['model']['n_trans'])
            gt_pose = gt_pose.astype(np.float32)
            interp_pose = interp_pose.astype(np.float32)
            input = torch.from_numpy(interp_pose).to(device)
            target_output = torch.from_numpy(gt_pose).to(device)

            output = model(input)
            # rotations
            local_q_pred = output[:, :, opt['model']['num_joints'] * 3:]  # 1, F, J*4
            local_q_gt = target_output[:, :, opt['model']['num_joints'] * 3:]  # 1, F, J*4
            # positions
            glbl_p_pred = output[:, :, 0:opt['model']['num_joints'] * 3]  # 1, F, J*3
            glbl_p_gt = target_output[:, :, 0:opt['model']['num_joints'] * 3]  # 1, F, J*3
            # root positions
            root_pred = glbl_p_pred[:, :, 0:3]  # 1, F, 3
            root_gt = glbl_p_gt[:, :, 0:3]  # 1, F, 3

            local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)    # 1, F, J, 4
            local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim=-1, keepdim=True)         # 1, F, J, 4
            pos_pred = skeleton_mocap.forward_kinematics(local_q_pred_, root_pred)
            glbl_p_gt_ = glbl_p_gt.view(glbl_p_gt.size(0), glbl_p_gt.size(1), -1, 3)                # 1, F, J, 3

            loss_pose += torch.mean(torch.abs(pos_pred - glbl_p_gt_) / x_std)
            loss_quat += torch.mean(torch.abs(local_q_pred - local_q_gt))
            loss_position += torch.mean(torch.abs(glbl_p_pred - glbl_p_gt))
            loss_total = opt['test']['loss_quat_weight'] * loss_quat + \
                         opt['test']['loss_position_weight'] * loss_position + \
                         opt['test']['loss_pose_weight'] * loss_pose
            loss_total_list.append(loss_total.item())

            output_dir = opt['test']['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if opt['data']['data_set'] == "mocap":
                style = batch_data['style']
                pd_output_name = ('test_%d_%s_pd.bvh' % (batch_i, style))
                gt_output_name = ('test_%d_%s_gt.bvh' % (batch_i, style))
            else:
                pd_output_name = ('test_%03d_pd.bvh'% batch_i)
                gt_output_name = ('test_%03d_gt.bvh' % batch_i)
            pd_output_path = pjoin(output_dir, pd_output_name)
            gt_output_path = pjoin(output_dir, gt_output_name)

            # interp_output_name = ('test_%03d_interp.bvh'% batch_i)
            # interp_output_path = pjoin(output_dir, interp_output_name)

            if opt['test']['save_bvh']:
                pd_bvh_data = torch.cat([root_pred[0], local_q_pred_[0].view(local_q_pred_.size(1), -1)], -1).detach().cpu().numpy()  # F, J*(3+4)
                gt_bvh_data = torch.cat([root_gt[0], local_q_gt[0].view(local_q_gt.size(1), -1)],-1).detach().cpu().numpy()  # F, J*(3+4)
                # print('bvh_data:', bvh_data.shape)
                # print('bvh_data:', bvh_data[0,3:7])
                write_to_bvhfile(pd_bvh_data, pd_output_path, opt['data']['data_set'])
                write_to_bvhfile(gt_bvh_data, gt_output_path, opt['data']['data_set'])

            # save_img & save_gif 还有问题
            if opt['test']['save_img'] and opt['test']['save_gif'] and batch_i < 50:
                gif_dir = opt['test']['gif_dir']
                if not os.path.exists(gif_dir):
                    os.mkdir(gif_dir)
                img_dir = opt['test']['img_dir']
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)
                num_joints = opt['model']['num_joints']
                position_0 = glbl_p_gt_[0, 0, ...].detach().to('cpu').numpy()
                position_1 = glbl_p_gt_[0, -1, ...].detach().to('cpu').numpy()
                for t in range(opt['model']['seq_length']):
                    # print(type(position_0))
                    # print(position_0.device)
                    plot_pose(np.concatenate([position_0, pos_pred[0, t].detach().to('cpu').numpy(), position_1], axis=0), \
                              t,
                              img_dir + '/pred_batch_'+str(batch_i))
                    plot_pose(np.concatenate([position_0,   #.detach().cpu().numpy(), \
                                              glbl_p_gt_[0, t].detach().to('cpu').numpy(),
                                              position_1], 0), \
                              t,
                              img_dir + '/gt_batch_'+str(batch_i))
                    pred_img = Image.open(img_dir + '/pred_batch_'+str(batch_i)+ '_' + str(t) + '.png', 'r')
                    gt_img = Image.open(img_dir + '/gt_batch_'+str(batch_i)+ '_' + str(t) + '.png', 'r')
                    pred_img_list.append(pred_img)
                    gt_img_list.append(gt_img)
                imageio.mimsave((gif_dir + '/pred_img_%03d.gif' % batch_i), pred_img_list, duration=0.1)
                imageio.mimsave((gif_dir + '/gt_img_%03d.gif' % batch_i), gt_img_list, duration=0.1)

                
    print("batch:%5d, avg test loss:%.6f"% (batch_i, np.mean(loss_total_list)))