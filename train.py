import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional

import glob
import os
import custom_transforms
from tqdm import tqdm
from data_loader_resize import kitti_training_set, training_data_kitti, my_training_set
from model.model_resize import Generator_Net, Discriminator_Net, get_scheduler, update_learning_rate

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

class Intensity_Loss(nn.Module):
    def __init__(self,l_num):
        super(Intensity_Loss,self).__init__()
        self.l_num=l_num
    def forward(self, gen_frames,gt_frames):
        return torch.sum(torch.mean(torch.abs((gen_frames-gt_frames)**self.l_num), (1,2,3,4)))
    
class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)
    
class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs ):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)
    
class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        b, t, c, h, w = gen_frames.shape
        gen_frames = gen_frames.reshape(-1, c, h, w)
        gt_frames = gt_frames.reshape(-1, c, h, w)
        
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)
    
def tensor2array(tensor):
    tensor = tensor.detach().cpu()
    array = 0.5 + tensor.numpy()*0.5
#     array = array.transpose(1, 2, 0)
    return array

def saver(model_state_dict, model_path, step,max_to_save=5):
    total_models=glob.glob(model_path+'*')
    if len(total_models)>=max_to_save:
        total_models = sorted(total_models, key=lambda name: int(name[len(model_path)+1:-4]))
        os.remove(total_models[0])
    torch.save(model_state_dict,model_path+'_'+str(step)+'.tar')
    print('model {} save successfully!'.format(model_path+'_'+str(step)+'.tar'))
    
class Config:
        DATASET_PATH = '/media/kuo/32AA7ACBAA7A8ADD/KITTI'
        SINGLE_TEST_PATH = ''
        WRITER_PATH = '/media/kuo/32AA7ACBAA7A8ADD/KITTI/VRSA/log'
        G_SAVE_PATH = '/media/kuo/32AA7ACBAA7A8ADD/KITTI/VRSA/save_model/generator/generator'
        D_SAVE_PATH = '/media/kuo/32AA7ACBAA7A8ADD/KITTI/VRSA/save_model/discriminator/discriminator'
        Pretrained = True
        Pretrained_generator_path = '/media/kuo/32AA7ACBAA7A8ADD/KITTI/VRSA/save_model/generator/generator_9000.tar'
        Pretrained_discriminator_path = '/media/kuo/32AA7ACBAA7A8ADD/KITTI/VRSA/save_model/discriminator/discriminator_9000.tar'
        IS_RGB = False
        BATCH_SIZE = 3
        niter = 10
        niter_decay = 0
        EPOCHS = niter + niter_decay
        step=1

# class Config:
#         DATASET_PATH = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/my_dataset/gray'
#         SINGLE_TEST_PATH = ''
#         WRITER_PATH = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/log'
#         G_SAVE_PATH = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/save_model/generator/generator'
#         D_SAVE_PATH = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/save_model/discriminator/discriminator'
#         Pretrained = True
#         Pretrained_generator_path = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/save_model/tmp/60epoch/generator_72000.tar'
#         Pretrained_discriminator_path = '/media/kuo/124C0E504C0E2ED3/KITTI/VRSA/save_model/tmp/60epoch/discriminator_72000.tar'
#         IS_RGB = False
#         BATCH_SIZE = 3
#         niter = 15
#         niter_decay = 15
#         EPOCHS = niter + niter_decay # 60
#         step=1


# load dataset
normalize = custom_transforms.Normalize(mean=[0.5], std=[0.5])

train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(), 
        normalize])

# kitti_training_set: scenes = ['campus', 'city', 'residential', 'road']
kitti_training_s = kitti_training_set(DATASET_PATH = Config.DATASET_PATH, scenes=['city', 'residential', 'road'],
                                        is_rgb = Config.IS_RGB)
training_data_kitti = training_data_kitti(kitti_training_s, is_rgb = Config.IS_RGB, transform=train_transform)

# my dataset
# vr_training_set = my_training_set(DATASET_PATH = Config.DATASET_PATH, scenes=['level_1', 'level_2', 'level_3'], is_rgb = False)
# training_data_kitti = training_data_kitti(vr_training_set, is_rgb = Config.IS_RGB, transform=train_transform)

trainloader_kitti = torch.utils.data.DataLoader(training_data_kitti, batch_size=Config.BATCH_SIZE,shuffle=True, drop_last=True)

# generate model
G_net = Generator_Net(is_rgb = Config.IS_RGB).to(torch.device("cuda"))
D_net = Discriminator_Net(is_rgb = Config.IS_RGB).to(torch.device("cuda"))
if Config.Pretrained:
    G_net.load_state_dict(torch.load(Config.Pretrained_generator_path))
    D_net.load_state_dict(torch.load(Config.Pretrained_discriminator_path))
    print('load success')
else:
    G_net.init_weights()
    D_net.init_weights()
    print('initialization success')
    
# tensorboard
writer=SummaryWriter(Config.WRITER_PATH)

# optimizer
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=0.00005, betas=(0.9, 0.999), weight_decay=1e-8) # lr=0.00005, betas=(0.9, 0.999), weight_decay=1e-8
optimizer_D = torch.optim.Adam(D_net.parameters(), lr=0.00005, betas=(0.9, 0.999), weight_decay=1e-8) # lr=0.00005, betas=(0.9, 0.999), weight_decay=1e-8
net_g_scheduler = get_scheduler(optimizer_G, Config)
net_d_scheduler = get_scheduler(optimizer_D, Config)

# loss
img_loss = Intensity_Loss(l_num=2).cuda()
gen_loss = Adversarial_Loss().cuda()
dis_loss = Discriminate_Loss().cuda()

#training
step = Config.step
for e in range(Config.EPOCHS):
    print('epochs_num: ', e)
    for image in tqdm(trainloader_kitti):
        real_label = torch.ones(size=(image.size(0), 1), requires_grad=False).cuda()
        fake_label = torch.zeros(size=(image.size(0), 1), requires_grad=False).cuda()

        image = image.cuda()
        
        #train_G
        for _ in range(2):
            optimizer_G.zero_grad()
            generated_image = G_net(image)
            d_out_fake = D_net(generated_image)
            int_loss = img_loss(generated_image, image)
            g_loss = gen_loss(d_out_fake) + int_loss

            g_loss.backward()
            optimizer_G.step()
        

        #train_D
        optimizer_D.zero_grad()

        generated_image = G_net(image)
        d_out_real = D_net(image)
        d_out_fake = D_net(generated_image.detach())
        d_loss = dis_loss(d_out_real, d_out_fake)

        d_loss.backward()
        optimizer_D.step()
        
        if step%100==0:
            print('g_loss {}, d_loss {}, int_loss {} ,'.format(g_loss, d_loss, int_loss))
            writer.add_scalar('total_loss/g_loss', g_loss, global_step=step)
            writer.add_scalar('total_loss/d_loss', d_loss, global_step=step)
            writer.add_scalar('g_loss/int_loss', int_loss, global_step=step)
            writer.add_image('image/train_target', tensor2array(image[-1,0]), global_step=step)
            writer.add_image('image/train_output', tensor2array(generated_image[-1,0]), global_step=step)
            print('G learning rate = %.7f' % optimizer_G.param_groups[0]['lr'])
            print('D learning rate = %.7f' % optimizer_D.param_groups[0]['lr'])
            
        if step%1000==0 and step != 0:
            saver(G_net.state_dict(), Config.G_SAVE_PATH, step, max_to_save=5)
            saver(D_net.state_dict(), Config.D_SAVE_PATH, step, max_to_save=5)
            
        step+=1
    update_learning_rate(net_g_scheduler)
    update_learning_rate(net_d_scheduler)