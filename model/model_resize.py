import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, normal_
from torch.optim import lr_scheduler

from model.ConvLSTM import ConvLSTM

def get_scheduler(optimizer, Config):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch+Config.step-Config.niter)/float(Config.niter_decay+1)
        return lr_l
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

def update_learning_rate(scheduler):
    scheduler.step()

def vgg16(channel_num):
    return nn.Sequential(
        nn.Conv2d(channel_num, 64, 3, stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.Conv2d(64, 64, 3, stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),     
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ELU(inplace=True)
    )

def spatial_decoder(channel_num):
    return nn.Sequential(
        nn.Conv2d(512, 256, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        nn.ConvTranspose2d(128, channel_num, 3, stride=2, padding=1, output_padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(channel_num, channel_num, 3, stride=1, padding=1),
        nn.Tanh()
    )

class Generator_Net(nn.Module):
    def __init__(self, is_rgb = False):
        super(Generator_Net, self).__init__()
        if is_rgb == True:
            self.vgg16 = vgg16(channel_num=3)
        else:
            self.vgg16 = vgg16(channel_num=1)
            
        self.conlstm1 = ConvLSTM(input_size=(20, 40), input_dim=512, hidden_dim=[256, 512], 
                                 kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        
#         self.conlstm1 = ConvLSTM(input_size=(20, 40), input_dim=512, hidden_dim=[256, 512], 
#                                  kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True)
        
        if is_rgb == True:
            self.spatial_decoder = spatial_decoder(channel_num=3)
        else:
            self.spatial_decoder = spatial_decoder(channel_num=1)
            
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
#                 normal_(m.weight, 0.0)
                if m.bias is not None:
                    zeros_(m.bias)
#         self.conlstm1._ini_weight()
    
    def forward(self, x):
        b_size, t_s, _, _, _ = x.size()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.vgg16(x)
        x = x.view(b_size, t_s, x.size(1), x.size(2), x.size(3))

        x, _ = self.conlstm1(x)
#         x, _ = self.conlstm2(x[0])
        
        x=x[0]
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.spatial_decoder(x)
        x = x.view(b_size, t_s, x.size(1), x.size(2), x.size(3))
        return x
    
class Discriminator_Net(nn.Module):
    def __init__(self, is_rgb = False):
        super(Discriminator_Net, self).__init__()
        if is_rgb == True:
            channel_num=3
        else:
            channel_num=1
        self.conv3d = nn.Sequential(
            nn.Conv3d(channel_num, 32, kernel_size=(5,5,5), stride=(1,2,2), padding=(0,2,2)),
            nn.ELU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2)),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True),
            nn.Conv3d(256, 1, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(49+1, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
#                 normal_(m.weight, 0.0)
                if m.bias is not None:
                    zeros_(m.bias)
                    
    def forward(self, x):
        x = torch.transpose(x, 1,2)
        x = self.conv3d(x)
        x = x.view(x.size(0), 49+1)
        x = self.linear(x)
        return x
    
class Predict_Net(nn.Module):
    def __init__(self, is_rgb = False):
        super(Predict_Net, self).__init__()
        if is_rgb == True:
            channel_num=3
        else:
            channel_num=1
        self.conv3d = nn.Sequential(
            nn.Conv3d(channel_num, 32, kernel_size=(5,5,5), stride=(1,2,2), padding=(0,2,2)),
            nn.ELU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2)),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True),
            nn.Conv3d(256, 1, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.ELU(inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(49+1, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
#                 normal_(m.weight, 0.0)
                if m.bias is not None:
                    zeros_(m.bias)
                    
    def forward(self, x):
        x = torch.transpose(x, 1,2)
        x = self.conv3d(x)
        x = x.view(x.size(0), 49+1)
        x = self.linear(x)
        return x

'''
if __name__ == "__main__":
    import numpy as np
    img = np.ones((3,11,1,224,224))

    G_net = Generator_Net().to(torch.device("cuda"))
    D_net = Discriminator_Net().to(torch.device("cuda"))
    G_net.init_weights()
    D_net.init_weights()

    optimizer_G = torch.optim.Adam(G_net.parameters(), lr=1e-4, eps=1e-06, weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=1e-4, eps=1e-06, weight_decay=1e-5)

    img_loss = nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    image = torch.from_numpy(img).float().cuda()
    real_label = torch.ones(size=(3, 1), requires_grad=False).cuda()
    fake_label = torch.zeros(size=(3, 1), requires_grad=False).cuda()
    
    #train_G
    optimizer_G.zero_grad()

    generated_image = G_net(image)
    d_out_fake = D_net(generated_image)
    g_loss = bce_loss(d_out_fake, real_label)+img_loss(generated_image, image)

    print('g_loss: ', g_loss)

    g_loss.backward()
    optimizer_G.step()

    #train_D
    optimizer_D.zero_grad()

    d_out_real = D_net(image)
    real_loss = bce_loss(d_out_real, real_label)
    d_out_fake = D_net(generated_image.detach())
    fake_loss = bce_loss(d_out_fake, fake_label)

    d_loss = real_loss + fake_loss

    print('d_loss: ', d_loss)

    d_loss.backward()
    optimizer_D.step()
 '''