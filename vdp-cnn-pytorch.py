import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import csv
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
torch.manual_seed(1);

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

class VDPFirstConv(nn.Module):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="valid",input_channels=1):
        super(VDPFirstConv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.w_mu = nn.Parameter(nn.init.normal_(torch.empty(self.kernel_num, input_channels,
                                             self.kernel_size, self.kernel_size), mean= 0.0 , std = 0.05),requires_grad = True)
        self.w_sigma = nn.Parameter(torch.full((self.kernel_num,), -2.2), requires_grad = True)


    def forward(self, mu_in):
        
        
        batch_size, num_channel, image_size,_ = mu_in.size()
        # Perform convolution
        mu_out = F.conv2d(mu_in, self.w_mu, stride=(self.kernel_stride,self.kernel_stride), padding=self.padding)

        # Extract patches and calculate X_XTranspose
        x_train_patches = mu_in.unfold(2,self.kernel_size, 1).unfold(3, self.kernel_size, 1) # shape = [batch_size, channels, H/2 , W/2, kernel size , kernel size ]
        #print(f'X_train_patches = {x_train_patches.shape}')
        x_train_patches = x_train_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_train_patches = x_train_patches.view(*x_train_patches.size()[:3], -1)
        #print(f'X_train_patches = {x_train_patches.shape}')
        x_train_matrix = x_train_patches.view(x_train_patches.size()[0],-1,x_train_patches.size()[3])
        #print(f'X_train_matrix = {x_train_matrix.shape}')
        X_XTranspose = torch.matmul(x_train_matrix, x_train_matrix.transpose(1, 2))
        #print(f'X_XTranspose = {X_XTranspose.shape}')
        X_XTranspose = torch.ones(1, 1, 1, self.kernel_num).to(device) * X_XTranspose.unsqueeze(-1)
        #print(f'X_XTranspose = {X_XTranspose.shape}')

        # Calculate Co-variance matrix Sigma_out
        Sigma_out = torch.log(1 + torch.exp(self.w_sigma)) * X_XTranspose
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        # Reshape for pytorch implementation
        Sigma_out = Sigma_out.permute(0,3,1,2)
        #print(Sigma_out)

        return mu_out, Sigma_out

class VDPMaxPooling(nn.Module):
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad=0):
        super(VDPMaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad

    def forward(self, mu_in, Sigma_in):
        batch_size, num_channel, hw_in, hw_in = mu_in.size()
        #print(f'Mu_in: {mu_in}')
        mu_out, argmax_out = F.max_pool2d(mu_in, kernel_size=self.pooling_size, stride=self.pooling_stride, padding=self.pooling_pad,return_indices = True)
        #print(f'Mu_out: {mu_out}')
        #print(f'Mu_out: {mu_out.shape}')
        argmax_out = argmax_out.view(batch_size,num_channel,argmax_out.size(2)*argmax_out.size(3),-1)
        argmax_out = argmax_out.squeeze(-1)
        #print(f'argmax_out: {argmax_out}')
        #print(argmax_out.shape)
        #print(f'Sigma_in: {Sigma_in} ')
        #print("\n")
        Sigma_out = torch.empty(batch_size, num_channel, argmax_out.size(-1),argmax_out.size(-1))
        #print(Sigma_out.shape)
        for batch in range(batch_size):
              for channel in range(num_channel):
                    m1 = Sigma_in[batch, channel, argmax_out[batch,channel,:], :]
                    m2 = m1[:,argmax_out[batch,channel,:]]
                    Sigma_out[batch,channel,:,:] = m2

        #print(f'Sigma_out: {Sigma_out}')

        return mu_out, Sigma_out

class VDPFlattenAndFC(nn.Module):
    def __init__(self, in_feature, units):
        super(VDPFlattenAndFC, self).__init__()
        self.in_feature = in_feature
        self.units = units
        self.w_mu = nn.Parameter(nn.init.normal_(torch.empty(self.in_feature,
                                             self.units), mean= 0.0 , std = 0.05),requires_grad = True) # shape = []
        self.w_sigma = nn.Parameter(torch.full((self.units,), -2.2),requires_grad = True)


    def forward(self, mu_in, Sigma_in):
        #Initialize weight parameters

        batch_size, num_channel, height, width = mu_in.size()
        #print(mu_in.shape)
        mu_flatt = torch.reshape(mu_in, (batch_size, -1))
        #print(mu_flatt.shape)
        #mu_flatt = mu_in.view(batch_size, -1)
        mu_out = torch.matmul(mu_flatt, self.w_mu)
        #print(f'mu_out = {mu_out.shape}')

        fc_weight_mu1 = torch.reshape(self.w_mu, (num_channel, height*width, self.units)).to(device)
        #fc_weight_mu1 = self.w_mu.view(num_channel, height * width, self.units)
        fc_weight_mu1T = fc_weight_mu1.permute(0, 2, 1)
        #print(f'fc_weight_mu1T = {fc_weight_mu1T.shape}')
        #sigma_in1 = Sigma_in.permute(0, 3, 1, 2)
        Sigma_1 = torch.matmul(torch.matmul(fc_weight_mu1T.to(device), Sigma_in.to(device)), fc_weight_mu1.to(device))
        Sigma_1 = Sigma_1.sum(dim=1).to(device)
        #print(f'Sigma_1 = {Sigma_1.shape}')


        diag_elements = torch.diagonal(Sigma_in, dim1=2, dim2=3)
        tr_sigma_b = diag_elements.sum(dim=2, keepdim=True)
        tr_sigma_b = tr_sigma_b.sum(dim = 1)

        #print(tr_sigma_b.shape)

        tr_sigma_h_sigma_b = torch.log(1. + torch.exp(self.w_sigma)).to(device) * tr_sigma_b.to(device)
        Sigma_2 = torch.diag_embed(tr_sigma_h_sigma_b)
        #print(Sigma_2.shape)

        mu_bT_mu_b = torch.sum(mu_flatt * mu_flatt, dim=1, keepdim=True)
        #print(mu_bT_mu_b.shape)
        mu_bT_sigma_h_mu_b = torch.log(1. + torch.exp(self.w_sigma)).to(device) * mu_bT_mu_b.to(device)
        Sigma_3 = torch.diag_embed(mu_bT_sigma_h_mu_b)
        #print(Sigma_3.shape)

        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3

        #Sigma_out[torch.isnan(Sigma_out)] = 0.0
        #Sigma_out[torch.isinf(Sigma_out)] = 0.0
        #Sigma_out = Sigma_out.clone().detach()
        #Sigma_out.diagonal(dim1=2, dim2=3).abs_()
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)

        return mu_out, Sigma_out

class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, mu_in, Sigma_in):
       # print(f"MU_in: {mu_in.shape}")
        mu_out = F.softmax(mu_in, dim=-1)
       # print(f"MU after softmax: {mu_out.shape}")
        pp1 = mu_out.unsqueeze(2)
        #print(pp1.shape)
        pp2 = mu_out.unsqueeze(1)
        #print(pp2.shape)
        ppT = torch.matmul(pp1, pp2)
        p_diag = torch.diag_embed(mu_out)
        #print(p_diag.shape)
        grad = p_diag - ppT
        Sigma_out = torch.matmul(grad, torch.matmul(Sigma_in, grad.transpose(1, 2)))
        #print(Sigma_out)
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        #Sigma_out[torch.isnan(Sigma_out)] = 0.0
        #Sigma_out[torch.isinf(Sigma_out)] = 0.0
        #Sigma_out = torch.nan_to_num(Sigma_out)
        #Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        #Sigma_out = torch.diag_embed(torch.abs(torch.diagonal(Sigma_out, dim1=-2, dim2=-1)))
        
        return mu_out, Sigma_out

def activation_function_Sigma(gradi, Sigma_in):
    batch_size, channels = gradi.size(0), gradi.size(1)
    gradient_matrix = torch.reshape(gradi,(batch_size, channels, -1))
    #gradient_matrix = gradi.view(batch_size, -1, channels)
    grad1 = gradient_matrix.unsqueeze(-1)
    grad_square = torch.matmul(grad1, grad1.permute(0,1,3,2))
    #grad_square = grad_square.transpose(1, 3).transpose(1, 2)
    sigma_out = Sigma_in * grad_square
    return sigma_out

class VDPReLU(nn.Module):
    def __init__(self):
        super(VDPReLU, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_out = F.relu(mu_in)
        #print(mu_out.shape)
        #with torch.autograd.set_grad_enabled(True):
        mu_in.requires_grad_(True)
        out = F.relu(mu_in)
        #gradi= out.sum().backward()
        #gradi = out.grad
        gradi = torch.autograd.grad(out.sum(), mu_in)[0]
        #gradi = torch.autograd.grad(out, mu_in, grad_outputs=torch.ones_like(out), create_graph=True)[0]
        #print(gradi.shape)
        Sigma_out = activation_function_Sigma(gradi, Sigma_in)
        return mu_out, Sigma_out
    
class DensityPropCNN(nn.Module):
    def __init__(self, kernel_size, num_kernel, pooling_size, pooling_stride, pooling_pad, units, input_channels):
        super(DensityPropCNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.units = units
        self.input_channels = input_channels
        #input = 1,1,28,28
        self.conv1 = VDPFirstConv(kernel_size=self.kernel_size, kernel_num=self.num_kernel, kernel_stride=1, padding="valid",input_channels = self.input_channels) # 1,16,24,24
        self.relu1 = VDPReLU() # 1,16,24,24
        self.maxpooling1 = VDPMaxPooling(pooling_size=self.pooling_size, pooling_stride=self.pooling_stride, pooling_pad=self.pooling_pad) # 1,16,12,12 = 4302
        self.fc1 = VDPFlattenAndFC( 2304 , self.units) # 1, 1, 10
        self.mysoftmax = MySoftmax()

    def forward(self, inputs):
        mu1, sigma1 = self.conv1(inputs)
        mu2, sigma2 = self.relu1(mu1, sigma1)
        mu3, sigma3 = self.maxpooling1(mu2, sigma2)
        mu4, sigma4 = self.fc1(mu3, sigma3)
        mu_out, Sigma_out = self.mysoftmax(mu4, sigma4)
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        return mu_out, Sigma_out
    
def main_function(input_dim=28, num_kernels=32, kernels_size=5, maxpooling_size=2, maxpooling_stride=2, maxpooling_pad=0, class_num=10 , batch_size=8,
        epochs =2, lr=0.0001, lr_end = 0.00001, kl_factor = 0.01, Random_noise=False, gaussain_noise_std=0.5, Adversarial_noise=False, epsilon = 0, adversary_target_cls=3, Targeted=False,
        Training = True , continue_training = False,  saved_model_epochs=50):
    
    mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    x_train = datasets.MNIST(root='data', train=True,
                                    transform=mnist_transform,
                                    target_transform=torchvision.transforms.Compose([
                                    lambda x:torch.LongTensor([x]),
                                    lambda x:F.one_hot(x,10)]),
                                    download=True)
    x_test = datasets.MNIST(root='data', train=False,
                                    transform=mnist_transform,
                                    target_transform=torchvision.transforms.Compose([
                                    lambda x:torch.LongTensor([x]),
                                    lambda x:F.one_hot(x,10)]),
                                    download=True)
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
    num_train_steps = epochs * int(len(x_train) /batch_size)

    cnn_model = DensityPropCNN(kernel_size= kernels_size, num_kernel= num_kernels, 
                               pooling_size= maxpooling_size, pooling_stride= maxpooling_stride, pooling_pad= maxpooling_pad, 
                               units= class_num, input_channels = 1).to(device)

    def getRegLoss(model):
        kl_factor = 0.01
        reg_loss = torch.tensor(0.).to(device)
        for name, param in model.named_parameters():
            if name == 'conv1.w_mu' or name == 'fc1.w_mu':
                reg_loss += torch.norm(param) ## L2 reg
            if name == 'conv1.w_sigma':
                f_s = torch.nn.functional.softplus(param)
                reg_loss += ((1 * kernels_size* kernels_size)*torch.mean(f_s-torch.log(f_s)-1.))
            elif name == 'fc1.w_sigma':
                f_s = torch.nn.functional.softplus(param)
                reg_loss += (2304*torch.mean(f_s-torch.log(f_s)-1.)) ##Custom reg
        return kl_factor*reg_loss
    
    def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):    
        y_pred_sd_ns = y_pred_sd 
        u, s, v = torch.svd(y_pred_sd_ns,some=False)	  
        s_ = s + 1.0e-3
        inv = 1./s_
        if torch.isinf(inv).any():
            inv = torch.where(torch.isinf(inv), torch.zeros_like(inv), inv)
        s_inv = torch.diag_embed(inv) 
        y_pred_sd_inv = torch.matmul(torch.matmul(v, s_inv), u.permute(0, 2, 1))
        mu_ = y_test - y_pred_mean
        mu_sigma = torch.matmul(mu_.unsqueeze(1), y_pred_sd_inv)     
        loss1 = torch.squeeze(torch.matmul(mu_sigma ,  mu_.unsqueeze(2)))
        loss1 = torch.where(torch.isnan(loss1), torch.zeros_like(loss1), loss1) 
        loss1 = torch.where(torch.isinf(loss1), torch.zeros_like(loss1), loss1) 
        loss2 = torch.mean(torch.sum(torch.log(s_), dim=-1))
        loss = torch.mean(loss1 + loss2)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss) 
        loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)   
        return loss

    

    optimizer = optim.Adam(cnn_model.parameters(), lr=lr) 
    learning_rate_fn = optim.lr_scheduler.PolynomialLR(optimizer, num_train_steps, power=2.)

    def train(model, device, train_loader, optimizer, epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            update_progress(batch_idx/ int(len(x_train) /batch_size) )
            target = target.squeeze(1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            mu_out, sigma = model(data)
            loss_final = nll_gaussian(target, mu_out,  torch.clamp(sigma, min=-1e+5, max=1e+5), class_num , batch_size)
            #print(batch_idx,loss_final.item())
            regularization_loss = getRegLoss(model)
            #if regularization_loss == 'nan':
            #    print('Reg Loss is NaN')
            loss = 0.5 * (loss_final + regularization_loss)           
            loss.backward() 
            optimizer.step()
            gradients = [param.grad for param in cnn_model.parameters()]
            gradients = [(torch.where(torch.isnan(grad), torch.tensor(1.0e-5).expand_as(grad).to(device), grad)) for grad in gradients]
            gradients = [(torch.where(torch.isinf(grad), torch.tensor(1.0e-5).expand_as(grad).to(device), grad)) for grad in gradients] 
            #print(batch_idx,loss.item())
            #if loss.item() == 'nan':
            #   print('loss.item is NaN')

    #corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
    #accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))                            
    #acc1+=accuracy.numpy()     

        print(f'Train Epoch: {epoch + 1} loss: {loss.item()}')
        return loss, mu_out, sigma, gradients

    def test(model,device,test_loader):
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                update_progress(batch_idx/ int(len(x_test) /batch_size))
                target = target.squeeze(1)
                data, target = data.to(device), target.to(device)
                mu_out, sigma = model(data)
                loss_final = nll_gaussian(target, mu_out,  torch.clamp(sigma, min=-1e+4, max=1e+4), class_num , batch_size)
                regularization_loss = getRegLoss(model)
                loss = 0.5 * (loss_final + regularization_loss)           
                #loss.backward() 
                #optimizer.step()
                #gradients = [param.grad for param in cnn_model.parameters()]
                #gradients = [(torch.where(torch.isnan(grad), torch.tensor(1.0e-5).expand_as(grad).to(device), grad)) for grad in gradients]
                #gradients = [(torch.where(torch.isinf(grad), torch.tensor(1.0e-5).expand_as(grad).to(device), grad)) for grad in gradients]
            return loss, mu_out, sigma
        
    train_acc = np.zeros(epochs) 
    valid_acc = np.zeros(epochs)
    train_err = np.zeros(epochs)
    valid_error = np.zeros(epochs)
    for epoch in range(epochs):
        acc1 = 0
        err1 = 0
        tr_no_steps = 0
        loss, mu_out, sigma, gradients = train(cnn_model, device = device, train_loader = train_loader, optimizer = optimizer, epochs = epoch)
        err1+= loss.cpu().numpy() 
        corr = torch.equal(torch.argmax(mu_out, axis=1),torch.argmax(y,axis=1))
        accuracy = torch.mean(corr.float())                            
        acc1+=accuracy.numpy()
        tr_no_steps+=1 
    train_acc[epoch] = acc1/tr_no_steps
    train_err[epoch] = err1/tr_no_steps        
    print('Training Acc  ', train_acc[epoch])
    print('Training error  ', train_err[epoch]) 

    torch.save(cnn_model.state_dict(), PATH)

if __name__ == '__main__':
    main_function()     
    