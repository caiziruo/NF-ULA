from __future__ import division
import numpy as np
from math import sqrt
from numpy.lib.function_base import gradient
from scipy.ndimage import filters
import cv2
import time
import random
from numpy import linalg as LA
import os
import sys
import os.path
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 7"
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, utils
from glow_pytorch_rosinality.model import Glow
from torch_radon import RadonFanbeam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=64, type=int, help="batch size")
parser.add_argument("--iter", default=10000, type=int, help="maximum iterations")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=5, type=int, help="number of blocks")
parser.add_argument("--no_lu", action="store_true", help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
parser.add_argument("--n_channel", default=3, type=int, help="channels of image")
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=16, type=int, help="number of samples")
parser.add_argument("--load_pt", default=212, type=int, help="load from checkpoint")

def psnr(img1, img2):
    # img2: reference image
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def Motion_blur_kernel_2d(kernel_size = 9):
    if (kernel_size % 2 == 0): raise Exception("kerner_size should be odd.")
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int(kernel_size / 2)] = np.ones(kernel_size) / kernel_size
    return kernel

class ULA_NF(object):
    def __init__(self, problem = 'deblurring', noise_variance = 0.0004, step_size = 1e-6, alpha = 1.0, sample_size = 10000, figure_name = 'figure_name', load_observation = True, projection_C = 100.0, lambda_Moreau = 0.1):
        self.problem                    = problem
        self.figure_name                = figure_name
        print("Restoration of ", self.figure_name)
        self.path                       = self.problem + '/' + figure_name + '/' 
        # ##################### Read Image by torchvision.
        self.OriginalFigure             = torchvision.io.read_image( self.path + figure_name + '.png').type(torch.float32)
        # ##################### Read Image by open-CV.
        # self.OriginalFigure             = cv2.imread(self.path + figure_name + '.png', flags = -1).astype('f')
        # self.OriginalFigure             = torch.Tensor(self.OriginalFigure)
        # #####################
        if len(self.OriginalFigure.shape) == 2: self.OriginalFigure = self.OriginalFigure.unsqueeze(dim = 0)
        self.channels, self.height, self.width = self.OriginalFigure.shape
        self.OriginalFigure             = self.OriginalFigure.unsqueeze(dim = 0).to(device) / 255.0
        print("Original image size: ", self.OriginalFigure.shape)

        self.noise_variance 			= noise_variance
        self.step_size 			    	= step_size
        self.alpha 					    = alpha
        self.sample_size 				= sample_size
        self.projection_C               = projection_C
        self.lambda_Moreau              = lambda_Moreau

        if self.problem == 'deblurring':
            kernel  = Motion_blur_kernel_2d(kernel_size = 9)
            weights = torch.tensor(kernel, dtype = torch.float32).to(device)
            weights = weights.view(1, 1, 9, 9).repeat(self.channels, 1, 1, 1)
            # ############################## Forward operator and adjoint: self.A, self.AT
            self.A 	= lambda x: torch.nn.functional.conv2d(x, weights, padding = 'same', groups = self.channels)
            self.AT	= lambda x: torch.nn.functional.conv2d(x, weights, padding = 'same', groups = self.channels)
        elif self.problem == 'inpainting':
            # ############################## Generate a mask
            # mask = torch.cuda.FloatTensor(1, 1, self.height, self.width).uniform_() > 0.8
            # torch.save(mask, self.problem + '/mask.pt')
            self.mask = torch.load(self.problem + '/mask.pt').to(device)
            self.A 	= lambda x: self.mask * x
            self.AT = lambda x: self.mask * x
        elif self.problem == 'CT_Gaussian_noise':
            angles = np.linspace(np.pi * 0.1, np.pi * 0.9, 144).astype(np.float32)
            # angles = np.linspace(0 * np.pi, 2.0 * np.pi, 64).astype(np.float32)
            self.radon_fanbeam = RadonFanbeam(resolution = 362, angles = angles, 
                    source_distance = 512, det_distance = 512, 
                    det_count= 512, det_spacing=4, clip_to_circle=False)

            self.A  = lambda x: self.radon_fanbeam.forward(x)
            self.AT = lambda x: self.radon_fanbeam.backward(x)
        elif self.problem == 'CT_Poisson_noise':
            angles = np.linspace(np.pi * 0.1, np.pi * 0.9, 144).astype(np.float32)
            self.radon_fanbeam = RadonFanbeam(resolution = 362, angles = angles, 
                    source_distance = 512, det_distance = 512, 
                    det_count= 512, det_spacing=4, clip_to_circle=False)

            self.A  = lambda x: self.radon_fanbeam.forward(x)
            self.AT = lambda x: self.radon_fanbeam.backward(x)
            photons_per_pixel = 4096
            mu_max = 0.05

            print('max(Ax): ', torch.max(self.A(self.OriginalFigure)))

            # ##### Beer Lambert law ####### Generate observation if no existing observation.
            if load_observation == False:
                proj_noisy = torch.poisson( photons_per_pixel * torch.exp(- self.A(self.OriginalFigure) * mu_max) )
                pre_log = proj_noisy / photons_per_pixel
                sinogram_noisy = - torch.log(pre_log) / mu_max
                utils.save_image( sinogram_noisy, self.path + 'observation.png', normalize=True)
                torch.save(  sinogram_noisy, self.path + 'observation.pt')
            # ############################## load the observation and define the data fidelity.
            self.observation = torch.load(self.path + 'observation.pt').to(device)
            self.phi  = lambda x: torch.sum(  torch.exp( - self.A(x) * mu_max ) * photons_per_pixel 
                + torch.exp(-self.observation * mu_max) * photons_per_pixel * (self.A(x) * mu_max - math.log(photons_per_pixel)  ) )
            return 
            
        
        print('max(Ax): ', torch.max(self.A(self.OriginalFigure)))

        # ############################## Generate observation if no existing observation.
        if load_observation == False:
            self.observation = self.A(self.OriginalFigure) + sqrt(self.noise_variance) * torch.randn( self.A(self.OriginalFigure).shape ).to(device)
            utils.save_image(self.observation, self.path + 'observation.png', normalize=True)
            torch.save( self.observation, self.path + 'observation.pt')
            # self.observation = torchvision.io.read_image(self.path + 'observation.png').type(torch.float32).unsqueeze(dim = 0).to(device)
            # ############################## compute FBP
            # filtered_sinogram = self.radon_fanbeam.filter_sinogram(self.observation)
            # fbp = self.radon_fanbeam.backward(filtered_sinogram)
            # print(fbp.shape)
            # utils.save_image(  fbp, self.path + 'fbp.png', normalize=False)
            # torch.save( fbp, self.path + 'fbp.pt')

        # ############################## Load the observation 
        self.observation = torch.load(self.path + 'observation.pt').to(device)
        print("Observation size: ", self.observation.shape)
        
        # ############################## define the data fidelity.
        self.phi  = lambda x: torch.sum( (self.A(x) - self.observation)**2 ) / (2.0 * self.noise_variance)
    # def phi(self, x): return torch.sum( (self.A(x) - self.observation)**2 ) / (2.0 * self.noise_variance)

    def NF_Lipschitz_estimation(self, load_pt):
        ###################################################################### Initialize the net
        args = parser.parse_args()
        print(args)
        net_single = Glow(args.n_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
        net = nn.DataParallel(net_single)
        net = net.to(device)
        print("Load pt: ", load_pt)
        net.load_state_dict(torch.load(load_pt))
        net.eval()
        print('#  parameters:', sum(param.numel() for param in net.parameters()))

        def function_f(input, model):
            model.zero_grad()
            log_p, logdet, _ = model(input)
            model.zero_grad()
            gradients = torch.autograd.grad(outputs = log_p, inputs = input,
                                grad_outputs = torch.ones(log_p.size(), device = input.device),
                                create_graph = True, retain_graph = True, only_inputs = True)[0]
            model.zero_grad()
            return  gradients
        
        # def function_f(x, model):
        #     y = x * x
        #     gradients = torch.autograd.grad(outputs = y, inputs = x,
        #                         grad_outputs = torch.ones(y.size(), device = x.device),
        #                         create_graph = True, retain_graph = True, only_inputs = True)[0]
        #     return  gradients

        for n in range(100):
            random_walk = 0.02 * torch.randn( 1, self.channels, self.height, self.width ).to(device)
            X = torch.rand(1, self.channels, self.height, self.width ).to(device) - 0.5

            ############################## For Glow-rosinality, input images should be in [-0.5, 0.5]
            X1 = X.clone().detach().requires_grad_()
            X2 = X.clone().detach().requires_grad_()
            # X2 = (X + random_walk).clone().detach().requires_grad_()

            Y1 = function_f(X1, net)
            # Y2 = function_f(X1, net)
            Y2 = function_f(X2, net)

            print(   torch.norm(Y2 - Y1),  torch.norm(X2 - X1))


    
    def ULA_NF_sample(self, start_from, save_samples, save_samples_size, load_pt):
        # X_current = torchvision.io.read_image(self.path + start_from).type(torch.float32).unsqueeze(dim = 0).to(device)
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []
        
        ###################################################################### Initialize the net
        if self.problem == 'deblurring' or self.problem == 'inpainting':
            args = parser.parse_args()
            print(args)
            net_single = Glow(args.n_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
            net = nn.DataParallel(net_single)
            # net = net_single
            net = net.to(device)
            print("Load pt: ", load_pt)
            net.load_state_dict(torch.load(load_pt))
            net.eval()
            print('#  parameters:', sum(param.numel() for param in net.parameters()))

            def gradient_log_density(input, model):
                log_p, logdet, _ = model(input)
                gradients = torch.autograd.grad(outputs = log_p, inputs = input,
                                    grad_outputs = torch.ones(log_p.size(), device = input.device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]

                return  gradients
        else:
            raise Exception("Problem is not deblurring or inpainting.")
        ######################################################################

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)

            gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance
            X_current_requires_grad = (X_current - 0.5).clone().detach().requires_grad_()
            ############################## For Glow-rosinality, input images should be in [-0.5, 0.5]
            gradient_log_prior      = gradient_log_density(X_current_requires_grad, net)
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_ULA_NF.pt")
                utils.save_image( X_current, self.path + 'sample_ULA_NF.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/ULA_NF_Samples.pt')
        return 0

    def ULA_NF_CT_sample(self, start_from, save_samples, save_samples_size, load_pt):
        # X_current = torchvision.io.read_image(self.path + start_from).type(torch.float32).unsqueeze(dim = 0).to(device) / 255.0
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []

        ###################################################################### Initialize the net
        if self.problem == 'CT_Gaussian_noise' or 'CT_Poisson_noise':
            sys.path.append('./patchNR')
            from model import create_NF
            net = create_NF(5, 512, dimension=6**2)
            print("Load pt: ", load_pt)
            weights = torch.load(load_pt)
            net.load_state_dict(weights['net_state_dict'])
            net = net.to(device)
            net.eval()
            print('#  parameters:', sum(param.numel() for param in net.parameters()))
            
            from utils import patch_extractor
            input_im2pat = patch_extractor(6, pad=False, center=False)
            pad = [4]*4  

            def gradient_data_fidelity(input, model):
                log_p = - self.phi(input)
                gradients = torch.autograd.grad(outputs = log_p, inputs = input,
                                    grad_outputs = torch.ones(log_p.size(), device = input.device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]

                return  gradients
            def gradient_log_density(input, model):
                input_pad = nn.functional.pad(input, pad, mode='reflect')
                input_pat = input_im2pat(input_pad, 40000)
                pred_inv, log_det_inv = model(input_pat, rev=True)    
                log_p = - (torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv))
                gradients = torch.autograd.grad(outputs = log_p, inputs = input,
                                    grad_outputs = torch.ones(log_p.size(), device = input.device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]

                return  gradients
        else:
            raise Exception("problem is not CT_Gaussian_noise or CT_Poisson_noise")
        ######################################################################

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)
            # gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance

            X_current_requires_grad = X_current.clone().detach().requires_grad_()
            gradient_log_likelihood = gradient_data_fidelity(X_current_requires_grad, net)
            gradient_log_prior      = gradient_log_density(X_current_requires_grad, net)
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_ULA_NF.pt")
                torchvision.utils.save_image( X_current, self.path + 'sample_ULA_NF.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/ULA_NF_Samples.pt')
        return 0

    def PnP_ULA_sample(self, start_from, save_samples, save_samples_size, load_pt):
        # X_current = torchvision.io.read_image(self.path + start_from).type(torch.float32).unsqueeze(dim = 0).to(device)
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []

        ###################################################################### Initialize the net
        #################### real Spectral Normalization DnCNN.
        sys.path.append('./Provable_Plug_and_Play')
        from training.model.full_realsn_models import DnCNN
        model = DnCNN(channels=3, num_of_layers=17, lip=1.0, no_bn=False)
        print("Load pt: ", load_pt)
        model.load_state_dict( torch.load(load_pt) ) 
        model = model.cuda()
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        print('#  parameters:', sum(param.numel() for param in model.parameters()))
        ######################################################################

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)

            gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance
            X_output = X_current - model(X_current)
            X_output = torch.clamp( X_output, 0., 1.)
            gradient_log_prior      = ( X_output - X_current )  / (5. / 255.)**2
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_PnP_ULA.pt")
                utils.save_image( X_current, self.path + 'sample_PnP_ULA.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/PnP_ULA_Samples.pt')
        return 0

    def PnP_ULA_DnCNN_sample(self, start_from, save_samples, save_samples_size, load_pt):
        # X_current = torchvision.io.read_image(self.path + start_from).type(torch.float32).unsqueeze(dim = 0).to(device)
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []

        ###################################################################### Initialize the net
        #################### DnCNN without Lipschitz condition.
        sys.path.append('./KAIR')
        from models.network_dncnn import DnCNN 
        model = DnCNN(in_nc = 3, out_nc = 3, nc = 64, nb = 17, act_mode = 'BR')
        print("Load pt: ", load_pt)
        model.load_state_dict(torch.load(load_pt), strict=False)
        # model = model.to(device)
        model = model.cuda()
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        print('#  parameters:', sum(param.numel() for param in model.parameters()))
        ######################################################################

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)

            gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance
            X_output = model(X_current)
            X_output = torch.clamp( X_output, 0., 1.)
            gradient_log_prior      = ( X_output - X_current )  / (5. / 255.)**2
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk
            # X_next                  = torch.clamp(X_next, min = -1.0, max = 2.0) 

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_PnP_ULA_DnCNN.pt")
                utils.save_image( X_current, self.path + 'sample_PnP_ULA_DnCNN.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/PnP_ULA_DnCNN_Samples.pt')
        return 0
    
    def PnP_ULA_DRUnet_sample(self, start_from, save_samples, save_samples_size, load_pt):
        # X_current = torchvision.io.read_image(self.path + start_from).type(torch.float32).unsqueeze(dim = 0).to(device)
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []

        ###################################################################### Initialize the net
        #################### DRUnet without Lipschitz condition.
        sys.path.append('./DPIR')
        from models.network_unet import UNetRes
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        print("Load pt: ", load_pt)
        model.load_state_dict(torch.load(load_pt), strict=True)
        model = model.cuda()
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        print('#  parameters:', sum(param.numel() for param in model.parameters()))
        ######################################################################

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)

            gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance
            ######################################################################
            X_output = model(torch.cat((X_current, torch.FloatTensor([5./255.]).repeat(1, 1, X_current.shape[2], X_current.shape[3]).cuda()), dim=1))
            ######################################################################
            X_output = torch.clamp( X_output, 0., 1.)
            gradient_log_prior      = ( X_output - X_current )  / (5. / 255.)**2
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_PnP_ULA_DRUnet.pt")
                utils.save_image( X_current, self.path + 'sample_PnP_ULA_DRUnet.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/PnP_ULA_DRUnet_Samples.pt')
        return 0

    def PnP_ULA_CT_sample(self, start_from, save_samples, save_samples_size, load_pt):
        ######################################################################
        #################### DnCNN without Lipschitz condition.
        # sys.path.append('./KAIR')
        # from models.network_dncnn import DnCNN 
        # model = DnCNN(in_nc = 3, out_nc = 3, nc = 64, nb = 17, act_mode = 'BR')
        # model.load_state_dict(torch.load('KAIR/denoising/dncnn25/models/515000_G.pth'), strict=False)
        # model = model.to(device)
        # model.eval()
        # for k, v in model.named_parameters():
        #     v.requires_grad = False
        ######################################################################
        #################### real Spectral Normalization DnCNN.
        sys.path.append('./Provable_Plug_and_Play')
        from training.model.full_realsn_models import DnCNN
        model = DnCNN(channels=1, num_of_layers=17, lip=1.0, no_bn=False)
        print("Load pt: ", load_pt)
        model.load_state_dict( torch.load(load_pt) ) 

        model = model.cuda()
        model.eval()
        print('#  parameters:', sum(param.numel() for param in model.parameters()))
        for k, v in model.named_parameters():
            v.requires_grad = False

        def gradient_data_fidelity(input, model):
            log_p = - self.phi(input) 
            gradients = torch.autograd.grad(outputs = log_p, inputs = input,
                                grad_outputs = torch.ones(log_p.size(), device = input.device),
                                create_graph = True, retain_graph = True, only_inputs = True)[0]

            return  gradients
        
        print('#  parameters:', sum(param.numel() for param in model.parameters()))
        ######################################################################

        # X_current = torchvision.io.read_image(self.path + 'posterior_mean_PnP_ULA.png').type(torch.float32).unsqueeze(dim = 0).to(device)
        X_current = torch.load(self.path + start_from).to(device)
        X_posterior_mean = torch.zeros_like(X_current)
        samples = []

        print("Start drawing ", self.sample_size, " samples!")
        start = time.time()
        for n in range(self.sample_size):
            random_walk = sqrt(2 * self.step_size) * torch.randn( 1, self.channels, self.height, self.width ).to(device)
            # gradient_log_likelihood = - self.AT(self.A(X_current) - self.observation) / self.noise_variance

            ######################################################################
            # X_output = model(X_current)
            X_output = X_current - model(X_current)
            ######################################################################
            X_output = torch.clamp( X_output, 0., 1.)
            gradient_log_prior      = ( X_output - X_current )  / (5. / 255.)**2
            X_current_requires_grad = X_current.clone().detach().requires_grad_()
            gradient_log_likelihood = gradient_data_fidelity(X_current_requires_grad, model)
            projection_C            = 1 * ( torch.clamp(X_current, min = -(self.projection_C - 1), max = self.projection_C) - X_current ) / self.lambda_Moreau
            X_next = X_current + self.step_size * ( gradient_log_likelihood + self.alpha * gradient_log_prior + projection_C) + random_walk

            if n % 100 == 0: 
                data_term = self.phi(X_current)
                print(n, "\t", psnr(X_current, self.OriginalFigure).cpu().detach().numpy(), "\t",
                data_term.cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_likelihood).cpu().detach().numpy(), "\t",
                torch.norm(gradient_log_prior).cpu().detach().numpy(), "\t",
                torch.max(X_current).cpu().detach().numpy(), "\t",
                torch.min(X_current).cpu().detach().numpy())

            #  record the last save_samples_size samples
            if (n >= self.sample_size - save_samples_size and save_samples): samples.append( X_next.cpu().detach() )
            X_posterior_mean += X_next.detach()
            
            X_current = X_next.clone().detach()
            if (n % 1000 == 0 and save_samples):
                torch.save( X_current, self.path + "sample_PnP_ULA.pt")
                utils.save_image( X_current, self.path + 'sample_PnP_ULA.png', normalize=False)

        end = time.time()
        print("time: ", end - start)

        X_posterior_mean /= self.sample_size
        print("posterior mean psnr  ", psnr(  X_posterior_mean.squeeze(), self.OriginalFigure.squeeze() ) )

        if save_samples:
            samples = torch.cat( samples, dim = 0)
            torch.save(samples, self.path + 'samples/PnP_ULA_Samples.pt')
        return 0
    
    
    def Quantification(self, samples, colorbar_range, algorithm, network):
        if self.problem == 'deblurring' or self.problem == 'inpainting':
            print(  "observation psnr  ", psnr(  self.observation.squeeze(), self.OriginalFigure.squeeze() ) )
        samples = torch.load(self.path + samples).to(device)
        print(  'shape of samples ', samples.shape)
        samples_std = torch.std(samples, dim = 0)
        import matplotlib.pyplot as plt
        for i in range(samples_std.shape[0]):
            im = plt.imshow(samples_std[i, :, :].cpu())
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.clim(0.0, colorbar_range)
            plt.savefig(self.path + 'std_' + algorithm + '_' + network + str(i) + '.png')
            plt.close()
            # utils.save_image( samples_std[i, :, :], self.path + 'std_ULA_NF' + str(i) + '.png', normalize=True)
        posterior_mean = torch.mean(samples, dim = 0)
        print(algorithm,  "posterior mean psnr  ", psnr(  posterior_mean, self.OriginalFigure.squeeze() ) )
        utils.save_image( posterior_mean, self.path + 'posterior_mean_' + algorithm + '_' + network + '.png', normalize=False)
        del samples

    def evolution_PSNR(self):
        samples_ULA_NF = torch.load(self.path + "samples/ULA_NF_Samples.pt").to(device)
        samples_PnP_ULA = torch.load(self.path + "samples/PnP_ULA_Samples.pt").to(device)
        print(  'shape of samples_ULA_NF ', samples_ULA_NF.shape)
        print(  'shape of samples_PnP_ULA ', samples_PnP_ULA.shape)
        samples_size_ULA_NF  = samples_ULA_NF.shape[0]
        samples_size_PnP_ULA = samples_PnP_ULA.shape[0]

        mmse_ULA_NF  = torch.mean(samples_ULA_NF[-10000:, :, :, :], dim = 0).unsqueeze(dim = 0)
        mmse_PnP_ULA = torch.mean(samples_PnP_ULA[-10000:, :, :, :], dim = 0).unsqueeze(dim = 0)
        evolution_PSNR_ULA_NF  = []
        evolution_PSNR_PnP_ULA = []
        for i in range(samples_size_ULA_NF):
            evolution_PSNR_ULA_NF.append( psnr(samples_ULA_NF[i],  mmse_ULA_NF).cpu().detach().numpy() )

        for i in range(samples_size_PnP_ULA):
            evolution_PSNR_PnP_ULA.append(psnr(samples_PnP_ULA[i], mmse_PnP_ULA).cpu().detach().numpy())

        evolution_PSNR_ULA_NF = np.array(evolution_PSNR_ULA_NF)
        evolution_PSNR_PnP_ULA = np.array(evolution_PSNR_PnP_ULA)

        import matplotlib.pyplot as plt
        evolution_samples = min(samples_size_ULA_NF, samples_size_PnP_ULA)
        x_axis = range(evolution_samples)
        plt.xlabel('i')
        plt.ylabel('PSNR(x_i, x_mmse)')
        plt.plot(x_axis, evolution_PSNR_ULA_NF[:evolution_samples], color='r',marker='.',linestyle='-', linewidth=0.01)
        plt.plot(x_axis, evolution_PSNR_PnP_ULA[:evolution_samples], color='b',marker='.',linestyle='-.', linewidth=0.01)
        plt.legend(["NF-ULA", "PnP-ULA"], fontsize="20", loc ="lower right")
        plt.savefig(self.path + '/PSNR_x_mmse.png')
        plt.close()

        # x_axis = np.arange(0, 600, 0.3)
        # x_axis_PnP = np.arange(0, 450, 0.003)
        # plt.xlabel('Time(seconds)')
        # plt.ylabel('PSNR(x_i, x_mmse)')
        # plt.plot(x_axis,     evolution_PSNR_ULA_NF[:2000], color='r',marker='.',linestyle='-', linewidth=0.01)
        # plt.plot(x_axis_PnP, evolution_PSNR_PnP_ULA[:150000], color='b',marker='.',linestyle='-.', linewidth=0.01)
        # plt.legend(["NF-ULA", "PnP-ULA"], fontsize="20", loc ="lower right")
        # plt.savefig(self.path + '/PSNR_x_mmse_time.png')
        # plt.close()

    def ACF(self, wavelet = False):
        dir = self.path + '/acf'
        for f in os.listdir(dir):
            if f.endswith('.png'):
                os.remove(os.path.join(dir, f))

        lags = range(100)
        if self.problem == 'deblurring': lags = range(100)
        elif self.problem == 'inpainting': lags = range(100)
        elif self.problem == 'CT_Gaussian_noise' or self.problem == 'CT_Poisson_noise': lags = range(2000)

        samples_ULA_NF = torch.load(self.path + 'samples/ULA_NF_Samples.pt') 
        samples_PnP_ULA = torch.load(self.path + 'samples/PnP_ULA_Samples.pt')
        
        acf_height = self.height
        acf_width = self.width

        J = 1
        if wavelet:
            from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
            xfm = DWTForward(J=J, wave='db3', mode='zero')
            samples_ULA_NF, _ = xfm(samples_ULA_NF)
            samples_PnP_ULA, _ = xfm(samples_PnP_ULA)
            # print(samples_ULA_NF.shape)
            # utils.save_image( Yl, self.path + 'wavelet_Yl.png', normalize=False)

            acf_height /= 2**J
            acf_width /= 2**J
        
        samples_ULA_NF  = np.array(samples_ULA_NF)
        samples_PnP_ULA = np.array(samples_PnP_ULA)

        import statsmodels.api as sm
        import matplotlib.pyplot as plt
        for k in range(500):
            i = random.randint(1, acf_height - 2)
            j = random.randint(1, acf_width - 2)
            acorr_ULA_NF  = sm.tsa.acf(samples_ULA_NF[:, 0, i, j],  nlags = len(lags)-1)
            acorr_PnP_ULA = sm.tsa.acf(samples_PnP_ULA[:, 0, i, j], nlags = len(lags)-1)
            
            plt.xlabel('Lag')
            plt.ylabel('ACF')
            plt.plot(lags, acorr_ULA_NF, color='r',marker='.',linestyle='-', linewidth=1.0)
            plt.plot(lags, acorr_PnP_ULA, color='b',marker='.',linestyle='-.', linewidth=1.0)
            plt.legend(["NF-ULA", "PnP-ULA"], fontsize="20", loc ="upper right")
            plt.savefig(self.path + '/acf/acf_' + str(i) + '_' + str(j) + '.png')
            plt.close()

