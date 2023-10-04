from ULA_NF import *

######### load NF and PnP denoisers from pre-trained model, or load the model trained by yourself.
pt_NF = './patchNR/patchNR_weights/weights_lung.pth'
pt_PnP = './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch50_noise5_PSNR37.58_SSIM0.97.pth'
# pt_PnP = './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch47_noise5_PSNR37.82_SSIM0.97.pth'

# noise_variance   = 1.0           # do not change this if load_observation = True
# step_size        = 1e-6          # stepsize for NF-ULA and PnP-ULA. 
# alpha            = 4000.0        # regularization parameter of the prior.
# sample_size      = 10000         # the number of samples you want to generate.
# figure_name      = figure1, figure2, figure3, figure4
# load_observation = True          # True: load from existing observation.pt. False: generate another observation by y = Ax + noise.
# projection_C     = 100           # the projection onto the compact set [-(projection_C-1), projection_C]^d
# lambda_Moreau    = 5e-5          # parameter of the Moreau envelope of the projection C.

ULA_NF_test = ULA_NF(problem = 'CT_Poisson_noise', noise_variance = 1, step_size = 1e-6, alpha = 4000.0, sample_size = 10000, figure_name = 'figure1', load_observation = True, projection_C = 100.0, lambda_Moreau = 5e-5)

''' 
Run NF-ULA initializing from the fbp 'CT_Poisson_noise/figure_name/fbp.pt' (this will imply a burn-in period)
or from a sample 'CT_Poisson_noise/figure_name/sample_ULA_NF.pt'.
If save_samples == True, samples (with size save_samples_size) will be saved to 'CT_Poisson_noise/figure_name/samples/ULA_NF_Samples.pt', 
and the last sample will be saved to'CT_Poisson_noise/figure_name/sample_ULA_NF.pt'. 
'''
# ULA_NF_test.ULA_NF_CT_sample(start_from = "fbp.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_NF) # 
ULA_NF_test.ULA_NF_CT_sample(start_from = "sample_ULA_NF.pt", save_samples = True, save_samples_size = 10000, load_pt = pt_NF) # 

######### Calculate the posterior mean and std from existing samples generated by NF-ULA.
ULA_NF_test.Quantification(samples = 'samples/ULA_NF_Samples.pt', colorbar_range = 0.05, algorithm = 'ULA_NF', network = 'patchNR')

# noise_variance   = 1.0           # do not change this if load_observation = True
# step_size        = 1e-6          # stepsize for NF-ULA and PnP-ULA. 
# alpha            = 3.0           # regularization parameter of the prior.
# sample_size      = 10000         # the number of samples you want to generate.
# figure_name      = figure1, figure2, figure3, figure4
# load_observation = True          # True: load from existing observation.pt. False: generate another observation by y = Ax + noise.
# projection_C     = 100           # the projection onto the compact set [-(projection_C-1), projection_C]^d
# lambda_Moreau    = 5e-5          # parameter of the Moreau envelope of the projection C.

PnP_ULA_test = ULA_NF(problem = 'CT_Poisson_noise', noise_variance = 1, step_size = 1e-6, alpha = 3.0, sample_size = 10000, figure_name = 'figure1', load_observation = True, projection_C = 100.0, lambda_Moreau = 5e-5)

######### Run PnP-ULA using realSN-DnCNN. 
# PnP_ULA_test.PnP_ULA_CT_sample(start_from = "fbp.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_PnP) # 
PnP_ULA_test.PnP_ULA_CT_sample(start_from = "sample_PnP_ULA.pt", save_samples = True, save_samples_size = 10000, load_pt = pt_PnP) # 

######### Calculate the posterior mean and std from existing samples generated by PnP-ULA, realSN-DnCNN.
PnP_ULA_test.Quantification(samples = 'samples/PnP_ULA_Samples.pt', colorbar_range = 0.05, algorithm = 'PnP_ULA', network = 'realSN_DnCNN')

######### Calculate the autocorrelation function (ACF) of samples generated by NF-ULA and PnP-ULA. wavelet_type = 'YL' or 'YH'.
ULA_NF_test.ACF(wavelet = True, wavelet_type = 'YH', samples_NF = 'samples/ULA_NF_Samples.pt', samples_PnP = 'samples/PnP_ULA_Samples.pt')
