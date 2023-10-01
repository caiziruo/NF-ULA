from ULA_NF import *

if __name__ == "__main__":
    ULA_NF_test = ULA_NF(problem = 'deblurring', noise_variance = 0.0004, step_size = 5e-5, alpha = 1.5, sample_size = 11000, figure_name = 'face1', load_observation = True, projection_C = 100.0, lambda_Moreau = 5e-5)

    pt_NF = "./glow_pytorch_rosinality/checkpoint/model_000212.pt"
    # ULA_NF_test.NF_Lipschitz_estimation(load_pt = pt_NF)
    # ULA_NF_test.ULA_NF_sample(start_from = "observation.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_NF) # 
    # ULA_NF_test.ULA_NF_sample(start_from = "sample_ULA_NF.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_NF) # 
    # ULA_NF_test.Quantification(samples = 'samples/ULA_NF_Samples.pt', colorbar_range = 0.08, algorithm = 'ULA_NF', network = 'Glow')

    pt_PnP = './Provable_Plug_and_Play/training/logs/2023-01-04/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch10_noise5_PSNR40.57_SSIM0.98.pth'
    # ULA_NF_test.PnP_ULA_sample(start_from = "observation.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_PnP) # 
    # ULA_NF_test.PnP_ULA_sample(start_from = "sample_ULA_NF.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_PnP) # 
    # ULA_NF_test.PnP_ULA_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, save_samples_size = 10000, load_pt = pt_PnP) # 
    # ULA_NF_test.Quantification(samples = 'samples/PnP_ULA_Samples.pt', colorbar_range = 0.08, algorithm = 'PnP_ULA', network = 'realSN_DnCNN')


    # ULA_NF_test.PnP_ULA_DnCNN_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, save_samples_size = 10000, load_pt = 'KAIR/denoising/dncnn25/models/500000_G.pth') # 

    # ULA_NF_test.PnP_ULA_DRUnet_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, save_samples_size = 10000, load_pt = 'KAIR/denoising/drunet/models/500000_G.pth') # 
    # ULA_NF_test.PnP_ULA_DRUnet_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, save_samples_size = 10000, load_pt = 'DPIR/model_zoo/drunet_color.pth') # 
    # ULA_NF_test.Quantification(samples = 'samples/PnP_ULA_DRUnet_Samples.pt', colorbar_range = 0.08, algorithm = 'PnP_ULA', network = 'DRUnet')
    
    # ULA_NF_test.ACF(wavelet = True)
    # ULA_NF_test.evolution_PSNR()

# './Provable_Plug_and_Play/training/logs/2023-01-04/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch10_noise5_PSNR40.57_SSIM0.98.pth'
# './Provable_Plug_and_Play/training/logs/2023-01-04/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch11_noise5_PSNR40.53_SSIM0.98.pth'
# 'KAIR/denoising/drunet/models/250000_G.pth'
# 'DPIR/model_zoo/drunet_color.pth'
