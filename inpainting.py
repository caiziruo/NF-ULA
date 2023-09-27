from ULA_NF import *

if __name__ == "__main__":
    ULA_NF_test = ULA_NF(problem = 'inpainting', noise_variance = 0.0004, step_size = 5e-5, alpha = 2.0, sample_size = 10000, figure_name = 'face4', load_observation = True, projection_C = 100.0, lambda_Moreau = 5)
        
    pt_NF = "./glow_pytorch_rosinality/checkpoint/model_000212.pt"
    # ULA_NF_test.ULA_NF_sample(start_from = "observation.pt", save_samples = False, load_pt = pt_NF) # 
    ULA_NF_test.ULA_NF_sample(start_from = "sample_ULA_NF.pt", save_samples = False, load_pt = pt_NF) # 
    # ULA_NF_test.Quantification(samples = 'samples/ULA_NF_Samples.pt', colorbar_range = 0.12, algorithm = 'ULA_NF', network = 'Glow')

    pt_PnP = './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch8_noise5_PSNR41.03_SSIM0.98.pth'
    # ULA_NF_test.PnP_ULA_sample(start_from = "observation.pt", save_samples = False, load_pt = pt_PnP) # 
    # ULA_NF_test.PnP_ULA_sample(start_from = "sample_ULA_NF.pt", save_samples = False, load_pt = pt_PnP) # 
    # ULA_NF_test.PnP_ULA_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, load_pt = pt_PnP) # 
    # ULA_NF_test.Quantification(samples = 'samples/PnP_ULA_Samples.pt', colorbar_range = 0.12, algorithm = 'PnP_ULA', network = 'realSN_DnCNN')



# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch8_noise5_PSNR41.03_SSIM0.98.pth'  # figure 3
# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch7_noise5_PSNR41.06_SSIM0.98.pth'  # figure 1

# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch8_noise5_PSNR41.03_SSIM0.98.pth'  # figure 3
# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch10_noise5_PSNR41.09_SSIM0.98.pth)'  # figure 2

# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch8_noise5_PSNR41.03_SSIM0.98.pth'  # figure 3

# './Provable_Plug_and_Play/training/logs/2023-01-13/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch8_noise5_PSNR41.03_SSIM0.98.pth'  # figure 4
