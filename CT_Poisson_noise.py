from ULA_NF import *

if __name__ == "__main__":
    ULA_NF_test = ULA_NF(problem = 'CT_Poisson_noise', noise_variance = 1, step_size = 1e-6, alpha = 4000.0, sample_size = 10000, figure_name = 'figure1', load_observation = True, projection_C = 100.0, lambda_Moreau = 5e-5)

    pt_NF = './patchNR/patchNR_weights/weights_lung.pth'
    # ULA_NF_test.ULA_NF_CT_sample(start_from = "fbp.pt", save_samples = False, load_pt = pt_NF) # 
    # ULA_NF_test.ULA_NF_CT_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, load_pt = pt_NF) # 
    # ULA_NF_test.ULA_NF_CT_sample(start_from = "sample_ULA_NF.pt", save_samples = False, load_pt = pt_NF) # 
    # ULA_NF_test.Quantification(samples = 'samples/ULA_NF_Samples.pt', colorbar_range = 0.05, algorithm = 'ULA_NF', network = 'patchNR')

    
    PnP_ULA_test = ULA_NF(problem = 'CT_Poisson_noise', noise_variance = 1, step_size = 1e-6, alpha = 3.0, sample_size = 10000, figure_name = 'figure1', load_observation = True, projection_C = 100.0, lambda_Moreau = 5)

    pt_PnP = './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch50_noise5_PSNR37.58_SSIM0.97.pth'
    # PnP_ULA_test.PnP_ULA_CT_sample(start_from = "fbp.pt", save_samples = False, load_pt = pt_PnP) # 
    PnP_ULA_test.PnP_ULA_CT_sample(start_from = "sample_PnP_ULA.pt", save_samples = False, load_pt = pt_PnP) # 
    # ULA_NF_test.Quantification(samples = 'samples/PnP_ULA_Samples.pt', colorbar_range = 0.05, algorithm = 'PnP_ULA', network = 'realSN_DnCNN')





# './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch2_noise5_PSNR37.65_SSIM0.97.pth'
# './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch7_noise5_PSNR37.82_SSIM0.97.pth'
# './Provable_Plug_and_Play/training/logs/2023-01-20/DnCNN_Sigma5_17Layers_modeS_LipConst/epoch50_noise5_PSNR37.58_SSIM0.97.pth'
