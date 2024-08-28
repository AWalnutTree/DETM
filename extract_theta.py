import torch
import torch.nn as nn
import scipy.io 

device = "cpu"
ckpt = 'results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1'

with open(ckpt, 'rb') as f:
    model = torch.load(f)
model = model.to(device)


print('saving theta...')
with torch.no_grad():
    theta = model.q_theta().cpu().numpy()
    scipy.io.savemat(ckpt+'_theta.mat', {'values': theta}, do_compression=True)

# Print Theta Contents
print('Theta: ', theta.shape)
print(theta)
    