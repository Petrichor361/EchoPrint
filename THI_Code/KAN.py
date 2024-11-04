import torch
import numpy as np
import os
import joblib

from model import KAN_THI

def KAN_inference(input_file_dic, output_file_dic, device):
    file_list = os.listdir(input_file_dic)
    current_dictionary = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(current_dictionary, "Standard.pkl"))
    net0 = KAN_THI([6,16,32,16,8,1]).to(device)
    if torch.cuda.is_available() and device != torch.device('cpu'):
        print('now is', device )
        net0.load_state_dict(torch.load(os.path.join(current_dictionary, "KAN_THI_big.pth"), map_location=device))
    else:
        print('now is cpu')
        net0.load_state_dict(torch.load(os.path.join(current_dictionary, "KAN_THI_big.pth"), map_location=torch.device('cpu')))
    for k in file_list:
        data = np.load(os.path.join(input_file_dic, k))
        Z1=data['Z1']
        V1=data['V1']
        W1=data['W1']
        SNR1=data['SNR1']
        LDR=data['LDR']
        Height=np.zeros((720,500))
        for j in range(500):
            Height[:,j]=j
        image=np.stack([Z1,V1,W1,SNR1,LDR,Height])
        with torch.no_grad():
            net0.eval()
            image = torch.from_numpy(image).float().to(device)
            image = image.permute(1, 2, 0).reshape(720*500, 6)
            image = scaler.transform(image.cpu())
            input = torch.from_numpy(image).float().to(device)
            output = net0(input)
            output = output.reshape(720, 500)
            label = np.zeros((720,500))
            for i in range(720):
                for j in range(500):
                    if np.isnan(Z1[i,j]):
                        label[i,j] = 2
                    else:
                        if output[i,j] > 0.5:
                            label[i,j] = 0
                        else:
                            label[i,j] = 1
            np.savez(os.path.join(output_file_dic, "Label_" + k[:-4] + ".npz"), label=label)