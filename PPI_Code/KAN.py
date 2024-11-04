import torch
import numpy as np
import os
import joblib

from model import KAN_PPI

def KAN_inference(input_file_dic, output_file_dic, device):
    file_list = os.listdir(input_file_dic)
    current_dictionary = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(current_dictionary, "Standard.pkl"))
    net0 = KAN_PPI([11,16,32,16,8,1]).to(device)
    if torch.cuda.is_available() and device != torch.device('cpu'):
        print('now is', device)
        net0.load_state_dict(torch.load(os.path.join(current_dictionary, "KAN_PPI_big.pth"), map_location=device))
    else:
        print('now is cpu')
        net0.load_state_dict(torch.load(os.path.join(current_dictionary, "KAN_PPI_big.pth"), map_location=torch.device('cpu')))
    for k in file_list:
        data = np.load(os.path.join(input_file_dic, k))
        Z1=data['Z1']
        V1=data['V1']
        W1=data['W1']
        SNR1=data['SNR1']
        TDBZ=np.full((360, 500), np.nan)
        SIGN=np.full((360, 500), np.nan)
        SPIN=np.full((360, 500), np.nan)
        MDVE=np.full((360, 500), np.nan)
        SDVE=np.full((360, 500), np.nan)
        MDSW=np.full((360, 500), np.nan)
        R=np.zeros((360,500))
        for j in range(500):
            R[:,j]=j
        temp=Z1.copy()
        temp[:,:-1]=temp[:,:-1]-Z1[:,1:]
        for i in range(1,359):
            for j in range(1,499):
                if ~np.isnan(Z1[i,j]) and ~np.isnan(temp[i,j]):
                    TDBZ[i,j]=np.nansum(temp[i-1:i+2,j-1:j+2]**2)/9
                    SIGN[i,j]=np.nansum(np.where(temp[i-1:i+2,j-1:j+2] > 0, -1, 1))/9
                    SPIN[i,j]=np.nansum(np.where(abs(temp[i-1:i+2,j-1:j+2]) < 2, 0, 1))/9
                    MDVE[i,j]=np.nansum(V1[i-1:i+2,j-2:j+3])/15
                    SDVE[i,j]=np.nanstd(V1[i-1:i+2,j-2:j+3])
                    MDSW[i,j]=np.nansum(W1[i-1:i+2,j-2:j+3])/15
        image=np.stack([Z1,V1,W1,SNR1,R,TDBZ,SIGN,SPIN,MDVE,SDVE,MDSW])
        with torch.no_grad():
            net0.eval()
            image = torch.from_numpy(image).float().to(device)
            image = image.permute(1, 2, 0).reshape(360*500, 11)
            image = scaler.transform(image.cpu())
            input = torch.from_numpy(image).float().to(device)
            output = net0(input)
            output = output.reshape(360, 500)
            label = np.zeros((360,500))
            for i in range(360):
                for j in range(500):
                    if np.isnan(Z1[i,j]):
                        label[i,j] = 2
                    else:
                        if output[i,j] > 0.5:
                            label[i,j] = 0
                        else:
                            label[i,j] = 1
            np.savez(os.path.join(output_file_dic, "Label_" + k[:-4] + ".npz"), label=label)
            