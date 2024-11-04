import numpy as np
import os
import cv2

def Threshold(input_file_dic, output_file_dic):
    files_list = os.listdir(input_file_dic)
    for k in files_list:
        data = np.load(os.path.join(input_file_dic, k))
        label = np.zeros((3,360,500))
        Z1 = data['Z1']
        V1 = data['V1']
        W1 = data['W1']
        SNR1 = data['SNR1']
        R = np.zeros((360,500))
        for j in range(500):
            R[:,j] = j
        for i in range(360):
            for j in range(0,500):           
                if np.isnan(Z1[i,j]):
                    label[2,i,j] = 1
                else:
                    if 0.2 > V1[i,j] > -0.2:
                        label[0,i,j] = 1
                    else:
                        label[1,i,j] = 1
                    if 1 > V1[i,j] > -1 and W1[i,j] > 0.4 and Z1[i,j] < -5:
                        label[0,i,j] = 0
                        label[1,i,j] = 1
                    if (Z1[i,j]>18 or SNR1[i,j]>18) and (V1[i,j]>0.2 or V1[i,j]<-0.2):
                        label[0,i,j] = 0
                        label[1,i,j] = 1
                    try:
                        kernel = V1[i-2:i+3,j-2:j+3]
                        if  0.2 > V1[i,j] > -0.2 and np.isnan(kernel).sum() <= 2 and np.nanmax(kernel) > 0.1 and np.nanmin(kernel) < -0.1 and np.nanmax(kernel)-np.nanmin(kernel) > 0.4:
                            label[0,i,j] = 0
                            label[1,i,j] = 1
                    except:pass

        kernel = np.ones((3,3))
        label[1,:,200:] = cv2.erode(label[1,:,200:], kernel, iterations=1)    
        label[1,:,200:] = cv2.dilate(label[1,:,200:], kernel, iterations=3)
        for i in range(360):
            for j in range(500):
                if np.isnan(Z1[i,j]):
                    label[0,i,j]=0
                    label[1,i,j]=0
                    label[2,i,j]=1
                elif label[1,i,j]==1:
                    label[0,i,j]=0
                elif label[1,i,j]==0:
                    label[0,i,j]=1

        flabel = np.zeros((360,500))
        for i in range(360):
            for j in range(500):
                if label[0,i,j]==1:
                    flabel[i,j]=0
                else:
                    flabel[i,j]=1
                if label[2,i,j]==1:
                    flabel[i,j]=2
                    
        np.savez(os.path.join(output_file_dic, "Label_" + k[:-4] + ".npz"), label=flabel)