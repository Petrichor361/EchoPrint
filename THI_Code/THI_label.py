import numpy as np
import os
import cv2

def Threshold(input_file_dic, output_file_dic):
    files_list = os.listdir(input_file_dic)
    for k in files_list:
        data = np.load(os.path.join(input_file_dic, k))
        #label数组
        label=np.zeros((3,720,500))
        Z1=data['Z1']
        V1=data['V1']
        W1=data['W1']
        SNR1=data['SNR1']
        LDR=data['LDR']
        Height=np.zeros((720,500))
        for j in range(500):
            Height[:,j]=j
        for i in range(720):
            for j in range(500):           
                if np.isnan(Z1[i,j]):
                    label[2,i,j]=1
                else:
                    condition=0
                    if j > 100:
                        label[1,i,j]=1
                    else:
                        if Z1[i,j] < -25:
                            condition += 1
                        if V1[i,j] > -0.2:
                            condition += 1
                        if SNR1[i,j] < 10 and j < 30:
                            condition += 1
                        if LDR[i,j] > -25:
                            condition += 1
                        if condition >= 2:
                            label[0,i,j] = 1
                        else:
                            label[1,i,j] = 1
        
        label1 = label.copy()
        for i in range(4, 716):
            if np.isnan(Z1[i-2:i+3,:]).sum() < 2125 and np.isnan(Z1[i-2:i+3,:200]).sum() < 500:            
                for j in range(10,110):
                    if np.sum(label1[1,i-4:i+5,j-4:j+5]) > 35:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0
            else:
                for j in range(10,100):
                    if np.sum(label1[1, i-4:i+5, j-10:j+11]) > 130:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0

            for j in range(100,200):
                if np.sum(label1[1,i-2:i+3,j-2:j+3]) > 13:
                    label[1,i,j]=1
                    label[0,i,j]=0
                    label[2,i,j]=0
                else:
                    label[0,i,j]=1
                    label[1,i,j]=0
                    label[2,i,j]=0
        # down-edge
        label1=label.copy()
        for i in range(4, 716):
            if np.isnan(Z1[i-2:i+3,:]).sum() < 2125: 
                for j in range(0,10):
                    if np.sum(label1[1, i-4:i+5, j:j+5]) > 23:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0
            else:
                for j in range(0,10):
                    if np.sum(label1[1, i-4:i+5, j:j+11]) > 60:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0
        # left-edge
        label1=label.copy()
        for i in range(0, 4):
            if np.isnan(Z1[0:4, :]).sum() < 1200:            
                for j in range(10,110):
                    if np.sum(label1[1,i:i+5,j-4:j+5])>18:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0

            else:
                for j in range(10,110):
                    if np.sum(label1[1,i:i+5,j-10:j+11])>60:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0

            for j in range(110,200):
                if np.sum(label1[1,i:i+5,j-4:j+5])>25:
                    label[1,i,j]=1
                    label[0,i,j]=0
                    label[2,i,j]=0
                else:
                    label[0,i,j]=1
                    label[1,i,j]=0
                    label[2,i,j]=0
        #right-edge
        label1=label.copy()
        for i in range(716,720):
            if np.isnan(Z1[716:720,:]).sum() < 1200:            
                for j in range(10,110):
                    if np.sum(label1[1, i-4:i, j-4:j+5]) > 18:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0
            else:
                for j in range(10,110):
                    if np.sum(label1[1,i-4:i,j-10:j+11])>60:
                        label[1,i,j]=1
                        label[0,i,j]=0
                        label[2,i,j]=0
                    else:
                        label[0,i,j]=1
                        label[1,i,j]=0
                        label[2,i,j]=0

            for j in range(110,200):
                if np.sum(label1[1,i-4:i,j-4:j+5])>25:
                    label[1,i,j]=1
                    label[0,i,j]=0
                    label[2,i,j]=0
                else:
                    label[0,i,j]=1
                    label[1,i,j]=0
                    label[2,i,j]=0
        #corners
        if label[0,0:8,0:20].sum()>80:
            label[0,0:4,0:10]=1
            label[1,0:4,0:10]=0
            label[2,0:4,0:10]=0
        else:
            label[0,0:4,0:10]=0
            label[1,0:4,0:10]=1
            label[2,0:4,0:10]=0
        
        if label[0,-8:,-20:].sum()>80:
            label[0,-4:,-10:]=1
            label[1,-4:,-10:]=0
            label[2,-4:,-10:]=0
        else:
            label[0,-4:,-10:]=0
            label[1,-4:,-10:]=1
            label[2,-4:,-10:]=0
            
        if label[0,0:8,-20:].sum()>80:
            label[0,0:4,-10:]=1
            label[1,0:4,-10:]=0
            label[2,0:4,-10:]=0
        else:
            label[0,0:4,-10:]=0
            label[1,0:4,-10:]=1
            label[2,0:4,-10:]=0
            
        if label[0,-8:,:20].sum()>80:
            label[0,-4:,:10]=1
            label[1,-4:,:10]=0
            label[2,-4:,:10]=0
        else:
            label[0,-4:,:10]=0
            label[1,-4:,:10]=1
            label[2,-4:,:10]=0

        # 腐蚀膨胀
        kernel = np.ones((3,3), np.uint8)
        label[0,:,:]=1 - label[0,:,:]
        # 腐蚀操作
        label[0,:,:] = cv2.erode(label[0,:,:], kernel, iterations=2)
        # 膨胀操作
        label[0,:,:] = cv2.dilate(label[0,:,:], kernel, iterations=1) 
        label[0,:,:] = 1 - label[0,:,:] 
        
        flabel = label[0,:,:].copy()
        for i in range(720):
            for j in range(500):
                if np.isnan(Z1[i,j]):
                    flabel[i,j] = 2
                else:
                    flabel[i,j] = 1 - flabel[i,j]
                    
        np.savez(os.path.join(output_file_dic, "Label_" + k[:-4] + ".npz"), label=flabel)
        
        
if __name__ == "__main__":
    input_file_dic = r'/scratch/zhaoy/Challenge-Cup/datasets/THIdata'
    output_file_dic = r'/scratch/zhaoy/Challenge-Cup/datasets/output/THI_label'
    Threshold(input_file_dic=input_file_dic, output_file_dic=output_file_dic)