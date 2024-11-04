import os
import numpy as np
import torch
import argparse

from PPI_label import Threshold
from KAN import KAN_inference

def SelectMethods(meid, input_dic, output_dic, device):
    if meid == 1:
        Threshold(input_file_dic=input_dic, output_file_dic=output_dic)
    elif meid == 2:
        KAN_inference(input_file_dic=input_dic, output_file_dic=output_dic, device=device)
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=r"D:/TestData/PPI")
    parser.add_argument("--output", type=str, default=r"D:/ExamResult/014")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--Method_selected", type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    Method_selected = args.Method_selected
    
    print(device)
    # print(Method_selected)
    SelectMethods(Method_selected, args.input, args.output, device)