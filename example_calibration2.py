import numpy as np
import CalibPredictionModules as PM
import csv


"""
Read correspondence from dash-cam A
"""
imsize = [1280, 720]
ObjPts = []
ImgPts = []
for scene in ['1-1_A', '1-2_A', '2-1_A', '2-2_A']:
    with open('Corrs/1-1_A.csv', 'r') as Data:
        Corr = [row for row in csv.reader(Data) if row]
    ObjPts.append(np.array([row[:3] for row in Corr], np.float32))
    ImgPts.append(np.array([row[3:] for row in Corr], np.float32))


"""
Calibration
"""
ret, Mtx, DistoCo, rvec, tvec = PM.multiCalibrator(imsize, ObjPts, ImgPts)

print(ret)