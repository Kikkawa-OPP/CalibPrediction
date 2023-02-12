import numpy as np
import CalibPredictionModules as PM
import csv


"""
Read correspondence of scene 1-1:A
"""
imsize = [1280, 720]
with open('Corrs/1-1_A.csv', 'r') as Data:
    Corr = [row for row in csv.reader(Data) if row]
ObjPts = np.array([row[:3] for row in Corr], np.float32)
ImgPts = np.array([row[3:] for row in Corr], np.float32)

"""
Calibration
"""
ret, Mtx, DistoCo, rvec, tvec = PM.singleCalibrator(imsize, ObjPts, ImgPts)

print(ret)