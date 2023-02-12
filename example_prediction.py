import numpy as np
import multiprocessing as mp
import CalibPredictionModules as PM

"""
Inputs for precidtion
(example for 1-1:A in the aricle)
"""
imsize = [1280, 720] #Image size
N = 22 #Number of the correspondence
p = [56,21] #Upper left edge of the bounding box containing image points
q = [1189,482] #Lower right edge of the bounding box containing image points
DepthCenter = 15 #Representative depth
DepthRange = 0.8 #Depth range
Width = 10 #Road width
Height = 3 #wall Height

trials = 1000 #Number of trials for prediction

"""
prediction code
"""
cam = PM.Camera(imsize)
view = [p,q]
def trial(void):
    ret = PM.calibrationTrial(
        N,
        cam,
        synthesizer=PM.UrbanLike, #Geometry template: UrbanLike / Random3D / Intersection
        view=view,
        DepthCenter=DepthCenter,
        DepthRange=DepthRange,
        Width=Width,
        Height=Height
    )
    return ret

if __name__ == '__main__':
    cpu_margin = 1
    process = mp.Pool(mp.cpu_count()-cpu_margin)
    results = process.map(trial,[None for i in range(trials)])
    del process
    
    #Summarize results
    measures = ['RE_corr', 'RE_entire', 'RE_view', 'Epos', 'Eori']
    lists = {measure:[] for measure in measures}
    for result in results:
        if result is not None:
            for i in range(len(result)):
                lists[measures[i]].append(result[i])
    
    predicted = {measure:None for measure in measures}
    for key in measures:
        predicted[key] = np.percentile(lists[key], [50,95])
    
    PM.dict2json('example_predicted', predicted)
    



