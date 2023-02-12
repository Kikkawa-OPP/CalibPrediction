import numpy as np
import cv2
import math
import json
import random


class Camera():
    """
    Default image size: 1920x1080
    return pinhole model if distortion == False
    """
    def __init__(self, imsize=[1920,1080], distortion = True):
        self.imsize = imsize
        magnification = self.imsize[0]/1920
        if distortion:
            self.Mtx = np.array(
                [
                    [1000*magnification, 0, 1020*magnification],
                    [0, 1010*magnification, 560*magnification],
                    [0,0,1]
                ], np.float32
            )
            self.DistoCo = np.array(
                [
                    [-0.3], [0.1], [0.02], [0.01], [0]
                ], np.float32
            )
        else:
            self.Mtx = np.array(
                [
                    [1000*magnification, 0, 960*magnification],
                    [0, 1000*magnification, 540*magnification],
                    [0,0,1]
                ], np.float32
            )

            self.DistoCo = np.array(
                [
                    [0],[0],[0],[0],[0]
                ], np.float32
            )

def dict2json(filename, data):
    class Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(Encoder, self).default(obj)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent = 4, cls = Encoder)

def shoot(point, K, D, R = np.zeros((3,1),np.float32), t = np.zeros((3,1),np.float32)):
    """
    pt: 3D horizonal vector It's allowed ndarray and non-ndarray
    return non-ndarray, 2D horizonal vector
    """

    pt = np.array(point, np.float32)
    u, v = cv2.projectPoints(pt, R, t, K, D)[0][0][0]
    return [round(u), round(v)]

def in_fov2d(point, p, q):
    """
    pt: 2D horizonal vector
    It's allowed ndarray and non-ndarray
    p:left-upper edge, q:right-lower edge of the boundingbox
    """
    u, v = point
    minU, minV = p
    maxU, maxV = q

    return (u >= minU) and (u < maxU) and (v >= minV) and (v < maxV)

def noise2(ImgPts, cam, STD=1.0):
    """
    input: horizonal 2Dvectors (ndarray)
    output: horizonal 2Dvectors (ndarray)
    Output always comes into view of the camera(input)
    """
    imsize = cam.imsize
    pts = ImgPts.copy()

    def noiseVector(std):
        r = random.normalvariate(0,std)
        a = random.uniform(0,2*np.pi)
        d = np.array(
            [
                round(r*math.cos(a)),
                round(r*math.sin(a))
            ], np.float32
        )
        return d
        
    if STD < 0:
        return None
    elif STD == 0:
        return pts
    else:
        for i in range(len(pts)):
            countLimit = 0
            while countLimit < 1000:
                pt = np.round(pts[i]+noiseVector(STD))
                if in_fov2d(pt, [0,0], imsize):
                    pts[i] = pt
                    break
                else:
                    countLimit += 1
            if countLimit == 1000:
                return None
        
        return pts
             
def noise3(ObjPts, cam, STD=0.015, R=np.zeros((3,1)), t=np.zeros((3,1))):
    """
    input: horizonal 3D vectors (ndarray)
    output: horizonal 3D vectors (ndarray)
    Output always comes into view of the camera(input)
    """
    Pts = ObjPts.copy()
    imsize = cam.imsize
    K = cam.Mtx
    D = cam.DistoCo

    def noiseVector(std):
        r = random.normalvariate(0,std)
        a = random.uniform(0,2*np.pi)
        b = random.uniform(0,2*np.pi)
        d = np.array(
            [
                r*math.sin(a)*math.cos(b),
                r*math.sin(a)*math.sin(b),
                r*math.cos(a)
            ], np.float32
        )
        return d

    if STD < 0:
        return None
    elif STD == 0:
        return Pts
    else:
        for i in range(len(Pts)):
            countLimit = 0
            while countLimit < 1000:
                Pt = Pts[i] + noiseVector(STD)
                if in_fov2d(shoot(Pt, K, D, R, t),[0,0],imsize):
                    Pts[i] = Pt
                    break
                else:
                    countLimit += 1

        return Pts

def EXsynth():
    """
    Set extrinsic parameters randomly (for both the camera & the scene) 
    return R0:rotation vector, t0:translation vector
    """
    rot_amount = random.uniform(0,2*math.pi)
    rot_phi1 = random.uniform(0,2*math.pi)
    rot_phi2 = random.uniform(0,2*math.pi)
    Z = rot_amount*math.sin(rot_phi1)
    R0 = np.array(
        [
            [Z*math.cos(rot_phi2)], [Z*math.sin(rot_phi2)], [Z]
        ], np.float32
    )

    trans_amount = random.uniform(0,100)
    trans_phi1 = random.uniform(0,2*math.pi)
    trans_phi2 = random.uniform(0,2*math.pi)
    Z = trans_amount*math.sin(trans_phi1)
    t0 = np.array(
        [
            [Z*math.cos(trans_phi2)], [Z*math.sin(trans_phi2)], [Z]
        ], np.float32
    )

    return R0, t0

def corrInit(ObjPts, ImgPts, cam, n2d=1.0, n3d=0.015):
    imsize = cam.imsize
    K = cam.Mtx
    D = cam.DistoCo

    #add noise to image points
    Img = noise2(ImgPts, cam, n2d)

    #add noise to object points
    Obj = noise3(ObjPts, cam, n3d)

    #decide true pose of the synthetic environment
    R0, t0 = EXsynth()
    Rmat = cv2.Rodrigues(R0)[0]
    for i in range(len(Obj)):
        Pt = Obj[i].reshape((3,1))
        Pt = (np.dot(Rmat.T, Pt-t0)).reshape((3,))
        Obj[i] = Pt
    
    return Obj, Img, R0, t0

def singleCalibrator(imsize, ObjPts, ImgPts, axis1 = [math.radians(i*5) for i in range(1,18)], axis2 = [i/(20) for i in range(-10,10)]):
    """
    Calibrate correspondence from a single image
    2-Axis grid search (1:vertical FoV, 2:the 1st term of distortion coefficient)
    Uses rational distortion model
    imsize = (width, height)
    ObjPts/ImgPts = ndarray, list of horizonal vector
    """
    cx = imsize[0]/2
    cy = imsize[1]/2
    
    ObjPts = [ObjPts]
    ImgPts = [ImgPts]
   
    Ret = None

    for fovV in axis1:
        for k1 in axis2:
            fy = cy/(math.tan(fovV))
            fx = fy

            initialMat = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
            initialDist = np.array([[k1],[0],[0],[0],[0],[0],[0],[0]],np.float32)
            
            try:
                ret1, camera_matrix1, distortion1, rvecs1, tvecs1 = cv2.calibrateCamera(
                    ObjPts, 
                    ImgPts, 
                    imsize, 
                    cameraMatrix = initialMat, 
                    distCoeffs = initialDist, 
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH
                )

                ret2, camera_matrix2, distortion2, rvecs2, tvecs2 = cv2.calibrateCamera(
                    ObjPts,
                    ImgPts, 
                    imsize, 
                    cameraMatrix = camera_matrix1, 
                    distCoeffs = distortion1, 
                    rvecs = rvecs1,
                    tvecs = tvecs1,
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_EXTRINSIC_GUESS
                )
            except:
                continue

            if (camera_matrix2[0][0] < 0) or (camera_matrix2[1][1] < 0):
                continue

            if (Ret is None) or ret2 < Ret:
                Ret = ret2
                Mtx = camera_matrix2
                DistoCo = distortion2
                Rvec = rvecs2[0]
                Tvec = tvecs2[0]

    if Ret is None:
        Mtx, DistoCo, Rvec, Tvec = None, None, None, None

    return Ret, Mtx, DistoCo, Rvec, Tvec

def multiCalibrator(imsize, ObjPts, ImgPts, axis1 = [math.radians(i*5) for i in range(1,18)], axis2 = [i/(20) for i in range(-10,10)]):
    """
    Calibrate by correspondence from multiple images
    2-Axis grit search
    Uses rational distortion model
    imsize = (width, height)
    ObjPts/ImgPts = ndarray, list of horizonal vector
    """
    cx = imsize[0]/2
    cy = imsize[1]/2   
 
    Ret = None

    for fovV in axis1:
        for k1 in axis2:
            fy = cy/(math.tan(fovV))
            fx = fy

            initialMat = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
            initialDist = np.array([[k1],[0],[0],[0],[0],[0],[0],[0]],np.float32)
            
            try:
                ret1, camera_matrix1, distortion1, rvecs1, tvecs1 = cv2.calibrateCamera(
                    ObjPts, 
                    ImgPts, 
                    imsize, 
                    cameraMatrix = initialMat, 
                    distCoeffs = initialDist, 
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_EXTRINSIC_GUESS
                )

                ret2, camera_matrix2, distortion2, rvecs2, tvecs2 = cv2.calibrateCamera(
                    ObjPts,
                    ImgPts, 
                    imsize, 
                    cameraMatrix = camera_matrix1, 
                    distCoeffs = distortion1, 
                    rvecs = rvecs1,
                    tvecs = tvecs1,
                    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_EXTRINSIC_GUESS
                )
            except:
                continue

            if (Ret is None) or ret2 < Ret:
                Ret = ret2
                Mtx = camera_matrix2
                DistoCo = distortion2
                Rvec = rvecs2
                Tvec = tvecs2
    
    if Ret is None:
        Mtx, DistoCo, Rvec, Tvec = None, None, None, None

    return Ret, Mtx, DistoCo, Rvec, Tvec

def opticalray(point, K, D):
    """
    pt: 2D horizonal vector
    K, D: Intrinsic parameters
    output: 3d horizonal vector (ndarray):Position of the pt on undistorted image plane (depth = 1)
    """
    P = np.eye(3)
    R = np.eye(3)
    pt = cv2.undistortPointsIter(np.array(point, np.float32), K, D, R, P, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, np.finfo(np.float32).eps))
    return np.array(pt[0][0].tolist()+[1], np.float32)

def entireRE(cam, R0, t0, p, q, DepthCenter, K, D, R, t):
    W, H = cam.imsize
    R0mat = cv2.Rodrigues(R0)[0]

    squares_entire=[]
    squares_view=[]
    for u in range(0,W,10):
        for v in range(0,H,10):
            pt = np.array([u,v], np.float32)
            point_on_image_plane = DepthCenter*opticalray(pt, cam.Mtx, cam.DistoCo).reshape((3,1))
            back_projected = np.dot(R0mat.T, point_on_image_plane-t0)
            back_projected = back_projected.reshape(1,3).astype(np.float32)
            reprojected = cv2.projectPoints(back_projected, R, t, K, D)[0][0][0]
            square = np.linalg.norm(reprojected-pt)**2
            squares_entire.append(square)
            if in_fov2d(pt, p, q):
                squares_view.append(square)
    
    RE_entire = np.sqrt(np.mean(squares_entire))
    RE_view = np.sqrt(np.mean(squares_view))

    return RE_entire, RE_view

def EOC(R0, t0, R, t):
    """
    R0, t0:True extrinsic parameters
    R, t:Estimated extrinsic parameters
    return errors regarding to the optical center
    ...position: positional error [m]
    ...direction: positional error [deg]
    ...orientation: optical axis error [deg]
    """
    Rmat0 = cv2.Rodrigues(R0)[0]
    Rmat = cv2.Rodrigues(R)[0]

    OC0 = -np.dot(Rmat0.T, t0)
    OC = -np.dot(Rmat.T, t)

    O = 100*(np.dot(Rmat0, OC)+t0)

    z = np.array([[0],[0],[100]], np.float32)
    OR0 = np.dot(Rmat0.T, z-t0)-OC0
    OR = np.dot(Rmat.T, z-t)-OC

    position = np.linalg.norm(OC-OC0)

    bottom = np.linalg.norm(OR0)*np.linalg.norm(OR)
    top = np.dot(OR0.T, OR)

    if abs(top) < abs(bottom):
        orientation = math.degrees(abs(math.acos(top/bottom)))
    else:
        orientation = None

    return position, orientation

def synthFromImgPt(pt, K, D, depth):
    """
    pt: 2D horizonal vector
    Synthesize object point from image one
    output: ndarray, 3D horizonal vector (camera corr)
    """
    OR = opticalray(pt, K, D)
    return (depth/np.linalg.norm(OR))*OR

def randomSubView(ratio, imsize):
    """
    Generate boundingbox rondomly from the size
    ratio: float, 0.1~1.0
    Output p,q: p->UpperLeft q->LowerRight point of the view
    """
    W, H = imsize
    S = W*H

    if ratio < 1.0:
        s = S*ratio
        
        h = random.randint(math.ceil(s/W), H)
        w = math.floor(s/h)
            
        p1 = random.randint(0,W-w)
        p2 = random.randint(0,H-h)

        q1 = p1 + w - 1
        q2 = p2 + h - 1
    else:
        (p1, p2) = (0,0)
        (q1, q2) = (W-1, H-1)

    return (p1, p2), (q1, q2)

def UrbanLike(N, cam, p = None, q = None, Occupancy = 0.8, DepthCenter = 20, DepthRange = 0.6, Width = 10, Height = 5, camHeight = 1.5, wallSTD = 0.5, groundSTD = 0.05):
    """
    Synthesize correspondence on the urban-like scene
    The bounding area on image is discripted by p & q (2 diagonal corners)
    ...If only BBsize (Occupancy) is set, the bounding area will be set randomly
    ...If both BBsize & bounding area are set, BBsize will be ignored

    ...return None if miss synthesized (correspondence distribution area may be too small)
    return:
        ObjPts, ImgPts: correspondence
        p,q: actual bounding box (the edges)
        DepthCenter, DepthRange: actual representative depths
    """
    imsize = cam.imsize
    K = cam.Mtx
    D = cam.DistoCo

    depthMax = DepthCenter*(1 + DepthRange)
    depthMin = max(0.01, DepthCenter*(1 - DepthRange))

    ObjPts = []
    ImgPts = []

    #Set bounding area on the image if it is not set.
    if p is None:
        p, q = randomSubView(Occupancy, imsize)
        Umin, Vmin = p
        Umax, Vmax = q
    else:
        Umin, Vmin = p
        Umax, Vmax = q

    counter = 0
    while len(ImgPts) < N:
        if counter > 1000*N:
            return None

        width = random.normalvariate(Width, wallSTD)
        groundHeight = random.normalvariate(camHeight, groundSTD)
        u = random.randint(Umin, Umax)
        v = random.randint(Vmin, Vmax)
        pt = [u,v]
        OR = opticalray(pt, K, D)
        x, y, z = OR
        if pt in ImgPts or abs(x) < 0.001 or abs(y) < 0.001:
            counter += 1
            continue

        OR = opticalray(pt, K, D)
        x, y, z = OR
        normOR = np.linalg.norm(OR)
        mag_to_wall = abs((width/2)/x)
        mag_to_ground = abs(groundHeight/y)

        if mag_to_wall*y > -Height + groundHeight:
            if mag_to_wall*y < groundHeight:
                if depthMin < mag_to_wall*normOR < depthMax:
                    ImgPts.append(pt)
                    ObjPts.append(mag_to_wall*OR)
                    continue
            elif depthMin < mag_to_ground*normOR < depthMax:
                ImgPts.append(pt)
                ObjPts.append(mag_to_ground*OR)
                continue

        counter += 1

    ObjPts = np.array(ObjPts, np.float32)
    ImgPts = np.array(ImgPts, np.float32)

    return ObjPts, ImgPts

def Random3D(N, cam, p = None, q = None, Occupancy = 0.8, DepthCenter = 20, DepthRange = 0.6):
    """
    Synthesize correspondence on the random3D scene
    The bounding area on image is discripted by p & q (2 diagonal corners)
    ...If only BBsize (Occupancy) is set, the bounding area will be set randomly
    ...If both BBsize & bounding area are set, BBsize will be ignored

    return 2 vals:ObjPts, ImgPts
    ...return None, None if miss synthesized (correspondence distribution area may be too small)
    """
    maxDepth = DepthCenter*(1 + DepthRange)
    minDepth = max(0.5, DepthCenter*(1 - DepthRange))

    imsize = cam.imsize

    if p is None:
        p, q = randomSubView(Occupancy, imsize)
        Umin, Vmin = p
        Umax, Vmax = q
    else:
        Umin, Vmin = p
        Umax, Vmax = q

    Img = []
    Obj = []
    counter = 0
    while len(Obj) < N:
        counter += 1
        if counter == 10000*N:
            return None, None
        pt = [random.randint(Umin, Umax), random.randint(Vmin, Vmax)]
        if pt in Img:
            continue
        else:
            Img.append(pt)
            Depth = random.uniform(minDepth, maxDepth)
            Obj.append(synthFromImgPt(pt, cam.Mtx, cam.DistoCo, Depth))

    ObjPts = np.array(ObjPts, np.float32)
    ImgPts = np.array(ImgPts, np.float32)
    
    return ObjPts, ImgPts

def Intersection(N, cam, p = None, q = None, Occupancy = 0.8, DepthCenter = 20, DepthRange = 0.6, Width = 10, height = 5, camHeight = 1.5, wallSTD = 0.5, groundSTD = 0.05, StopLine_to_Cross = 10):
    """
    Synthesize correspondence on the urban-like scene whitch on crossroad.
    The bounding area on image is discripted by p & q (2 diagonal corners)
    ...If only BBsize (Occupancy) is set, the bounding area will be set randomly
    ...If both BBsize & bounding area are set, BBsize will be ignored

    return 2 vals:ObjPts, ImgPts
    ...return None, None if miss synthesized (correspondence distribution area may be too small)
    """
    imsize = cam.imsize
    K = cam.Mtx
    D = cam.DistoCo

    depthMax = DepthCenter*(1 + DepthRange)
    depthMin = max(0.5, DepthCenter*(1 - DepthRange))

    ObjPts = []
    ImgPts = []

    #Set bounding area on the image if it is not set.
    if p is None:
        p, q = randomSubView(Occupancy, imsize)
        Umin, Vmin = p
        Umax, Vmax = q
    else:
        Umin, Vmin = p
        Umax, Vmax = q

    counter = 0
    while len(ImgPts) < N:
        if counter > 1000*N:
            return None, None

        width = random.normalvariate(Width, wallSTD)
        crossDistance = StopLine_to_Cross+width
        groundHeight = random.normalvariate(camHeight, groundSTD)
        u = random.randint(Umin, Umax)
        v = random.randint(Vmin, Vmax)
        pt = [u,v]
        OR = opticalray(pt, K, D)
        x, y, z = OR
        if pt in ImgPts or abs(x) < 0.001 or abs(y) < 0.001:
            counter += 1
            continue

        OR = opticalray(pt, K, D)
        x, y, z = OR
        mag_to_wall = abs((width/2)/x)
        mag_to_ground = abs(groundHeight/y)

        if StopLine_to_Cross < mag_to_wall < crossDistance:
            if crossDistance*y >= -height+groundHeight:
                if crossDistance*y <= groundHeight:
                    if depthMin <= crossDistance*np.linalg.norm(OR) <= depthMax:
                        ImgPts.append(pt)
                        ObjPts.append(crossDistance*OR)
                        continue
                elif depthMin <= mag_to_ground*np.linalg.norm(OR) <= depthMax:
                    ImgPts.append(pt)
                    ObjPts.append(mag_to_ground*OR)
                    continue
        elif mag_to_wall*y >= -height + groundHeight:
            if mag_to_wall*y <= groundHeight:
                if depthMin <= mag_to_wall*np.linalg.norm(OR) <= depthMax:
                    ImgPts.append(pt)
                    ObjPts.append(mag_to_wall*OR)
                    continue
            elif depthMin <= mag_to_ground*np.linalg.norm(OR) <= depthMax:
                ImgPts.append(pt)
                ObjPts.append(mag_to_ground*OR)
                continue

        counter += 1

    ObjPts = np.array(ObjPts, np.float32)
    ImgPts = np.array(ImgPts, np.float32)
    
    return ObjPts, ImgPts

def calibrationTrial(N, cam, synthesizer=UrbanLike, view=None, DepthCenter=20, DepthRange=0.6, Width=10, Height=5):
    if view is not None:
        p, q = view
    else:
        p = [0,0]
        q = cam.imsize
    if synthesizer == Random3D:
        ObjPts, ImgPts = synthesizer(N, cam, p=p, q=q)
    else:
        ObjPts, ImgPts = synthesizer(N, cam, p=p, q=q, DepthCenter=DepthCenter, DepthRange=DepthRange, Width=Width, Height=Height)
    
    ObjPts, ImgPts, R0, t0 = corrInit(ObjPts, ImgPts, cam)
    RE_corr, K, D, R, t = singleCalibrator(cam.imsize, ObjPts, ImgPts)
    RE_entire, RE_view = entireRE(cam, R0, t0, p, q, DepthCenter, K, D, R, t)
    Epos, Eori = EOC(R0, t0, R, t)

    return RE_corr, RE_entire, RE_view, Epos, Eori
