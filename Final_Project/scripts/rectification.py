import numpy as np
import cv2
import os 



#Loading Paths of Files 
calibFile = "Final_Project/models/stereo_calibration.npz"
LEFT_IMAGE = "Final_Project/data/34759_final_project_raw/calib/image_02/data/000000.png"
RIGHT_IMAGE = "Final_Project/data/34759_final_project_raw/calib/image_03/data/000000.png"
OUTPUT_DIR = "Final_Project/models"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rectification_comparison.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)




# Loading Calibration 
data = np.load(calibFile)
mtxL, distL = data['mtxL'], data['distL']
mtxR, distR = data['mtxR'], data['distR']
R, T = data['R'], data['T']
R1, R2, P1, P2, Q = data['R1'], data['R2'], data['P1'], data['P2'], data['Q']



#Loading Stero Images
def _load_image_with_fallback(path):
    img = cv2.imread(path)
    if img is None:
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Image path '{path}' not found and directory '{directory}' does not exist.")
        candidates = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not candidates:
            raise FileNotFoundError(f"Couldn't read '{path}' and no image files found in directory '{directory}'.")
        candidates.sort()
        fallback = os.path.join(directory, candidates[0])
        print(f"Warning: couldn't read '{path}'. Using fallback '{fallback}'.")
        img = cv2.imread(fallback)
        if img is None:
            raise IOError(f"Failed to read fallback image '{fallback}'.")
    return img

leftImg = _load_image_with_fallback(LEFT_IMAGE)
rightImg = _load_image_with_fallback(RIGHT_IMAGE)
h, w = leftImg.shape[:2]


#Generating Rectification Maps
map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)

#Rectify Images 
rectL = cv2.remap(leftImg,map1x,map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(rightImg,map2x,map2y,cv2.INTER_LINEAR)


#Draw Epipolar Lines 
step = 50
for y in range (0,h , step):
    cv2.line(rectL, (0,y),(w,y),(0,255,0),1)
    cv2.line(rectR, (0,y),(w,y),(0,255,0),1)





# Drawing Chess Board Corner Opt
chessboard_size = (7, 5)
grayL = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

if retL:
    for pt in cornersL.reshape(-1, 2):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(rectL, (x, y), 4, (0, 0, 255), -1)

if retR:
    for pt in cornersR.reshape(-1, 2):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(rectR, (x, y), 4, (0, 0, 255), -1)

comparison = np.hstack((rectL, rectR))
cv2.imwrite(OUTPUT_FILE, comparison)

print("Rectification comparison saved to:", OUTPUT_FILE)
