from utils import detector_utils as detector_utils
from utils import predictor_utils as predictor_utils

from utils.hiragana import hiragana
from utils.kanji import kanji
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from scipy.signal import savgol_filter
from PIL import ImageFont, ImageDraw, Image
import numpy


detection_graph, sess = detector_utils.load_inference_graph()
hiragana_model= predictor_utils.load_hiragana_model()
kanji_model= predictor_utils.load_kanji_model()
katakana_model=None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.8, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=640, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')

    arrayDrawed = []

    modifiedPoints = []
    drawedPoints = []
    lineCounts = -1
    isStart = False

    args = parser.parse_args()
    print(args.video_source)
    cap = cv2.VideoCapture(0)
    codec = 0x47504A4D  # MJPG
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    
    # kalman ##########################################################
    kalman = cv2.KalmanFilter(4, 2)
    """
        - dynamParams: This parameter states the dimensionality of the state
        - MeasureParams: This parameter states the dimensionality of the measurement
        - ControlParams: This parameter states the dimensionality of the control
        - vector.type: This parameter states the type of the created matrices that should be CV_32F or CV_64F
    """
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.003 # lưu ý
    # kalman.processNoiseCov = np.array([[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3]],np.float32) * 0.0003 # lưu ý

    check = False # to check hội tụ
    countPassedPoint = 0 # làm thời gian chờ để hiện điểm

    isFirstPoint = False
    countListPoints = -1
    isBacked = True
    #################################################################
    l = 2020 # left, right, top, bottom to crop image and predict
    r = 0
    t = 2020
    b = 0

    #################################################################
    pixelDraw = 5
    pre_predict = -1

    is_start_time_back = -1 # tính giờ cho back nếu lớn hơn 1,5s xóa hết
    is_start_time_write = -1
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        if not ret:
            continue
        image_np = cv2.flip(image_np, 1)
        
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # actual detection
        boxes, scores, classes = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        # DRAWWWW
        (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                          boxes[0][0] * im_height, boxes[0][2] * im_height)
        
        cv2.putText(image_np, str(scores[0]), (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,222))

        if (classes[0] == 5 or classes[0] == 6 or classes[0] == 7) and scores[0] > args.score_thresh:
            is_start_time_back = -1
            ## kalman #####################################################
            p = np.array([np.float32(left+(right-left)/8), np.float32(top+(bottom-top)/8)])
            ptemp = np.array([np.float32(left+(right-left)/8), np.float32(top+(bottom-top)/8)]) # thằng này không đổi để chờ hội tụ
            coor = (int(left+(right-left)/8),int(top+(top-bottom)/8))
            isBacked = True
            kalman.correct(p)
            p = kalman.predict()
            # print(p)
            # if(classes[0] == 7 and (pre_predict == 5 or pre_predict == 6)): # kiểu thứ 3 đầu ngón qua bên phải
            #     p = np.array([np.float32(left+(right-left)/8*7), np.float32(top+(bottom-top)/8)])
            #     ptemp = np.array([np.float32(left+(right-left)/8*7), np.float32(top+(bottom-top)/8)]) # thằng này không đổi để chờ hội tụ
            #     coor = (int(left+(right-left)/8*7),int(top+(top-bottom)/8))
            #     while(abs(int(p[0])-coor[0]) > 0.1 and abs(int(p[1])-coor[1]) > 0.1):
            #         arrayDrawed.append((int(p[0]),int(p[1])))
            #         kalman.correct(ptemp)
            #         p = kalman.predict()
            while(abs(int(p[0])-coor[0]) > 0.1 and abs(int(p[1])-coor[1]) > 0.1 and check == False):
                kalman.correct(ptemp)
                p = kalman.predict()

            check = True
            cv2.line(image_np, (int(p[0]),int(p[1])), (int(p[0]),int(p[1])), (255, 255, 0), 30)
            # if(is_start_time_write == -1):
                # is_start_time_write = datetime.datetime.now()
            # else:
            #     if((datetime.datetime.now()-is_start_time_write).total_seconds()>1.5):
            arrayDrawed.append((int(p[0]),int(p[1])))


        elif classes[0] == 4 and scores[0] > args.score_thresh:
            is_start_time_back = -1
            is_start_time_write = -1
            check = False
            isBacked = True
            isFirstPoint = False
            countPassedPoint = 0

            if(len(arrayDrawed) > 0): # có thì mới add được
                modifiedPoints.append(arrayDrawed)
            arrayDrawed = []

        elif ((classes[0] == 3 or classes[0] == 2)  and scores[0] > args.score_thresh): # check and stop

            is_start_time_write = -1
            is_start_time_back = -1
            check = False
            isBacked = True
            # print(l,r,t,b)
            if(len(arrayDrawed) > 0): # có thì mới add được
                modifiedPoints.append(arrayDrawed)
            for modifiedPoint in modifiedPoints:
                for i in range(1,len(modifiedPoint)):
                    if(l > modifiedPoint[i-1][0]):
                        l = modifiedPoint[i-1][0]
                    if(r < modifiedPoint[i-1][0]):
                        r = modifiedPoint[i-1][0]
                    if(t > modifiedPoint[i-1][1]):
                        t = modifiedPoint[i-1][1]
                    if(b < modifiedPoint[i-1][1]):
                        b = modifiedPoint[i-1][1]
                    # cv2.line(image_np, modifiedPoint[i], modifiedPoint[i-1], (0, 255, 0), 15)
                if(l > modifiedPoint[len(modifiedPoint)-1][0]):
                    l = modifiedPoint[len(modifiedPoint)-1][0]
                if(r < modifiedPoint[len(modifiedPoint)-1][0]):
                    r = modifiedPoint[len(modifiedPoint)-1][0]
                if(t > modifiedPoint[len(modifiedPoint)-1][1]):
                    t = modifiedPoint[len(modifiedPoint)-1][1]
                if(b < modifiedPoint[len(modifiedPoint)-1][1]):
                    b = modifiedPoint[len(modifiedPoint)-1][1]
            if(r-l > 0 and b-t > 0):
                img = numpy.zeros([b-t+30, r-l+30, 3])
                
                for modifiedPoint in modifiedPoints:
                    for i in range(1,len(modifiedPoint)):
                        cv2.line(img, (modifiedPoint[i][0]-l+15,modifiedPoint[i][1]-t+15), (modifiedPoint[i-1][0]-l+15,modifiedPoint[i-1][1]-t+15), (255,255,255), 15)
                        # print((modifiedPoint[i][0]-l,modifiedPoint[i][1]-t))
                cv2.imwrite('img.jpg', img)
                # print(hiragana[predictor_utils.predict_all(img, hiragana_model)])
                print(kanji[predictor_utils.predict_all(img, kanji_model=kanji_model)])

            l = 2020 # left, right, top, bottom to crop image and predict
            r = 0
            t = 2020
            b = 0

            # cv2.imshow('st2',image_np[t:b,l:r])
            # print(hiragana[predictor_utils.predict_all()])
            arrayDrawed = []
            modifiedPoints = []
            # isStart = False
            # check = False
        elif classes[0] == 1  and scores[0] > args.score_thresh:
            is_start_time_write = -1
            check = False
            if(is_start_time_back == -1): # nếu chưa back trước lần nào thì gán
                is_start_time_back = datetime.datetime.now()
                if(isBacked == True):
                    if(len(arrayDrawed) > 0): # có thì mới add được # kiểm tra xem đã vẽ gì chưa để add vào trước khi xóa
                        modifiedPoints.append(arrayDrawed)
                        arrayDrawed = [] # pop rồi nhưng thằng này vẫn vẽ ??:D??
                    if(len(modifiedPoints) > 0): # if empty không cần pop
                        modifiedPoints.pop(-1)
                    isBacked = False
            else:
                if((datetime.datetime.now()-is_start_time_back).total_seconds()>1.5):
                    arrayDrawed = []
                    modifiedPoints = []

        pre_predict = classes[0]

        # print(modifiedPoints)
        
        for modifiedPoint in modifiedPoints:
            for i in range(1,len(modifiedPoint)):
                cv2.line(image_np, modifiedPoint[i], modifiedPoint[i-1], (0, 255, 0), pixelDraw)

        for i in range(1,len(arrayDrawed)):
            cv2.line(image_np, arrayDrawed[i], arrayDrawed[i-1], (0, 255, 0), pixelDraw) 
 

        detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, classes, scores, boxes, im_width, im_height, image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), image_np)

            cv2.imshow('Single-Threaded Detection', cv2.cvtColor(
                image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
