import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import imutils
import time


class Helper:
    def __init__(self):
        self.numOfFrames = 480

    def getframelist(self, path):
        framelist = os.listdir(path)
        framelist.sort(key=lambda x: int(x[5:-4]))
        validframes = [path + "\\" + x for x in framelist[:480]]
        return validframes


class MotionVector(object):

    def __init__(self, path):
        self.path = path
        self.width = 360
        self.height = 640
        self.numOfFrames = 50
        self.blockSize = 20
        self.k = 3
        self.frames = np.empty((self.numOfFrames, self.width, self.height))
        self.motionDescriptors = np.empty(self.numOfFrames)

    def getYChannel(self):
        framelist = os.listdir(self.path)
        framelist.sort(key=lambda x: int(x[5:-4]))
        for i in range(self.numOfFrames):
            img = cv2.imread(self.path + "\\" + framelist[i])
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            self.frames[i, :, :], u, v = cv2.split(img_yuv)
        print(self.frames.shape)
        # yprime = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        # result = np.vstack([img, yprime])
        # cv2.imwrite('testimg.jpg', result)

    def getMotionVector(self):
        for frameid in range(1, self.numOfFrames):
            for x in range(0, self.width, self.blockSize):
                for y in range(0, self.height, self.blockSize):
                    mindiff = 256
                    minmn = 1
                    dist = 0
                    for i in range(-self.k, self.k):
                        if x + i < 0 or x + i + self.blockSize < 0 or x + i >= self.width or x + i + self.blockSize >= self.width:
                            continue
                        for j in range(-self.k, self.k):
                            if y + j < 0 or y + j + self.blockSize < 0 or y + j >= self.height or y + j + self.blockSize >= self.height:
                                continue
                            diff = 0
                            mn = 0
                            for p in range(x, x + self.blockSize):
                                for q in range(y, y + self.blockSize):
                                    if 0 <= p + i < self.width and 0 <= q + j < self.height:
                                        mn += 1
                                        diff += abs(self.frames[frameid, p, q] - self.frames[frameid - 1, p + i, q + j])
                            if mn > 0 and mindiff / minmn > diff / mn:
                                minmn = mn
                                mindiff = diff
                                dist = (np.sqrt(i * i + j * j)).astype(int)
                    # print("Frame " + str(frameid) + " position [" + str(x) + ", " + str(y) + "] dist = " + str(dist))
                    self.motionDescriptors[frameid] += dist
            print(str(frameid) + " md = " + str(self.motionDescriptors[frameid]))
        self.motionDescriptors[0] = self.motionDescriptors[1]


class MotionDetector:
    def motiondetect(self, framelist):
        firstFrame = None
        min_area = 500
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (33, 65), 0)
            if index == 0:
                firstFrame = gray
                continue
            frameDelta = cv2.absdiff(firstFrame, gray)
            # fill out holes by threshold to find contours
            thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # eliminate too small area
            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                # calculate the bounding box
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Original frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        cv2.destroyAllWindows()


class ImageSmoothing:
    def smooth(self, framelist):
        sumofdiff = np.empty(480)
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            blur = cv2.bilateralFilter(h, 9, 75, 75)
            sumofdiff[index] = (h - blur).sum()
        return sumofdiff

class FreqDetector:
    def highfreqframe(self, framefile):
        frame = cv2.imread(framefile)
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame1.shape[:2]
        framef32 = np.zeros((h,w), np.float32)
        framef32[:h, :w] = frame1
        dct = cv2.dct(cv2.dct(framef32))
        dct = np.around(dct)
        high = dct[356:360, 636:640]
        print(np.sum(np.abs(high)))

    def highfreq(self, framelist):
        high = np.zeros([480], int)
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame1.shape[:2]
            framef32 = np.zeros((h,w), np.float32)
            framef32[:h, :w] = frame1
            dct = cv2.dct(cv2.dct(framef32))
            dct = np.around(dct)
            high[index] = np.sum(np.abs((dct[356:360, 636:640])))
        return high

class SkinDetector:

    def cr_ostu_frame(self, framefile):
        frame = cv2.imread(framefile)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(skin)
        cv2.namedWindow("skin crotsu", cv2.WINDOW_NORMAL)
        cv2.imshow("skin crotsu", skin)
        cv2.namedWindow("seperate", cv2.WINDOW_NORMAL)
        cv2.imshow("seperate", cv2.bitwise_and(frame, frame, mask=skin))
        cv2.waitKey()

    def cr_ostu(self, framelist):
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            y, cr, cb = cv2.split(ycrcb)
            cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
            _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(skin)
            break

class ColorDistribution:
    def coloraround(self, framelist):
        same = np.zeros([20])
        n = 0
        for index, framefile in enumerate(framelist):
            if index % 10 > 0:
                continue
            k = int(index / 24)
            frame = cv2.imread(framefile)
            frame = imutils.resize(frame, width=160)
            for i in range(89):
                if i == 0:
                    continue
                for j in range(319):
                    if j == 0:
                        continue
                    for p in [-1, 0, 1]:
                        for q in [-1, 0, 1]:
                            if p==q==0:
                                continue
                            if np.sum(cv2.absdiff(frame[i][j], frame[i+p][j+q])) < 3:
                                n+=1
                    if n > 6:
                        same[k]+=1
                    n=0
        return same

class KeyFrameDetector:
    def getkeyfbyhist(self, framelist):
        frame = cv2.imread(framelist[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lastHist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        diff = np.empty([480], dtype=int)
        keyframes = np.zeros([480], dtype=int)
        secondHist = np.array([])
        firsthist = lastHist
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            diff[index] = min(np.sum(cv2.absdiff(hist, lastHist)), np.sum(cv2.absdiff(hist, firsthist)))
            if diff[index] > 18000:
                if secondHist.size == 0:
                    secondHist = hist
                    diff[index] = 0
                else:
                    diff[index] = min(diff[index], np.sum(cv2.absdiff(hist, secondHist)))
                    if diff[index] > 18000:
                        keyframes[index] = 1
                        diff[index] = 0
            lastHist = hist
        # print("find " + str(np.sum(keyframes)) + " key frames")
        return diff, keyframes

    def getkeyfbyhisthue(self, framelist):
        frame = cv2.imread(framelist[0])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        lastHist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        diff = np.empty([480], dtype=int)
        # minordiffframes = np.zeros([480], dtype=int)
        keyframes = np.zeros([480], dtype=int)
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
            diff[index] = np.sum(cv2.absdiff(hist, lastHist))
            if diff[index] > 80000:
                keyframes[index] = 1
            lastHist = hist
        # print("find " + str(np.sum(keyframes)) + " key frames")
        return diff, keyframes

    def getkeyfbypixeldiff(self, framelist):
        lastframe = cv2.imread(framelist[0])
        diff = np.empty([480], dtype=int)
        minordiffframes = np.zeros([480], dtype=int)
        keyframes = np.zeros([480], dtype=int)
        for index, framefile in enumerate(framelist):
            frame = cv2.imread(framefile)
            tempdiff = cv2.absdiff(frame, lastframe)
            diff[index] = np.sum(tempdiff)
            if diff[index] > 2.5e7:
                keyframes[index] = 1
            elif diff[index] > 3e6:
                minordiffframes[index] = 1
            lastframe = frame
        return diff, keyframes, minordiffframes

class Visualizer:
    def plotone(self, y):
        x = np.arange(480)
        plt.plot(x, y)
        plt.show()

    def scattertwo(self, y1, y2):
        x = np.arange(480)
        plt.scatter(x, y1, s=2, c='b', marker='o')
        plt.scatter(x, y2, s=2, c='r', marker='*')
        plt.show()

    def scatter(self, y):
        x = np.arange(480)
        plt.scatter(x, y)
        plt.show()

    def plot(self, y1, y2, y3, y4, y5, y6):
        x = np.arange(480)
        plt.plot(x, y1, 'r', label='l1')
        plt.plot(x, y2, 'm', label='l2')
        plt.plot(x, y3, 'orange', label='l3')
        plt.plot(x, y4, 'g', label='l4')
        plt.plot(x, y5, 'lime', label='l5')
        plt.plot(x, y6, 'y', label='l6')
        plt.xlabel('frame')
        plt.ylabel('sum of diff after smoothing')
        plt.legend()
        plt.show()

def draw_linechart(y, imgname):
    x = np.arange(480)
    plt.plot(x, y)
    plt.savefig(imgname + ".png")

def getframelist(video):
    pathprefix = "D:\\USC\\CS576\\project\\Data_jpg\\"
    i = video.find('_')
    videoclass = video[0:i]
    videopath = pathprefix + videoclass + "\\" + video
    framelist = os.listdir(videopath)
    framelist.sort(key=lambda x: int(x[5:-4]))
    validframes = [path + "\\" + x for x in framelist[:480]]
    return validframes

def calculate_dct(framelist, videoname):
    high = np.zeros([480], int)
    for index, framefile in enumerate(framelist):
        frame = cv2.imread(framefile)
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame1.shape[:2]
        framef32 = np.zeros((h,w), np.float32)
        framef32[:h, :w] = frame1
        dct = cv2.dct(cv2.dct(framef32))
        dct = np.around(dct)
        high[index] = np.sum(np.abs((dct[356:360, 636:640])))
    np.save(videoname + "_DCT.npy", high)
    draw_linechart(high, videoname + "_DCT")
    print(videoname + " " + str(np.sum(high)))
    return np.sum(high)

def class_DCT(video):
    framelist = getframelist(video)
    video_dct = calculate_dct(framelist, video)
    base = 130000
    if video_dct > base:
        video_dct = base
    ads_std = 7000 # 8454.25
    cartoon_std = 15720.2
    concerts_std = 65506.5
    interview_std = 31305
    movie_std = 1700
    values = []
    values.append((base - abs(video_dct - ads_std)) / base)
    values.append((base - abs(video_dct - cartoon_std)) / base)
    values.append((base - abs(video_dct - concerts_std)) / base)
    values.append((base - abs(video_dct - interview_std)) / base)
    values.append((base - abs(video_dct - movie_std)) / base)
    return values

def get_DCT_data(video):
    data = np.load(video + "_DCT.npy")
    return data.tolist()

def get_DCT(video, videolist):
    video_dct = sum(np.load(video + "_DCT.npy"))
    base = 130000
    if video_dct > base:
        video_dct = base
    values = []
    for item in videolist:
        item_dct = sum(np.load(item + "_DCT.npy"))
        values.append((base - abs(video_dct - item_dct)) / base)
    return values

def calculate_kframes(framelist, videoname):
    frame = cv2.imread(framelist[0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lastHist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    diff = np.empty([480], dtype=int)
    keyframes = np.zeros([480], dtype=int)
    secondHist = np.array([])
    firsthist = lastHist
    for index, framefile in enumerate(framelist):
        frame = cv2.imread(framefile)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        diff[index] = min(np.sum(cv2.absdiff(hist, lastHist)), np.sum(cv2.absdiff(hist, firsthist)))
        if diff[index] > 20000:
            if secondHist.size == 0:
                secondHist = hist
            else:
                diff[index] = min(diff[index], np.sum(cv2.absdiff(hist, secondHist)))
                if diff[index] > 20000:
                    keyframes[index] = 1
        lastHist = hist
    np.save(videoname + "_kframes.npy", keyframes)
    np.save(videoname + "_KFDiff.npy", diff)
    draw_linechart(diff, videoname + "_KFDiff")
    return sum(keyframes)

def class_KeyFrame(video):
    framelist = getframelist(video)
    video_kframes = calculate_kframes(framelist, video)
    base = 270
    if video_kframes > base:
        video_kframes = base
    ads_std = 160
    cartoon_std = 98.4
    concerts_std = 22.5
    interview_std = 0.33
    movie_std = 51
    values = []
    values.append((base - abs(video_kframes - ads_std)) / base)
    values.append((base - abs(video_kframes - cartoon_std)) / base)
    values.append((base - abs(video_kframes - concerts_std)) / base)
    values.append((base - abs(video_kframes - interview_std)) / base)
    values.append((base - abs(video_kframes - movie_std)) / base)
    return values

def get_KeyFrame(video, videolist):
    video_kf = sum(np.load(video + "_kframes.npy"))
    base = 270
    if video_kf > base:
        video_kf = base
    values = []
    for item in videolist:
        item_kf = sum(np.load(item + "_kframes.npy"))
        values.append((base - abs(video_kf - item_kf)) / base)
    return values

def get_KFDiff_data(video):
    data = np.load(video + "_KFDiff.npy")
    return data.tolist()

def build_db_dct():
    pathprefix = "D:\\USC\\CS576\\project\\Data_jpg\\"
    path = ["ads\\ads_0", "ads\\ads_1", "ads\\ads_2", "ads\\ads_3",
            "cartoon\\cartoon_0", "cartoon\\cartoon_1", "cartoon\\cartoon_2", "cartoon\\cartoon_3",
            "cartoon\\cartoon_4",
            "concerts\\concerts_0", "concerts\\concerts_1", "concerts\\concerts_2", "concerts\\concerts_3",
            "interview\\interview_0", "interview\\interview_1", "interview\\interview_2", "interview\\interview_3",
            "interview\\interview_4", "interview\\interview_5",
            "movies\\movies_0", "movies\\movies_1", "movies\\movies_2", "movies\\movies_3", "movies\\movies_4",
            ]
    for videopath in path:
        framelist = getframelist(pathprefix + videopath)
        brk = videopath.find('\\')
        videoname = videopath[(brk + 1):]
        calculate_dct(framelist, videoname)

def build_db_KeyFrame():
    pathprefix = "D:\\USC\\CS576\\project\\Data_jpg\\"
    path = ["ads\\ads_0", "ads\\ads_1", "ads\\ads_2", "ads\\ads_3",
            "cartoon\\cartoon_0", "cartoon\\cartoon_1", "cartoon\\cartoon_2", "cartoon\\cartoon_3",
            "cartoon\\cartoon_4",
            "concerts\\concerts_0", "concerts\\concerts_1", "concerts\\concerts_2", "concerts\\concerts_3",
            "interview\\interview_0", "interview\\interview_1", "interview\\interview_2", "interview\\interview_3",
            "interview\\interview_4", "interview\\interview_5",
            "movies\\movies_0", "movies\\movies_1", "movies\\movies_2", "movies\\movies_3", "movies\\movies_4",
            ]
    for videopath in path:
        framelist = getframelist(pathprefix + videopath)
        brk = videopath.find('\\')
        videoname = videopath[(brk + 1):]
        calculate_kframes(framelist, videoname)

if __name__ == "__main__":
    helper = Helper()
    imagesmoothing = ImageSmoothing()
    visualizer = Visualizer()
    motiondetector = MotionDetector()
    skindetector = SkinDetector()
    keyframedetector = KeyFrameDetector()
    freqdetector = FreqDetector()
    colordistribution = ColorDistribution()
    path = ["ads\\ads_0", "ads\\ads_1", "ads\\ads_2", "ads\\ads_3",
            "cartoon\\cartoon_0", "cartoon\\cartoon_1", "cartoon\\cartoon_2", "cartoon\\cartoon_3",
            "cartoon\\cartoon_4",
            "concerts\\concerts_0", "concerts\\concerts_1", "concerts\\concerts_2", "concerts\\concerts_3",
            "interview\\interview_0", "interview\\interview_1", "interview\\interview_2", "interview\\interview_3",
            "interview\\interview_4", "interview\\interview_5",
            "movies\\movies_0", "movies\\movies_1", "movies\\movies_2", "movies\\movies_3", "movies\\movies_4",
            ]
    pathprefix = "D:\\USC\\CS576\\project\\Data_jpg\\"
    pathads = ["ads\\ads_0", "ads\\ads_1", "ads\\ads_2", "ads\\ads_3"]
    pathcartoon = ["cartoon\\cartoon_0", "cartoon\\cartoon_1", "cartoon\\cartoon_2", "cartoon\\cartoon_3",
                   "cartoon\\cartoon_4"]
    pathconcerts = ["concerts\\concerts_0", "concerts\\concerts_1", "concerts\\concerts_2", "concerts\\concerts_3"]
    pathinterview = ["interview\\interview_0", "interview\\interview_1", "interview\\interview_2",
                     "interview\\interview_3", "interview\\interview_4", "interview\\interview_5"]
    pathmovies = ["movies\\movies_0", "movies\\movies_1", "movies\\movies_2", "movies\\movies_3", "movies\\movies_4"]
    pathsports = ["sport\\sport_0", "sport\\sport_1", "sport\\sport_2", "sport\\sport_4", "sport\\sport_5"]
    # for i, item in enumerate(path1):
    #     framelist = helper.getframelist(pathprefix + path[i])
    #     # sumofdiff = imagesmoothing.smooth(framelist)
    #     # print(np.average(sumofdiff))
    #     # motiondetector.motiondetect(framelist)
    #     keyframedetector.getkeyframes(framelist)

    for item in path:
        framelist = helper.getframelist(pathprefix + item)
        same = colordistribution.coloraround(framelist)
        print(sum(same))
        # high = freqdetector.highfreq(framelist)
        # print(sum(high))
        # diff, keyframes = keyframedetector.getkeyfbyhisthue(framelist)
        # numofkeyframes = np.sum(keyframes)
        # numofminor = np.sum(minordiffframes)
        # # visualizer.plotone(diff)
        # print(numofkeyframes)

    # framepath = pathprefix + "cartoon\\cartoon_4" + "\\frame127.jpg"
    # freqdetector.highfreqframe(framepath)

    # motionVector = MotionVector(path)
    # motionVector.getYChannel()
    # print("y channel completed!")
    # motionVector.getMotionVector()
    # print(motionVector.motionDescriptors)
    # visualizer.plot(sumofdiff, sumofdiff1, sumofdiff2, sumofdiff3, sumofdiff4, sumofdiff5)
