import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def draw_linechart(y, imgname, descriptor):
    x = np.arange(480)
    # ylimit = 30000
    # plt.ylim((0, ylimit))
    plt.plot(x, y)
    plt.savefig("/static/Data_local/" + descriptor + "/" + imgname + ".png")
    plt.close()

def getframelist(video):
    pathprefix = "/static/Data_jpg"
    videopath = pathprefix + "/" + video
    framelist = os.listdir(videopath)
    framelist.sort(key=lambda x: int(x[5:-4]))
    validframes = [videopath + "/" + x for x in framelist[:480]]
    return validframes


def calculate_dct(framelist, videoname):
    high = np.zeros([480], int)
    for index, framefile in enumerate(framelist):
        frame = cv2.imread(framefile)
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame1.shape[:2]
        framef32 = np.zeros((h, w), np.float32)
        framef32[:h, :w] = frame1
        dct = cv2.dct(cv2.dct(framef32))
        dct = np.around(dct)
        high[index] = np.sum(np.abs((dct[356:360, 636:640])))
    np.save("/static/Data_local/DCT/" + videoname + "_DCT.npy", high)
    draw_linechart(high, videoname + "_DCT", "DCT")
    print(videoname + " " + str(np.sum(high)))
    return np.sum(high)


def class_DCT(video):
    framelist = getframelist(video)
    video_dct = calculate_dct(framelist, video)
    values = np.array([7000, 15720.2, 65506.5, 31305, 1700])
    values = np.abs(video_dct - values)
    values = nore(values)
    return values.tolist()


def get_DCT_data(video):
    data = np.load("/static/Data_local/DCT/" + video + "_DCT.npy")
    return data.tolist()


def get_DCT(video, videolist):
    video_dct = sum(np.load("/static/Data_local/DCT/" + video + "_DCT.npy"))
    base = 130000
    if video_dct > base:
        video_dct = base
    values = []
    for item in videolist:
        item_dct = sum(np.load("/static/Data_local/DCT/" + item + "_DCT.npy"))
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
    np.save("/static/Data_local/KF/" + videoname + "_kframes.npy", keyframes)
    np.save("/static/Data_local/KF/" + videoname + "_KFDiff.npy", diff)
    draw_linechart(diff, videoname + "_KFDiff", "KF")
    return sum(keyframes)


def class_KeyFrame(video):
    framelist = getframelist(video)
    video_kframes = calculate_kframes(framelist, video)
    values = np.array([160, 98.4, 22.5, 0.33, 51])
    values = np.abs(video_kframes - values)
    values = nore(values)
    return values.tolist()


def get_KeyFrame(video, videolist):
    video_kf = sum(np.load("/static/Data_local/KF/" + video + "_kframes.npy"))
    base = 270
    if video_kf > base:
        video_kf = base
    values = []
    for item in videolist:
        item_kf = sum(np.load("/static/Data_local/KF/" + item + "_kframes.npy"))
        values.append((base - abs(video_kf - item_kf)) / base)
    return values


def get_KFDiff_data(video):
    data = np.load("/static/Data_local/KF/" + video + "_KFDiff.npy")
    return data.tolist()


def build_db_dct():
    path = ["ads_0", "ads_1", "ads_2", "ads_3",
            "cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4",
            "concerts_0", "concerts_1", "concerts_2", "concerts_3",
            "interview_0", "interview_1", "interview_2", "interview_3", "interview_4", "interview_5",
            "movies_0", "movies_1", "movies_2", "movies_3", "movies_4"]
    for videopath in path:
        framelist = getframelist(videopath)
        brk = videopath.find('\\')
        videoname = videopath[(brk + 1):]
        calculate_dct(framelist, videoname)


def build_db_KeyFrame():
    path = ["ads_0", "ads_1", "ads_2", "ads_3",
            "cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4",
            "concerts_0", "concerts_1", "concerts_2", "concerts_3",
            "interview_0", "interview_1", "interview_2", "interview_3", "interview_4", "interview_5",
            "movies_0", "movies_1", "movies_2", "movies_3", "movies_4"]
    for videopath in path:
        framelist = getframelist(videopath)
        brk = videopath.find('\\')
        videoname = videopath[(brk + 1):]
        calculate_kframes(framelist, videoname)

# build_db_dct()

# build_db_KeyFrame()

# print(get_DCT_data("ads_0"))
