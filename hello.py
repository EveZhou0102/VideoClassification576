import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

###zq
path_prefix = "static/Data_local/"
path_query = "static/Data_query/"
path_jpg = "static/Data_jpg"

def smooth(framelist):
    sumofdiff = np.empty(480)
    for index, framefile in enumerate(framelist):
        frame = cv2.imread(framefile)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blur = cv2.bilateralFilter(h, 2, 100, 100)
        sumofdiff[index] = (h - blur).sum()
    return sumofdiff

def draw_linechart(y, imgname, descriptor, flag):
    x = np.arange(480)
    # ylimit = 30000
    # plt.ylim((0, ylimit))
    plt.plot(x, y)
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    plt.savefig(path + descriptor + "/" + imgname + ".png")
    plt.close()

# def getframelist(video):
#     pathprefix = "D:\\USC\\CS576\\project\\query\\test_jpg"
#     videopath = pathprefix + "\\" + video
#     framelist = os.listdir(videopath)
#     framelist.sort(key=lambda x: int(x[5:-4]))
#     validframes = [videopath + "\\" + x for x in framelist[:480]]
#     return validframes

def getframelist(video):
    pathprefix = "D:\\USC\\CS576\\project\\Data_jpg\\"
    i = video.find('_')
    videoclass = video[0:i]
    videopath = pathprefix + videoclass + "\\" + video
    framelist = os.listdir(videopath)
    framelist.sort(key=lambda x: int(x[5:-4]))
    validframes = [videopath + "\\" + x for x in framelist[:480]]
    return validframes

def calculate_dct(framelist, videoname, flag):
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
        print(dct[356:360, 636:640])
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    np.save(path + "DCT/" + videoname + "_DCT.npy", high)
    draw_linechart(high, videoname + "_DCT", "DCT", flag)
    return np.sum(high)


def class_DCT(video):
    video_dct = np.sum(np.load(path_query + "DCT/" + video + "_DCT.npy"))
    print("DCT = " + str(video_dct))
    values = np.array([7000, 15720.2, 65506.5, 31305, 1700])
    values = np.abs(video_dct - values)
    values = nor1(values)
    return values.tolist()


def get_DCT_data(video, flag):
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    data = np.load(path + "DCT/" + video + "_DCT.npy")
    return data.tolist()


def get_DCT(video, videolist):
    video_dct = sum(np.load(path_query + "DCT/" + video + "_DCT.npy"))
    values = np.zeros([len(videolist)])
    for i,item in enumerate(videolist):
        values[i] = sum(np.load(path_prefix + "DCT/" + item + "_DCT.npy"))
    values = np.abs(video_dct - values)
    values = nor1(values)
    return values.tolist()


def calculate_kframes(framelist, videoname, flag):
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
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    np.save(path + "KF/" + videoname + "_kframes.npy", keyframes)
    np.save(path + "KF/" + videoname + "_KFDiff.npy", diff)
    draw_linechart(diff, videoname + "_KFDiff", "KF", flag)
    return sum(keyframes)

def class_KeyFrame(video):
    framelist = getframelist(video)
    video_kframes = np.sum(np.load(path_query + "KF/" + video + "_kframes.npy"))
    print("kframes = " + str(video_kframes))
    # values = np.array([160, 98.4, 22.5, 0.33, 51])
    values = np.array([160, 98, 8, 1, 40])
    values = np.abs(video_kframes - values)
    values = nor1(values)
    return values.tolist()

# def class_KeyFrame(video):
#     video_diff = np.load(path_query + "KF/" + video + "_KFDiff.npy")
#     path = [["ads_0", "ads_1", "ads_2", "ads_3"],
#             ["cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4"],
#             ["concerts_0", "concerts_1", "concerts_2", "concerts_3"],
#             ["interview_0", "interview_1", "interview_2", "interview_3", "interview_4", "interview_5"],
#             ["movies_0", "movies_1", "movies_2", "movies_3", "movies_4"],
#             ["sport_0", "sport_1", "sport_2", "sport_3", "sport_4"]]
#     values = np.zeros([5], int)
#     for i in range(5):
#         for item in path[i]:
#             item_diff = np.load(path_prefix + "KF/" + item + "_KFDiff.npy")
#             values[i] += dtw([video_diff], [item_diff])
#         values[i] = values[i] / len(path[i])
#     values = nor1(values)
#     return values.tolist()

def get_KeyFrame(video, videolist):
    video_diff = np.load(path_query + "KF/" + video + "_KFDiff.npy")
    values = np.zeros([len(videolist)], int)
    for i, item in enumerate(videolist):
        item_diff = np.load(path_prefix + "KF/" + item + "_KFDiff.npy")
        values[i] = dtw([video_diff], [item_diff])
    values = nor1(values)
    return values.tolist()

def get_KFDiff_data(video, flag):
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    data = np.load(path + "KF/" + video + "_KFDiff.npy")
    return data.tolist()

def dtw(M1, M2) :
    # 初始化数组 大小为 M1 * M2
    M1_len = len(M1)
    M2_len = len(M2)
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]

    # 初始化 dis 数组
    dis = []
    for i in range(M1_len) :
        dis_row = []
        for j in range(M2_len) :
            dis_row.append(distance(M1[i], M2[j]))
        dis.append(dis_row)

    # 初始化 cost 的第 0 行和第 0 列
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len) :
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len) :
        cost[0][j] = cost[0][j - 1] + dis[0][j]

    # 开始动态规划
    for i in range(1, M1_len) :
        for j in range(1, M2_len) :
            cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1, \
                            cost[i- 1][j - 1] + dis[i][j] * 2, \
                            cost[i][j - 1] + dis[i][j] * 1)
    return cost[M1_len - 1][M2_len - 1]

# 两个维数相等的向量之间的距离
def distance(x1, x2) :
    sum = 0
    for i in range(len(x1)) :
        sum = sum + abs(x1[i] - x2[i])
    return sum

def nor1(point):
    SUM = sum(point)
    point = point/SUM
    return point

def nor2(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# build_db_dct()

# build_db_KeyFrame()

querylist = ["ads_1", "ads_2", "cartoon_1", "cartoon_2", "concert_1", "interview_1", "interview_2", "movies_1", "movies_2", "sport_1"]
path = [["ads_0", "ads_1", "ads_2", "ads_3"],
        ["cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4"],
        ["concerts_0", "concerts_1", "concerts_2", "concerts_3"],
        ["interview_0", "interview_1", "interview_2", "interview_3", "interview_4", "interview_5"],
        ["movies_0", "movies_1", "movies_2", "movies_3", "movies_4"],
        ["sport_0", "sport_1", "sport_2", "sport_3", "sport_4"]]
# for videos in path:
#     for video in videos:
#         # print(np.average(smooth(getframelist(video))))
#         calculate_dct(getframelist(video), video, 0)
calculate_dct(getframelist("movies_4"), "movies_4", 0)
# for video in querylist:
#     framelist = getframelist(video)
#     # calculate_kframes(framelist, video, 1)
#     # calculate_dct(framelist, video, 1)
#     # print("现在查找：" + video)
#     # print("DCT分类：")
#     # print(class_DCT(video))
#     # print("KeyFrame分类：")
#     # print(class_KeyFrame(video))
#     # print(video)
#     np.save(path_query + "SM/" + video + ".npy", np.average(smooth(framelist)))
    # print(np.average(smooth(framelist)))

# h = np.array([6000000, 4000000, 4])
# np.save(path_prefix + "statistics.npy", h)
