import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path_prefix = "static/Data_local/"
path_query = "static/Data_query/"
path_jpg = "static/Data_jpg"

# def get_dct_video(video, flag):
#     if flag == 0:
#         path = path_prefix
#     else:
#         path = path_query


def draw_linechart(y, imgname, descriptor, flag):
    x = np.arange(480)
    # ylimit = 30000
    # plt.ylim((0, ylimit))
    plt.plot(x, y)
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    os.makedirs(path + descriptor + "_video/" + imgname)
    ymax = np.max(y)
    for i in range(480):
        plt.plot(x, y, 'b')
        plt.vlines(i, 0, ymax, 'r', 'solid')
        plt.savefig(path + descriptor + "_video/" + imgname + "/" + str(i) + ".jpg")
        plt.close()

path = [["ads_0", "ads_1", "ads_2", "ads_3"],
        ["cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4"],
        ["concerts_0", "concerts_1", "concerts_2", "concerts_3"],
        ["interview_0", "interview_1", "interview_2", "interview_3", "interview_4", "interview_5"],
        ["movies_0", "movies_1", "movies_2", "movies_3", "movies_4"],
        ["sport_0", "sport_1", "sport_2", "sport_3", "sport_4"]]
querylist = ["ads_1", "ads_2", "cartoon_1", "cartoon_2", "concert_1", "interview_1", "interview_2", "movies_1", "movies_2", "sport_1"]

for video in querylist:
    data = np.load(path_query + "DCT/" + video + "_DCT.npy")
    draw_linechart(data, video, "DCT", 1)
    data = np.load(path_query + "KF/" + video + "_KFDiff.npy")
    draw_linechart(data, video, "KF", 1)

# for videos in path:
#     for video in videos:
#         data = np.load(path_prefix + "DCT/" + video + "_DCT.npy")
#         draw_linechart(data, video, "DCT", 0)

# for videos in path:
#     for video in videos:
#         data = np.load(path_prefix + "KF/" + video + "_KFDiff.npy")
#         draw_linechart(data, video, "KF", 0)
