import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from ortools.graph import pywrapgraph
import math
import csv


# Data_Path
# llw
Data_Path = 'static'
Data_Query = 'static/Data_query'
Data_MFCC = 'static/Data_local/MFCC_cut20.npy'
Data_Des_MFCC = 'static/Data_local/MFCC_jpg'
Data_DTW = 'static/Data_local/video_Dtw.npy'

### zy
Image_Path = 'static/Data_jpg'
Data_Path_MC_l = 'static/Data_local'
Data_Path_MC_q = 'static/Data_query'
N = 8

###zq
path_prefix = "static/Data_local/"
path_jpg = "static/Data_jpg"
path_query = "static/Data_query/"

### web
Video_Path = 'static/Data_mp4'
videoInfo = {}

### llw

def classification(video):
    result = np.zeros((6),dtype = float)

    # Step 1: If sport or not
    if class_isSport(video):
        result[5] = -1
        return result
    result[5] = 999

    # Step 2: If not sport:
    ## Feature: 1*5
    MFCC = class_MFCC(video)
    DCT = class_DCT(video)
    KeyFrame = class_KeyFrame(video)
    Color = class_Color(video)

    ## Feature Matrix: 5*4
    feature = (np.array([MFCC,DCT,KeyFrame,Color])).T
    # feature=np.log10(100*feature)
    # for i in range(0,len(feature)):
    #     feature[i] = nor1(feature[i])

    # np.save(path_prefix+video+'_class_f.npy',feature)

    ## Weight Matrix:  4*5
    weight = np.array([
        (0.7,0.1,0.05,0,0.4),
        (0,0,0,0,0.5),
        (0.3,0.4,0.9,1,0.1),
        (0,0.5,0.05,0,0)
    ])

    ## Point Matrix: Feature X Weight = 5*5
    point = feature.dot(weight)

    ##visualization
    # print("Feature Matrix")
    # for i in range(0,4):
    #     print(feature[:,i])


    # Step 3： Calculate point Matrix
    for i in range(0,5):
        result[i] = point[i][i]

    # Step 4: Using local data to refine result
    result = refine(result,video)



    return result

def ranking(video, cate):
    Categories = ['ads','cartoon','concerts','interview','movies','sport']
    Num = [4,5,4,6,5,5]
    Normal = [1,3,4,5]
    Min = -1
    Sec = -1
    result = []
    points = []



    # Step 1: Find the category
    ## Todo: What if two group has same points?
    Min = np.argmin(cate)
    if Min not in Normal:
        cate[Min] = 999
        Sec = np.argmin(cate)


    # Step 2: Generate the Video list
    video_list = []
    sec_list = []
    # Add the First cate
    for i in range(0,Num[Min]):
          video_list.append(Categories[Min]+'_'+str(i))

    if Sec!=-1:
        for i in range(0,Num[Sec]):
            sec_list.append(Categories[Sec]+'_'+str(i))


    # Step 3: Calculate the Ranking

    ## Weight Matrix: 6*3
    weight = np.array([
        (0.5,0.25,0.25),
        (0.1,0.9,0.0),
        (0.6,0.2,0.2),
        (0.5,0.6,0.1),
        (0.5,0.2,0.3),
        (0.5,0.4,0.1)
    ])

    ## Calculate the Main Ranking:
    MFCC = get_MFCC(video, video_list)
    Color = get_Color(video, video_list)
    KeyFrame = get_KeyFrame(video,video_list)
    ### Feature Matrix: Num[Max]*3
    feature = (np.array([MFCC,Color,KeyFrame])).T
    point = feature.dot(weight[Min])
    # print(feature)
    # print(point)

    ## Calculate the Sec Ranking
    if Sec!=-1:
        MFCC_sec = get_MFCC(video, sec_list)
        Color_sec = get_Color(video, sec_list)
        KeyFrame_sec = get_KeyFrame(video,sec_list)
        ### Feature Matrix: 3*Num[Max]
        feature_sec = (np.array([MFCC_sec,Color_sec,KeyFrame_sec])).T
        np.save(path_prefix+video+'_rank2_f.npy',feature_sec)
        point_sec = feature.dot(weight[Sec])

    # Step 3: Return the final result
    if Sec==-1:
        for i in range(0,5):
            Min = np.argmin(point)
            result.append(video_list[Min])
            points.append(point[Min])
            point[Min] = 999
    else :
        for i in range(0,4):
            Min = np.argmin(point)
            result.append(video_list[Min])
            points.append(point[Min])
            point[Min] = 999
        result.append(sec_list[np.argmin(point_sec)])
        points.append(point_sec[np.argmin(point_sec)])

    # Step 4: Generate ranking point
    if Sec!=-1:
        points[4] = str(float(points[3])+0.1*float(points[4]))
    points = nor1(np.array(points,dtype=float))

    points = 1-points
    return result,points.tolist()
    # return result

def class_MFCC(video):
    video_Dtw = []

    # Step 1: Get source video MFCC vector
    audio = os.path.join(Data_Query,"MFCC",video+".npy")
    video_MFCC = np.load(audio)


    # Step 2: Load local videos MFCC vector
    MFCC = np.load(Data_MFCC)


    # Step 3: Calculate AN Distance
    Cate_Range = [[0,4],[4,9],[9,13],[13,19],[19,24]]
    Num = [4,5,4,6,5]
    AN_Distance = []

    ## Calculate Average for each category
    for i in range(0,5):
        cate_sum = 0
        for a in range(Cate_Range[i][0],Cate_Range[i][1]):
            dtw_now = dtw(video_MFCC,MFCC[a])
            video_Dtw.append(dtw_now)
            cate_sum = cate_sum+dtw_now
        AN_Distance.append(cate_sum*1.0/Num[i])

    ## Calculate Sport and save video_Dtw for later use
    for i in range(24,29):
        video_Dtw.append(dtw(video_MFCC,MFCC[i]))
    video_Dtw = np.array(video_Dtw)
    np.save(Data_DTW,video_Dtw)

    ## Normalize
    AN_Distance = np.array(AN_Distance)
    AN_Distance = nor2(AN_Distance)




    # Step 4: Return final result
    return AN_Distance.tolist()


# print(class_MFCC("ads_0"))

def get_MFCC(video, video_list):
    # Step1 : load the local data
    video_Dtw = np.load(Data_DTW)

    # Step 2: Return final Result
    Result = []
    Categories = ['ads','cartoon','concerts','interview','movies','sport']
    Cate_Range = [[0,4],[4,9],[9,13],[13,19],[19,24],[24,29]]

    ## Find the class
    cate = (video_list[0]).split("_")[0]
    num = Categories.index(cate)

    ## Add the result
    for i in range(Cate_Range[num][0],Cate_Range[num][1]):
        Result.append(video_Dtw[i])

    ## Normarlize the result
    Result = np.array(Result)
    Result = nor2(Result)

    return Result.tolist()


# l = ['ads_0','ads_1']
# print(get_MFCC("1",l))

def get_descri_MFF(video,flag):
    # Dataset
    if flag==0:
        return os.path.join(Data_Des_MFCC,video+'.png')
    return os.path.join(Data_Query,'MFCC_jpg',video+'.png')

def refine(result,video):
    Sta = np.load(os.path.join(Data_Path,'Data_local','statistics.npy'))
    Sta_concert = Sta[0]
    Sta_movie = Sta[1]
    Sta_interview = Sta[2]

    ## Concert&& Movie
    av_sm = np.load(os.path.join(Data_Query,'SM',video+".npy"))
    if av_sm < Sta_concert:
        result[2]=999
    if av_sm < Sta_movie:
        result[4] = 999

    ## Interview
    num_key = sum(np.load(os.path.join(Data_Query,'KF',video+'_kframes.npy')))
    if num_key>Sta_interview:
        result[3]=999


    ## Cartoon VS Ads
    if result[2]==999 and result[3]==999 and result[4]==999 and abs(result[0]-result[1])<0.5:
        video_list = ["ads_0", "ads_1", "ads_2", "ads_3"]
        xx = ["cartoon_0", "cartoon_1", "cartoon_2", "cartoon_3", "cartoon_4"]

        weight = np.array([
            (0.5,0.25,0.25),
            (0.1,0.9,0),
            (0.6,0.2,0.2),
            (0.5,0.4,0.1),
            (0.5,0.2,0.3),
            (0.5,0.4,0.1)
        ])

        MFCC = get_MFCC(video, video_list)
        Color = get_Color(video, video_list)
        KeyFrame = get_KeyFrame(video,video_list)

        feature = (np.array([MFCC,Color,KeyFrame])).T
        point = feature.dot(weight[0])
        av1 = np.average(point)


        MFCC = get_MFCC(video, xx)
        Color = get_Color(video,xx)
        KeyFrame = get_KeyFrame(video,xx)
        feature = (np.array([MFCC,Color,KeyFrame])).T
        point = feature.dot(weight[1])
        av2 = np.average(point)

        if 3*(av1-av2)+result[0]-result[1]<0:
            result[1] = 998
        else :
            result[0] = 998
        #
        # print("av1: " + str(av1))
        # print("av2: " + str(av2))
        # print("av1-av2=" + str(av1-av2))
        # print("result= " + str(result[0] - result[1]))

    return result

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
    #print(SUM)
    point = point/SUM
    return point

def nor2(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

####ZQ


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

def getframelist(video):
    videopath = path_jpg + "/" + video
    framelist = os.listdir(videopath)
    framelist.sort(key=lambda x: int(x[5:-4]))
    validframes = [videopath + "/" + x for x in framelist[:480]]
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
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    np.save(path + "DCT/" + videoname + "_DCT.npy", high)
    draw_linechart(high, videoname + "_DCT", "DCT", flag)
    return np.sum(high)


def class_DCT(video):
    video_dct = np.sum(np.load(path_query + "DCT/" + video + "_DCT.npy"))
    values = np.array([7000, 15720.2, 65506.5, 31305, 1700])
    values = np.abs(video_dct - values)
    values = nor2(values)
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
    values = nor2(values)
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
    video_kframes = np.sum(np.load(path_query + "KF/" + video + "_kframes.npy"))
    #print("kframes = " + str(video_kframes))
    values = np.array([160, 98.4, 8, 0.33, 40])
    values = np.abs(video_kframes - values)
    values = nor2(values)
    return values.tolist()

def get_KeyFrame(video, videolist):
    video_diff = np.load(path_query + "KF/" + video + "_KFDiff.npy")
    values = np.zeros([len(videolist)], int)
    for i, item in enumerate(videolist):
        item_diff = np.load(path_prefix + "KF/" + item + "_KFDiff.npy")
        values[i] = dtw([video_diff], [item_diff])
    values = nor2(values)
    return values.tolist()

def get_KFDiff_data(video, flag):
    if flag == 0:
        path = path_prefix
    else:
        path = path_query
    data = np.load(path + "KF/" + video + "_KFDiff.npy")
    return data.tolist()

def smooth(framelist):
    sumofdiff = np.empty(480)
    for index, framefile in enumerate(framelist):
        frame = cv2.imread(framefile)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blur = cv2.bilateralFilter(h, 2, 100, 100)
        sumofdiff[index] = (h - blur).sum()
    return sumofdiff


### zy


# calculate distance between two pixel of color
def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

# build the network flow graph
def build_graph(color1, pc1, color2, pc2):
    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []

    # add edge (source, img1)
    for i in range(N):
        start_nodes.append(0)
        end_nodes.append(i+1)
        capacities.append(pc1[i])
        unit_costs.append(0)

    # add edge (img1, img2)
    for i in range(N):
        for j in range(N):
            start_nodes.append(i+1)
            end_nodes.append(j+N+1)
            capacities.append(200)
            unit_costs.append(ColourDistance(color1[i],color2[j]))

    # add edge (img2, ends)
    for i in range(N):
        start_nodes.append(i+N+1)
        end_nodes.append(2*N+1)
        capacities.append(pc2[i])
        unit_costs.append(0)

    supplies = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100]
    return start_nodes, end_nodes, capacities, unit_costs, supplies

# calculate distance between color distribution of two video
def getDist(color1, pc1, color2, pc2):
    start_nodes, end_nodes, capacities, unit_costs, supplies = build_graph(color1, pc1, color2, pc2)
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(round(start_nodes[i])), int(round(end_nodes[i])),
                                                    int(round(capacities[i])), int(round(unit_costs[i])))
    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

     # Find the minimum cost flow
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        return min_cost_flow.OptimalCost()
    else:
        print('There was an issue with the min cost flow input.')
        return -1

# load local data
def readLocalData():
    color = {}
    pct = {}
    dic = {'ads':4, 'cartoon':5, 'concerts':4, 'interview':6, 'movies':5, 'sport':5}
    #f = open("/Users/zy/PycharmProjects/576Web/mainColor/test.csv", "r")
    fpath = Data_Path_MC_l +'/MainColor.csv'
    f = open(fpath, "r")
    line = f.readline()
    while line:
        key = f.readline().strip('\n')
        for j in range(dic[key]):
            cpct=[]
            ccolor = []
            # read pct
            for i in range(N):
                pc = round(float(f.readline().strip('\n')) * 100)
                cpct.append(pc)
            # read color
            for i in range(N):
                tmp = f.readline().strip('\n').split(',')
                col = []
                for c in tmp:
                    col.append(round(float(c)))
                ccolor.append(col)

            # proce for pc
            if np.sum(cpct)!=100:
                tindex = cpct.index(max(cpct))
                cpct[tindex] = int(cpct[tindex] + 100 - np.sum(cpct))
            pct[key+"_"+str(j)] = cpct
            color[key+"_"+str(j)] = ccolor
        line = f.readline()
    f.close()
    return color,pct

# read local saved query info
def readLocalQueryMC(video):
    color_q = []
    pct_q = []
    fpath = Data_Path_MC_q +'/MainColor/'+video+'.csv'
    f = open(fpath, "r")
    # read pct
    for i in range(N):
        pct = round(float(f.readline().strip('\n')) * 100)
        pct_q.append(pct)
    # read color
    for i in range(N):
        tmp = f.readline().strip('\n').split(',')
        color = []
        for c in tmp:
            color.append(round(float(c)))
        color_q.append(color)

    f.close()
    # proce for pc
    if np.sum(pct_q)!=100:
        tindex = pct_q.index(max(pct_q))
        pct_q[tindex] = int(pct_q[tindex] + 100 - np.sum(pct_q))

    return color_q,pct_q


# test whether a video is a sports video
#def isSports(color, pc):
def class_isSport(video):
    #key,index = videoID.split('_')
    # N=8
    # color_q, pct_q = getMainColor(video,N)
    # # save to local
    # with open(Data_Path_zy + "/query_mainColor.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerows(hist)
    #     writer.writerows(map(lambda x: [x], pct_q))
    #     writer.writerows(color_q)

    color_q, pct_q = readLocalQueryMC(video) # List
    # RGB to HSV
    colorImg =[]
    colorImg.append(color_q)
    colorImg = cv2.cvtColor(np.float32(colorImg), cv2.COLOR_RGB2HSV)
    color = colorImg[0]
    pct_total = 0
    for i in range(len(color)):
        c = color[i]
        #if 40 < c[0] and c[0] < 200 and c[1] > 0.2 and c[1] < 0.9 and c[2] > 80 and c[2] < 160:
        if 60 < c[0] and c[0] < 160 and c[1] > 0.2 and c[1] < 0.7 and c[2] > 80 and c[2] < 160:
            pct_total = pct_total + pct_q[i]
            #print(color[i])
    #print(pct_total)
    if pct_total > 40:
        return True
    return False

def get_Color(video, videoList):
    # load local data
    color_l, pct_l = readLocalData() # dict
    # calculate main color for the query video
    #startframe = 0
    #col, pc = getMainColor(video, startframe,N)
    color_q, pct_q = readLocalQueryMC(video)

    distList = []
    # get data of video in list
    for v in videoList:
        color_v = color_l[v]
        pct_v = pct_l[v]
        score = getDist(color_q, pct_q, color_v, pct_v)
        distList.append(score)
    distList = np.array(distList)
    return nor2(distList).tolist()

def class_Color(video):
    # calculate main color of the video
    #col, pc = getMainColor(video, N)
    color_q, pct_q = readLocalQueryMC(video) # List
    color_l, pct_l = readLocalData() # dict
    # calculate distance from the query video to every video
    distList = []
    dist_avg =  {'ads':0, 'cartoon':0, 'concerts':0, 'interview':0, 'movies':0}
    dic_cnt = {'ads':4, 'cartoon':5, 'concerts':4, 'interview':6, 'movies':5}
    # get data of video in list
    for key in color_l:
        type_key = key.split('_')[0]
        if type_key == 'sport':
            continue
        color_key = color_l[key]
        pct_key = pct_l[key]
        dist_key = getDist(color_q, pct_q, color_key, pct_key)
        dist_avg[type_key] = dist_avg[type_key] + dist_key

    res = []
    # print(dist_avg)
    for key in dist_avg:
        dist_avg[key] = dist_avg[key] / dic_cnt[key]
        res.append(dist_avg[key])
    return nor2(np.array(res)).tolist()

def Test():
    # video = 'ads_1'
    vlist = ['ads_1','ads_2','cartoon_1','cartoon_2','concert_1','interview_1','interview_2','movies_1','movies_2','sport_1']
    for video in vlist:
        print(video+"=========================")
        cate = classification(video)
        print(cate)
        print(ranking(video, cate)[0])
    # video = 'cartoon_1'
    # cate = classification(video)
    # print(ranking(video,cate))

    # color_l, pct_l = readLocalData() # dict
    # for q_video in vlist:
    #     print("query:"+q_video)
    #     for l_video in vlist:
    #         print("match:"+l_video)
    #         print()
    #     #print(class_isSport(video))
    #     print(class_Color(video))

if __name__ == '__main__':
    #app.run()
    Test()
