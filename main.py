import environment
from environment import Environment
import result
from result import CaptureResult
from result import LOSResult
from result import TargetResult
from draw import Drawer
import debug

import cv2

import time
import numpy
from argparse import ArgumentParser
from collections import deque
import collections
import itertools

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import threading

env = environment.load()

class Relay:
    mainCaptureResult, subCaptureResult = None, None
    mainLOSResultDequeue, subLOSResultDequeue = None, None
    lock = None
    velocity = None
    mainFps, subFps = 0, 0
    mainQueSize, subQueSize = 0, 0
    frames, timeInterval, distance = None, None, None
    targetResult = None
    showGraph = False
    debugTargetResultList = None
    stopped = False

    def __init__(self, showGraph, saveDebugData):
        self.mainLOSResultDequeue = deque([])
        self.subLOSResultDequeue = deque([])
        self.lock = Lock()
        self.showGraph = showGraph
        if saveDebugData:
            self.debugTargetResultList =[]

def getOption():
    argparser = ArgumentParser()
    argparser.add_argument('-cm', '--calibrateMainCamera', action='store_true')
    argparser.add_argument('-cs', '--calibrateSubCamera', action='store_true')
    argparser.add_argument('-pm', '--previewMainCamera', action='store_true')
    argparser.add_argument('-ps', '--previewSubCamera', action='store_true')
    argparser.add_argument('-am', '--adjustMainCamera', action='store_true')
    argparser.add_argument('-as', '--adjustSubCamera', action='store_true')
    argparser.add_argument('-g', '--showGraph', action='store_true')
    argparser.add_argument('-s', '--saveDebugData', action='store_true')
    argparser.add_argument('-l', '--loadDebugData', type=int)
    return argparser.parse_args()

def initCamera(camera):

    deviceId = camera.id

    if deviceId is None:
        return None

    capture = cv2.VideoCapture(int(deviceId))

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    capture.set(cv2.CAP_PROP_FPS, camera.fps)

    return capture

def calibrate(camera):

    patternPoints = numpy.zeros( (numpy.prod((env.calibration.patternWidth, env.calibration.patternHeight)), 3), numpy.float32 )
    patternPoints[:,:2] = numpy.indices((env.calibration.patternWidth, env.calibration.patternHeight)).T.reshape(-1, 2)
    patternPoints *= env.calibration.squareSize
    objpoints = []
    imgpoints = []
    isInterrupted = False

    capture = initCamera(camera)

    while len(objpoints) < env.calibration.referenceImageNum:

        ret, img = capture.read()
        img = cv2.resize(img , (env.width, env.height))
        height = img.shape[0]
        width = img.shape[1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corner = cv2.findChessboardCorners(gray, (env.calibration.patternWidth, env.calibration.patternHeight))

        if ret == True:
            print(str(len(objpoints)+1) + "/" + str(env.calibration.referenceImageNum))
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corner, (5,5), (-1,-1), term)
            imgpoints.append(corner.reshape(-1, 2))
            objpoints.append(patternPoints)

        cv2.imshow('image', img)
        if cv2.waitKey(100) >= 0:
            isInterrupted = True
            break

    if isInterrupted:
        print("interrupted.")
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("mtx: {}".format(mtx))
        print("dist: {}".format(dist))
        print("-")
        print("mtx_fx = {}".format(mtx[0][0]))
        print("mtx_fy = {}".format(mtx[1][1]))
        print("mtx_cx = {}".format(mtx[0][2]))
        print("mtx_cy = {}".format(mtx[1][2]))
        print("dist_k1 = {}".format(dist[0][0]))
        print("dist_k2 = {}".format(dist[0][1]))
        print("dist_p1 = {}".format(dist[0][2]))
        print("dist_p2 = {}".format(dist[0][3]))
        print("dist_k3 = {}".format(dist[0][4]))

    capture.release()
    cv2.destroyAllWindows()

def previewPoints(camera):

    capture = initCamera(camera)

    print("measure top left point (blue circle) and center point (yellow circle).")

    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame , (env.width, env.height))
        frame = cv2.undistort(frame, camera.mtx, camera.dist, None)
        cv2.circle(frame, (0, 0), 10, (0, 255, 255), 2)
        cv2.circle(frame, (int(env.width/2), int(env.height/2)), 10, (0, 0, 255), 2)

        cv2.imshow('preview', frame)
        if cv2.waitKey(100) >= 0:
            break

    capture.release()
    cv2.destroyAllWindows()

def convertFrame(frame, hsvmin, hsvmax, medianBlurSize):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    binaryFrame = cv2.inRange(hsvFrame, hsvmin, hsvmax)
    resultFrame = cv2.medianBlur(binaryFrame, medianBlurSize)
    return hsvFrame, binaryFrame, resultFrame

def adjust(camera):
    def nothing(x):
        pass
    hmin, smin, vmin = camera.hsvmin
    hmax, smax, vmax = camera.hsvmax
    medianBlurSize = camera.medianBlurSize

    whiteFrame = numpy.ones((env.height, env.width, 3),numpy.uint8)*0

    cv2.imshow("adjust parameters", whiteFrame)

    cv2.createTrackbar("H min", "adjust parameters", 0, 179, nothing)
    cv2.createTrackbar("H max", "adjust parameters", 0, 179, nothing)
    cv2.createTrackbar("S min", "adjust parameters", 0, 255, nothing)
    cv2.createTrackbar("S max", "adjust parameters", 0, 255, nothing)
    cv2.createTrackbar("V min", "adjust parameters", 0, 255, nothing)
    cv2.createTrackbar("V max", "adjust parameters", 0, 255, nothing)
    cv2.createTrackbar("Median Blur", "adjust parameters", 0, 5, nothing)

    cv2.setTrackbarPos("H min", "adjust parameters", hmin)
    cv2.setTrackbarPos("H max", "adjust parameters", hmax)
    cv2.setTrackbarPos("S min", "adjust parameters", smin)
    cv2.setTrackbarPos("S max", "adjust parameters", smax)
    cv2.setTrackbarPos("V min", "adjust parameters", vmin)
    cv2.setTrackbarPos("V max", "adjust parameters", vmax)
    cv2.setTrackbarPos("Median Blur", "adjust parameters", int(medianBlurSize / 2))

    cv2.moveWindow("adjust parameters", 0, 0)
    cv2.imshow("original frame", whiteFrame)
    cv2.moveWindow("original frame", env.width, 0)
    cv2.imshow("hsv frame", whiteFrame)
    cv2.moveWindow("hsv frame", env.width * 2, env.height)
    cv2.imshow("binary frame", whiteFrame)
    cv2.moveWindow("binary frame", env.width, env.height)

    capture = initCamera(camera)

    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame , (env.width, env.height))

        hmin = cv2.getTrackbarPos("H min", "adjust parameters")
        hmax = cv2.getTrackbarPos("H max", "adjust parameters")
        smin = cv2.getTrackbarPos("S min", "adjust parameters")
        smax = cv2.getTrackbarPos("S max", "adjust parameters")
        vmin = cv2.getTrackbarPos("V min", "adjust parameters")
        vmax = cv2.getTrackbarPos("V max", "adjust parameters")
        medianBlurSize = cv2.getTrackbarPos("Median Blur", "adjust parameters") * 2 + 1

        hsvmin = (hmin, smin, vmin)
        hsvmax = (hmax, smax, vmax)

        hsvFrame, binaryFrame, resultFrame = convertFrame(frame, hsvmin, hsvmax, medianBlurSize)

        cv2.imshow("adjust parameters", resultFrame)
        cv2.imshow("original frame", frame)
        cv2.imshow("hsv frame", hsvFrame)
        cv2.imshow("binary frame", binaryFrame)

        key = cv2.waitKey(10)

        if key >= 0:
            print("hue_min = {}".format(hmin))
            print("hue_max = {}".format(hmax))
            print("saturation_min = {}".format(smin))
            print("saturation_max = {}".format(smax))
            print("value_min = {}".format(vmin))
            print("value_max = {}".format(vmax))
            print("median_blur_size = {}".format(medianBlurSize))
            break

    capture.release()
    cv2.destroyAllWindows()

# 3次元座標系において点a,bを通る直線と点c,dを通る直線の共通垂線を導出する
def calculateCommonPerpendicular(a, b, c, d):
    # http://math-juken.com/kijutu/kyoutusuisen/ の「共通垂線で解く」
    """
    →AB = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    →CD = (d[0]-c[0], d[1]-c[1], d[2]-c[2])

    点Pは以下のようにおける
        (x, y, z) = (a[0]+s*(b[0]-a[0]), a[1]+s*(b[1]-a[1]), a[2]+s*(b[2]-a[2]))
    点Qは以下のようにおける
        (x, y, z) = (c[0]+t*(d[0]-c[0]), c[1]+t*(d[1]-c[1]), c[2]+t*(d[2]-c[2]))
    よって、
    →PQ = (c[0]+t*(d[0]-c[0])-a[0]-s*(b[0]-a[0]), c[1]+t*(d[1]-c[1])-a[1]-s*(b[1]-a[1]), c[2]+t*(d[2]-c[2])-a[2]-s*(b[2]-a[2]))

    ここから、→AB・→PQ = 0, →CD・→PQ = 0 の2式を解く

    →AB・→PQ
     = (b[0]-a[0])*(c[0]+t*(d[0]-c[0])-a[0]-s*(b[0]-a[0])) + (b[1]-a[1])*(c[1]+t*(d[1]-c[1])-a[1]-s*(b[1]-a[1])) + (b[2]-a[2])*(c[2]+t*(d[2]-c[2])-a[2]-s*(b[2]-a[2]))
     = -s*((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2) + t*((b[0]-a[0])*(d[0]-c[0])+(b[1]-a[1])*(d[1]-c[1])+(b[2]-a[2])*(d[2]-c[2])) + ((b[0]-a[0])*(c[0]-a[0])+(b[1]-a[1])*(c[1]-a[1])+(b[2]-a[2])*(c[2]-a[2]))
    →CD・→PQ
     = (d[0]-c[0])*(c[0]+t*(d[0]-c[0])-a[0]-s*(b[0]-a[0])) + (d[1]-c[1])*(c[1]+t*(d[1]-c[1])-a[1]-s*(b[1]-a[1])) + (d[2]-c[2])*(c[2]+t*(d[2]-c[2])-a[2]-s*(b[2]-a[2]))
     = -s*((b[0]-a[0])*(d[0]-c[0])+(b[1]-a[1])*(d[1]-c[1])+(b[2]-a[2])*(d[2]-c[2])) + t*((d[0]-c[0])**2+(d[1]-c[1])**2+(d[2]-c[2])**2) - ((d[0]-c[0])*(a[0]-c[0])+(d[1]-c[1])*(a[1]-c[1])+(d[2]-c[2])*(a[2]-c[2]))

    →AB・→PQ = 0, →CD・→PQ = 0 は s, t の連立方程式となる

    連立方程式を行列式によって表し、両辺に左から両辺に逆行列をかけることで解く
    """

    M00 = (b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2
    M01 = -((b[0]-a[0])*(d[0]-c[0])+(b[1]-a[1])*(d[1]-c[1])+(b[2]-a[2])*(d[2]-c[2]))
    M10 = (b[0]-a[0])*(d[0]-c[0])+(b[1]-a[1])*(d[1]-c[1])+(b[2]-a[2])*(d[2]-c[2])
    M11 = -((d[0]-c[0])**2+(d[1]-c[1])**2+(d[2]-c[2])**2)
    M = numpy.matrix([[M00, M01], [M10, M11]])

    n0 = (b[0]-a[0])*(c[0]-a[0])+(b[1]-a[1])*(c[1]-a[1])+(b[2]-a[2])*(c[2]-a[2])
    n1 = (d[0]-c[0])*(c[0]-a[0])+(d[1]-c[1])*(c[1]-a[1])+(d[2]-c[2])*(c[2]-a[2])
    n = numpy.matrix([[n0], [n1]])

    iM = None
    try:
        iM = numpy.linalg.inv(M)
    except numpy.linalg.LinAlgError:
        iM = numpy.linalg.inv(M + numpy.eye(2) * 0.01)

    st = iM@n
    s, t = st[0, 0], st[1, 0]

    return numpy.array([[a[0]+s*(b[0]-a[0]), a[1]+s*(b[1]-a[1]), a[2]+s*(b[2]-a[2])], [c[0]+t*(d[0]-c[0]), c[1]+t*(d[1]-c[1]), c[2]+t*(d[2]-c[2])]])

# 3次元座標系において点a,bを通る直線と点c,d,eからなる平面上の交点を導出する
# (ただし、交点を s*→cd + t*→ceとする時、s<0またはt<0の場合、Noneを返す)
def calculateLinePlaneIntersection(a, b, c, d, e):
    """
    →CD = (d[0]-c[0], d[1]-c[1], d[2]-c[2])
    →CE = (e[0]-c[0], e[1]-c[1], e[2]-c[2])

    点CDEを通る平面上の交点Pは以下のようにおける
        (x, y, z) = (c[0] + s*(d[0]-c[0]) + t*(e[0]-c[0]), c[1] + s*(d[1]-c[1]) + t*(e[1]-c[1]), c[2] + s*(d[2]-c[2] + t*(e[2]-c[2]))

    →AB上の点Qは以下のようにおける
        (x, y, z) = (u*(b[0]-a[0]) + a[0], u*(b[1]-a[1]) + a[1], u*(b[2]-a[2]) + a[2])

    よって
        s*(d[0]-c[0]) + t*(e[0]-c[0]]) - u*(b[0]-a[0]) = a[0] - c[0]
        s*(d[1]-c[1]) + t*(e[1]-c[1]) - u*(b[1]-a[1]) = a[1] - c[1]
        s*(d[2]-c[2]) + t*(e[2]-c[2]) - u*(b[2]-a[2]) = a[2] - c[2]

    連立方程式を行列式によって表し、両辺に左から両辺に逆行列をかけることで解く
    """
    M = numpy.matrix([[d[0]-c[0], e[0]-c[0], a[0]-b[0]], [d[1]-c[1], e[1]-c[1], a[1]-b[1]], [d[2]-c[2], e[2]-c[2], a[2]-b[2]]])
    n = numpy.matrix([[a[0] - c[0]], [a[1] - c[1]], [a[2] - c[2]]])
    iM = None

    try:
        iM = numpy.linalg.inv(M)
    except numpy.linalg.LinAlgError:
        iM = numpy.linalg.inv(M + numpy.eye(3) * 0.01)
    stu = iM@n

    s, t, u = stu[0, 0], stu[1, 0], stu[2, 0]
    if s < 0 or t < 0:
        return None
    else:
        return numpy.array([u*(b[0]-a[0]) + a[0] , u*(b[1]-a[1]) + a[1], u*(b[2]-a[2]) + a[2]])

def observeCapture(relay, isMain):
    capture = initCamera(env.mainCamera if isMain else env.subCamera)
    while not relay.stopped:
        time.sleep(0.001)
        _, frame = capture.read()
        unixtime = time.time()
        captureResult = CaptureResult(frame, unixtime)
        if isMain:
            relay.mainCaptureResult = captureResult
        else:
            relay.subCaptureResult = captureResult
    capture.release()

def observeMainCapture(relay):
    observeCapture(relay, True)

def observeSubCapture(relay):
    observeCapture(relay, False)

def observeLOS(relay, isMain):
    lastCaptureResultTime = 0.0
    intUnixtime, fps = 0, 0

    while not relay.stopped:
        time.sleep(0.001)

        # 画像を取得
        captureResult = relay.mainCaptureResult if isMain else relay.subCaptureResult
        if captureResult is None or not lastCaptureResultTime < captureResult.unixtime:
            continue
        lastCaptureResultTime = captureResult.unixtime

        # fps表示のための処理
        newIntUnixtime = int(time.time())
        if intUnixtime != newIntUnixtime:
            if isMain:
                relay.mainFps = fps
            else:
                relay.subFps = fps
            intUnixtime = newIntUnixtime
            fps = 0
        else:
            fps = fps + 1

        # 計算に使う情報の選択
        camera = env.mainCamera if isMain else env.subCamera

        # 計算量削減のため、リサイズした画像を使用する
        frame = cv2.resize(captureResult.frame , (env.width, env.height))
        unixtime = captureResult.unixtime

        # 特定HSV範囲で二値化した画像を取得
        _, _, resultFrame = convertFrame(frame, camera.hsvmin, camera.hsvmax, camera.medianBlurSize)

        # 重心座標（＝対象物の中心座標）を取得
        moments = cv2.moments(resultFrame, False)
        losResult = None
        if int(moments["m00"]) != 0:
            x, y = moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]
            # デバッグ用に、重心座標に円を描画
            cv2.circle(resultFrame, (int(x), int(y)), 5, 100, 2, 4)
            # キャリブレーション後の重心座標を計算する
            # 画像の状態でキャリブレーションを行わないのは計算量削減のため
            undistortedPoints = cv2.undistortPoints(numpy.array([[[x, y]]], dtype=numpy.float64), camera.mtx, camera.dist, P=camera.mtx)[0]
            ux, uy = undistortedPoints[0][0], undistortedPoints[0][1]

            topLeftPoint, topRightPoint, bottomRightPoint, bottomLeftPoint = camera.corners
            topLeft = numpy.array(topLeftPoint)
            topRight = numpy.array(topRightPoint)
            bottomLeft = numpy.array(bottomLeftPoint)

            # 定義した座標系の原点をO, カメラにちょうど収まって写る16:9の四角形の左上の点をA, 右上の点をB, 右下の点をC, 左下の点をDとする
            # 画像の重心座標を縦横それぞれ0.0 - 1.0 の比率でs,tと表した時、
            # 画像上の重心座標を16:9の四角形に対応させて得られる点Pは
            # →OP = (1-s)*→OA + s*→OB + (1-t)*→OA + t*→OD
            pointx = (topLeft * (1 - x / env.width) + topRight * x / env.width)
            pointy = (topLeft * (1 - y / env.height) + bottomLeft * y / env.height)
            point = pointx + pointy - topLeft

            losResult = LOSResult(point, unixtime, frame, resultFrame, pointx, pointy)
        else:
            losResult = LOSResult(None, unixtime, frame, resultFrame, None, None)

        relay.lock.acquire()
        if isMain:
            relay.mainLOSResultDequeue.append(losResult)
        else:
            relay.subLOSResultDequeue.append(losResult)
        relay.lock.release()

def observeMainLOS(relay):
    observeLOS(relay, True)

def observeSubLOS(relay):
    observeLOS(relay, False)

def observeVelocity(relay):

    successTargetResultDequeue = deque([], env.targetResultDequeueSize)

    while not relay.stopped:

        time.sleep(0.00333)

        relay.mainQueSize = len(relay.mainLOSResultDequeue)
        relay.subQueSize = len(relay.subLOSResultDequeue)

        # キューに積んである処理結果が main, sub で偏りが出てきた場合、多い方を切り捨てる
        if relay.mainQueSize < env.losResultDequeueSize:
            if relay.subQueSize > env.losResultDequeueSize:
                relay.subLOSResultDequeue.popleft()
            continue
        if relay.subQueSize < env.losResultDequeueSize:
            if relay.mainQueSize > env.losResultDequeueSize:
                relay.mainLOSResultDequeue.popleft()
            continue

        relay.lock.acquire()
        mainLOSResultList = list(itertools.islice(relay.mainLOSResultDequeue, 0, env.losResultDequeueSize))
        subLOSResultList = list(itertools.islice(relay.subLOSResultDequeue, 0, env.losResultDequeueSize))
        relay.lock.release()

        targetResult = None
        mainLOSResult = mainLOSResultList[int(env.losResultDequeueSize / 2)]
        if mainLOSResult.point is not None:
            # 2直線(subPrevLOS, subNextLOS)を通る平面と直線(mainLOS)の交点を対象物の空間座標とする
            # main と sub で LOS の計測にズレが生じるため、main を基準とし、sub は連続して取得した2つの直線を使った平面で考えている
            for i in range(env.losResultDequeueSize - 1):
                subPrevLOSResult = subLOSResultList[i]
                if subPrevLOSResult.point is None:
                    continue
                subNextLOSResult = subLOSResultList[i + 1]
                if subNextLOSResult.point is None:
                    continue
                target = calculateLinePlaneIntersection(env.mainCamera.position, mainLOSResult.point, env.subCamera.position, subPrevLOSResult.point, subNextLOSResult.point)
                if target is not None:
                    targetResult = TargetResult(mainLOSResult, subLOSResultList, i, i + 1, None, None, target)
            # 前述の交点が存在しない場合、変わりに共通垂線の中点があれば、それを空間座標とする
            # 対象物が静止している場合などはこちらで位置を導出する
            # (基本こちらで計測することはないが、デバッグ時に便利なので残してある)
            if targetResult is None:
                # 2直線(mainLOS, subLOS)の最短距離が10cm以上である場合は誤差が大きいとして無視
                minNorm = env.commonPerpendicularMinNorm
                for i in range(env.losResultDequeueSize):
                    subLOSResult = subLOSResultList[i]
                    if subLOSResult.point is None:
                        continue
                    commonPerpendicular = calculateCommonPerpendicular(env.mainCamera.position, mainLOSResult.point, env.subCamera.position, subLOSResult.point)
                    norm = numpy.linalg.norm(commonPerpendicular[0]-commonPerpendicular[1])
                    if minNorm < norm:
                        continue
                    minNorm = norm
                    target = (commonPerpendicular[0] + commonPerpendicular[1]) / 2
                    targetResult = TargetResult(mainLOSResult, subLOSResultList, None, None, i, commonPerpendicular, target)

        if targetResult is None:
            targetResult = TargetResult(mainLOSResult, subLOSResultList, None, None, None, None, None)

        relay.targetResult = targetResult

        if relay.debugTargetResultList is not None:
            print("index: {}".format(len(relay.debugTargetResultList)))
            relay.debugTargetResultList.append(targetResult)

        if targetResult.target is not None:
            successTargetResultDequeue.append(targetResult)
            latestTargetResult = successTargetResultDequeue[-1]
            velocity, timeInterval, distance, frames = None, None, None, None

            # 連続して取得した targetResult のリストから、velocity_caluculate_max_sec を超えない期間で最も古いものと最新のものを比較し、
            # その距離と時間の差分から速度を計算する
            for i in range(len(successTargetResultDequeue) - env.velocityCaluculateMinNum, -1, -1):
                prevTargetResult = successTargetResultDequeue[i]
                _timeInterval = latestTargetResult.mainLOSResult.unixtime - prevTargetResult.mainLOSResult.unixtime
                if _timeInterval > env.velocityCaluculateMaxSec:
                    break
                timeInterval = _timeInterval
                distance = numpy.linalg.norm(latestTargetResult.target - prevTargetResult.target)
                velocity = (distance / 1000) / (timeInterval / 3600)
                frames = len(successTargetResultDequeue) - i

            if velocity is not None:
                print("{:.2f}km/h".format(velocity))
                if velocity > env.velocityMin:
                    relay.velocity = velocity
                    relay.timeInterval = timeInterval
                    relay.distance = distance
                    relay.frames = frames

        relay.mainLOSResultDequeue.popleft()
        relay.subLOSResultDequeue.popleft()

def main():
    args = getOption()
    showGraph = False
    saveDebugData = False

    if args.calibrateMainCamera:
        calibrate(env.mainCamera)
        exit()
    if args.calibrateSubCamera:
        calibrate(env.subCamera)
        exit()
    if args.previewMainCamera:
        previewPoints(env.mainCamera)
        exit()
    if args.previewSubCamera:
        previewPoints(env.subCamera)
        exit()
    if args.adjustMainCamera:
        adjust(env.mainCamera)
        exit()
    if args.adjustSubCamera:
        adjust(env.subCamera)
        exit()
    if args.showGraph:
        showGraph = True
    if args.saveDebugData:
        saveDebugData = True
    if args.loadDebugData is not None:
        debug.loadDebugData(Drawer(), args.loadDebugData)
        exit()

    relay = Relay(showGraph, saveDebugData)
    drawer = Drawer() if showGraph else None

    mainCapture = threading.Thread(target=observeMainCapture, args=([relay]))
    subCapture = threading.Thread(target=observeSubCapture, args=([relay]))
    mainLOS = threading.Thread(target=observeMainLOS, args=([relay]))
    subLOS = threading.Thread(target=observeSubLOS, args=([relay]))
    velocity = threading.Thread(target=observeVelocity, args=([relay]))

    mainCapture.start()
    subCapture.start()
    mainLOS.start()
    subLOS.start()
    velocity.start()

    while True:
        screen = numpy.zeros((405,720,3), numpy.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 球速を表示
        velocityStr = "{:3d}km/h".format(int(relay.velocity)) if relay.velocity is not None else "  -km/h"
        cv2.putText(screen, velocityStr, (80,200), font, 4, (255, 255, 255), 6, cv2.LINE_AA)
        # カメラ映像の計測に利用された時間, 距離, フレーム数を表示
        timeIntervalStr = "{:1.3f}".format(relay.timeInterval) if relay.timeInterval is not None else "   -"
        distanceStr = "{:1.2f}".format(relay.distance) if relay.distance is not None else "  -"
        framesStr = "{:2d}".format(int(relay.frames)) if relay.frames is not None else " -"
        infoStr = "{}sec {}m {}frames".format(timeIntervalStr, distanceStr, framesStr)
        cv2.putText(screen, infoStr, (30,320), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        subInfoStr = "main({:3d}fps{:3d}que) sub({:3d}fps{:3d}que)".format(relay.mainFps, relay.mainQueSize, relay.subFps, relay.subQueSize)
        cv2.putText(screen, subInfoStr, (30,360), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Screen", screen)
        k = cv2.waitKey(100)
        if k >= 0:
            relay.stopped = True
            break

        if drawer is not None:
            if not drawer.isActive():
                relay.stopped = True
                break
            if relay.targetResult is not None:
                drawer.clearLines()
                drawer.drawTargetResult(relay.targetResult)
                drawer.pause()

    if saveDebugData and len(relay.debugTargetResultList) > 0:
        startIndex = input("input debug data start index (min: 0): ")
        endIndex = input("input debug data end index (max: {}): ".format(len(relay.debugTargetResultList)))
        print("\ndebug data saving...")
        for index in range(int(startIndex), int(endIndex) + 1):
            debug.saveTargetResult(index, relay.debugTargetResultList[index])

    if showGraph:
        drawer.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
