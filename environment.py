import configparser
import json
import numpy

class Environment:
    mainCamera = None
    subCamera = None

    width = None
    height = None
    losResultDequeueSize = None
    targetResultDequeueSize = None
    velocityCaluculateMaxSec = None
    velocityCaluculateMinNum = None
    velocityMin = None
    commonPerpendicularMinNorm = None
    calibration = None

    xmin = None
    xmax = None
    ymin = None
    ymax = None
    zmin = None
    zmax = None

class Calibration:
    squareSize = None
    patternWidth = None
    patternHeight = None
    referenceImageNum = None

class Camera:
    id = None
    width = None
    height = None
    fps = None
    position = None
    corners = None
    mtx = None
    dist = None
    hsvmin = None
    hsvmax = None
    medianBlurSize = None

# n軸θ回転の行列R (ロドリゲスの回転公式)
def R(n,th):
    n = n.reshape([3,1])
    # Rn(θ) = Icosθ + n^sinθ + nn'(1-cosθ)
    return numpy.cos(th)*numpy.eye(3) + numpy.sin(th)*vecamera2skew(n) + (1-numpy.cos(th))*n@n.T

# v∈R^3-->v_× (外積作用の行列)
def vecamera2skew(v):
    v = v.reshape([3,])
    return numpy.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])

# 「カメラにちょうど収まって写る4:3の四角形」の四隅の座標を計算する
# カメラで撮影した（キャリブレーション後の）中央に写る任意の場所の座標と、左上に写る任意の場所の座標を利用する
def getCorners(position, centerPoint, topLeftPoint, width, height):

    if position is None or centerPoint is None or topLeftPoint is None:
        return None

    x, y, z = position
    camera = numpy.array([x, y, z]).reshape([3,1])
    x, y, z = centerPoint
    center = numpy.array([x, y, z]).reshape([3,1])
    x, y, z = topLeftPoint

    # 「左上に写る任意の場所の座標」はそのまま四隅の一つとして使用する
    topLeft = numpy.array([x, y, z]).reshape([3,1])

    # カメラと、カメラで撮影した（キャリブレーション後の）中央に写る任意の場所を結ぶ直線を軸に
    # 「4:3の四角形」の左上の点を回転することで、右上、右下、左下の座標を求めていく

    v = center - camera # カメラ映像の中央を表す直線が原点(0,0,0)を通る直線になるように平行移動
    o = topLeft - camera # カメラ映像の中央を表す直線も平行移動
    n = v / numpy.linalg.norm(v) # 単位行列化
    topRight = (R(n, numpy.arctan(width/height)*2) @ o + camera).reshape(1,3)[0]
    bottomRight = (R(n, numpy.pi) @ o + camera).reshape(1,3)[0]
    bottomLeft = (R(n, numpy.pi + numpy.arctan(width/height)*2) @ o + camera).reshape(1,3)[0]

    topRightPoint = (topRight[0], topRight[1], topRight[2])
    bottomRightPoint = (bottomRight[0], bottomRight[1], bottomRight[2])
    bottomLeftPoint = (bottomLeft[0], bottomLeft[1], bottomLeft[2])

    return [topLeftPoint, topRight, bottomRight, bottomLeft]

def loadCamera(ini, section):

    camera = Camera()

    camera.id = ini.getint(section, "id")
    camera.width = ini.getint(section, "width")
    camera.height = ini.getint(section, "height")
    camera.fps = ini.getint(section, "fps")

    position = (ini.getfloat(section, "position_x"), ini.getfloat(section, "position_y"), ini.getfloat(section, "position_z"))
    topLeftPoint = (ini.getfloat(section, "lefttop_x"), ini.getfloat(section, "lefttop_y"), ini.getfloat(section, "lefttop_z"))
    centerPoint = (ini.getfloat(section, "center_x"), ini.getfloat(section, "center_y"), ini.getfloat(section, "center_z"))

    camera.position = position
    camera.corners = getCorners(position, centerPoint, topLeftPoint, camera.width, camera.height)

    camera.mtx = numpy.mat([
    [ini.getfloat(section,"mtx_fx"), 0.0, ini.getfloat(section,"mtx_cx")],
    [0.0, ini.getfloat(section,"mtx_fy"), ini.getfloat(section,"mtx_cy")],
    [0.0, 0.0, 1.0]
    ])

    camera.dist = numpy.array([
    ini.getfloat(section,"dist_k1"),
    ini.getfloat(section,"dist_k2"),
    ini.getfloat(section,"dist_p1"),
    ini.getfloat(section,"dist_p2"),
    ini.getfloat(section,"dist_k3")
    ])

    camera.hsvmin = (ini.getint(section,"hue_min"), ini.getint(section,"saturation_min"), ini.getint(section,"value_min"))
    camera.hsvmax = (ini.getint(section,"hue_max"), ini.getint(section,"saturation_max"), ini.getint(section,"value_max"))

    camera.medianBlurSize = ini.getint(section,"median_blur_size")

    return camera

def loadCalibration(ini):
    calibration = Calibration()

    section = "calibration"

    calibration.squareSize = ini.getfloat(section,"square_size")
    calibration.patternWidth = ini.getint(section,"pattern_width")
    calibration.patternHeight = ini.getint(section,"pattern_height")
    calibration.referenceImageNum = ini.getint(section,"reference_image_num")

    return calibration

def _load(ini):

    env = Environment()

    section = "common"

    env.mainCamera = loadCamera(ini, "main")
    env.subCamera = loadCamera(ini, "sub")
    env.width = ini.getint(section,"width")
    env.height = ini.getint(section,"height")
    env.losResultDequeueSize = ini.getint(section,"los_result_dequeue_size")
    env.targetResultDequeueSize = ini.getint(section,"target_result_dequeue_size")
    env.velocityCaluculateMaxSec = ini.getfloat(section,"velocity_caluculate_max_sec")
    env.velocityCaluculateMinNum = ini.getint(section,"velocity_caluculate_min_num")
    env.velocityMin = ini.getfloat(section,"velocity_min")
    env.commonPerpendicularMinNorm = ini.getfloat(section,"common_perpendicular_min_norm")
    env.calibration = loadCalibration(ini)

    env.xmin = ini.getfloat(section,"xmin")
    env.xmax = ini.getfloat(section,"xmax")
    env.ymin = ini.getfloat(section,"ymin")
    env.ymax = ini.getfloat(section,"ymax")
    env.zmin = ini.getfloat(section,"zmin")
    env.zmax = ini.getfloat(section,"zmax")

    return env

def load():

    ini = configparser.ConfigParser()
    ini.read("environment.ini", encoding='utf-8')

    return _load(ini)
