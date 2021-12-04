import numpy
import cv2
import time

# キャプチャの取得結果　
class CaptureResult:
    frame = None
    unixtime = None

    def __init__(self, frame = None, unixtime = None):
        self.frame = frame
        self.unixtime = unixtime

# カメラと対象物を結ぶ直線（対象物が存在し得る直線）の計算結果
class LOSResult:
    point = None
    unixtime = None

    ## デバッグ用
    originalFrame = None
    frame = None
    pointx = None
    pointy = None

    def __init__(self, point = None, unixtime = None, originalFrame = None, frame = None, pointx = None, pointy = None):
        self.point = point
        self.unixtime = unixtime
        self.originalFrame = originalFrame
        self.frame = frame
        self.pointx = pointx
        self.pointy = pointy

# 対象物の空間座標における位置の計算結果
class TargetResult:
    mainLOSResult = None
    subLOSResultList = None
    subPrevLOSResultIndex = None
    subNextLOSResultIndex = None
    subLOSResultIndex = None
    commonPerpendicular = None
    target = None

    def __init__(self, mainLOSResult = None, subLOSResultList = None, subPrevLOSResultIndex = None, subNextLOSResultIndex = None, subLOSResultIndex = None, commonPerpendicular = None, target = None):
        self.mainLOSResult = mainLOSResult
        self.subLOSResultList = subLOSResultList
        self.subPrevLOSResultIndex = subPrevLOSResultIndex
        self.subNextLOSResultIndex = subNextLOSResultIndex
        self.subLOSResultIndex = subLOSResultIndex
        self.commonPerpendicular = commonPerpendicular
        self.target = target
