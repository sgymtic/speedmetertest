from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from matplotlib import _pylab_helpers
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d.art3d as art3d
from argparse import ArgumentParser
import cv2
import numpy
import environment
from environment import Environment
import result
from result import LOSResult
from result import TargetResult
from draw import Drawer

PATH = "debug"

env = environment.load()

# 最初まだプログラムの方針が定まってない頃に作ったデバッグ用の処理を、そのまま騙し騙し修正して使ってる
# 色々と使いづらいが、今更直すのも面倒なので、放置…

def loadDebugData(drawer, index):

    class Index(object):
        i = index
        di = None

        def next(self, event):
            self.i += 1
            self.drawTargetResult()
        def prev(self, event):
            self.i -= 1
            self.drawTargetResult()
        def dnext(self, event):
            if self.di is None:
                self.di = 0
            else:
                self.di += 1
            self.drawTargetResult()
        def dprev(self, event):
            if self.di is None:
                self.di = 0
            else:
                self.di -= 1
            if self.di < 0:
                self.di = None
            self.drawTargetResult()

        def drawTargetResult(self):
            drawer.clearLines()
            targetResult = loadTargetResult(self.i)
            print("curent index: {}, time: {}, target: {}".format(self.i, targetResult.mainLOSResult.unixtime, targetResult.target))
            drawer.drawTargetResult(targetResult, self.di)
            if targetResult.mainLOSResult is not None:
                cv2.imshow('main frame', cv2.undistort(targetResult.mainLOSResult.frame, env.mainCamera.mtx, env.mainCamera.dist, None))
                cv2.imshow('main original frame', cv2.undistort(targetResult.mainLOSResult.originalFrame, env.mainCamera.mtx, env.mainCamera.dist, None))
            else:
                cv2.imshow('main frame', imageWhite)
                cv2.imshow('main original frame', imageWhite)
            if targetResult.subPrevLOSResultIndex is not None:
                subPrevLOSResult = targetResult.subLOSResultList[targetResult.subPrevLOSResultIndex]
                cv2.imshow('subPrev frame', cv2.undistort(subPrevLOSResult.frame, env.subCamera.mtx, env.subCamera.dist, None))
                cv2.imshow('subPrev original frame', cv2.undistort(subPrevLOSResult.originalFrame, env.subCamera.mtx, env.subCamera.dist, None))
            else:
                cv2.imshow('subPrev frame', imageWhite)
                cv2.imshow('subPrev original frame', imageWhite)
            if targetResult.subNextLOSResultIndex is not None:
                subNextLOSResult = targetResult.subLOSResultList[targetResult.subNextLOSResultIndex]
                cv2.imshow('subNext frame', cv2.undistort(subNextLOSResult.frame, env.subCamera.mtx, env.subCamera.dist, None))
                cv2.imshow('subNext original frame', cv2.undistort(subNextLOSResult.originalFrame, env.subCamera.mtx, env.subCamera.dist, None))
            else:
                cv2.imshow('subNext frame', imageWhite)
                cv2.imshow('subNext original frame', imageWhite)
            if targetResult.subLOSResultIndex is not None:
                subLOSResult = targetResult.subLOSResultList[targetResult.subLOSResultIndex]
                cv2.imshow('sub frame', cv2.undistort(subLOSResult.frame, env.subCamera.mtx, env.subCamera.dist, None))
                cv2.imshow('sub original frame', cv2.undistort(subLOSResult.originalFrame, env.subCamera.mtx, env.subCamera.dist, None))
            else:
                cv2.imshow('sub frame', imageWhite)
                cv2.imshow('sub original frame', imageWhite)
            if self.di is not None and self.di < len(targetResult.subLOSResultList):
                subLOSResult = targetResult.subLOSResultList[self.di]
                cv2.imshow('sub di frame', cv2.undistort(subLOSResult.frame, env.subCamera.mtx, env.subCamera.dist, None))
                cv2.imshow('sub di original frame', cv2.undistort(subLOSResult.originalFrame, env.subCamera.mtx, env.subCamera.dist, None))
            pyplot.draw()


    imageWhite = numpy.ones((env.height, env.width, 3),numpy.uint8)*0

    cv2.imshow('main frame', imageWhite)
    cv2.moveWindow('main frame', 0, 0)
    cv2.imshow('main original frame', imageWhite)
    cv2.moveWindow('main original frame', 0, env.height)
    cv2.imshow('subPrev frame', imageWhite)
    cv2.moveWindow('subPrev frame', env.width, 0)
    cv2.imshow('subPrev original frame', imageWhite)
    cv2.moveWindow('subPrev original frame', env.width, env.height)
    cv2.imshow('subNext frame', imageWhite)
    cv2.moveWindow('subNext frame', env.width * 2, 0)
    cv2.imshow('subNext original frame', imageWhite)
    cv2.moveWindow('subNext original frame', env.width * 2, env.height)
    cv2.imshow('sub frame', imageWhite)
    cv2.moveWindow('sub frame', env.width * 3, 0)
    cv2.imshow('sub original frame', imageWhite)
    cv2.moveWindow('sub original frame', env.width * 3, env.height)

    callback = Index()
    axprev = pyplot.axes([0.7, 0.05, 0.1, 0.030])
    axnext = pyplot.axes([0.81, 0.05, 0.1, 0.030])
    axdprev = pyplot.axes([0.7, 0.09, 0.1, 0.030])
    axdnext = pyplot.axes([0.81, 0.09, 0.1, 0.030])
    bnext = Button(axnext, '>')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, '<')
    bprev.on_clicked(callback.prev)
    bdnext = Button(axdnext, '(>)')
    bdnext.on_clicked(callback.dnext)
    bdprev = Button(axdprev, '(<)')
    bdprev.on_clicked(callback.dprev)
    callback.drawTargetResult()
    pyplot.show()
    cv2.destroyAllWindows()
    pyplot.close()

def losResultUnixtimeFileName(index, id):
    return PATH + "/losResult-unixtime-{}-{}".format(index, id)

def losResultArrayFileName(index, id):
    return PATH + "/losResult-array-{}-{}".format(index, id)

def losResultOriginalFrameFileName(index, id):
    return PATH + "/losResult-originalFrame-{}-{}".format(index, id)

def losResultFrameFileName(index, id):
    return PATH + "/losResult-frame-{}-{}".format(index, id)

def loadLOSResult(index, id):
    point = None
    pointx = None
    pointy = None
    unixtime = None

    try:
        unixtime = numpy.load(losResultUnixtimeFileName(index, id) + ".npy")
    except FileNotFoundError:
        return None

    try:
        array = numpy.load(losResultArrayFileName(index, id) + ".npy")
        point = array[0]
        pointx = array[1]
        pointy = array[2]
    except FileNotFoundError:
        pass

    originalFrame = cv2.imread(losResultOriginalFrameFileName(index, id) + ".png")
    frame = cv2.imread(losResultFrameFileName(index, id) + ".png")

    return LOSResult(point, unixtime, originalFrame, frame, pointx, pointy)

def saveLOSResult(index, id, losResult):
    if losResult.point is not None:
        array = numpy.array([losResult.point, losResult.pointx, losResult.pointy])
        numpy.save(losResultArrayFileName(index, id), array)

    numpy.save(losResultUnixtimeFileName(index, id), losResult.unixtime)

    if losResult.originalFrame is not None:
        cv2.imwrite(losResultOriginalFrameFileName(index, id) + ".png", losResult.originalFrame)

    if losResult.frame is not None:
        cv2.imwrite(losResultFrameFileName(index, id) + ".png", losResult.frame)

def subLOSResultIndexFileName(index, id):
    return PATH + "/result-subLOSResultIndex-{}-{}".format(index, id)

def loadSubLOSResultIndex(index, id):
    try:
        subLOSIndex = numpy.load(subLOSResultIndexFileName(index, id) + ".npy")
        return subLOSIndex
    except FileNotFoundError:
        return None

def saveSubLOSResultIndex(index, id, subLOSResultIndex):
    numpy.save(subLOSResultIndexFileName(index, id), subLOSResultIndex)

def commonPerpendicularFileName(index, id1, id2):
    return PATH + "/result-commonPerpendicular-{}-{}-{}".format(index, id1, id2)

def loadCommonPerpendicular(index, id1, id2):
    try:
        commonPerpendicular = numpy.load(commonPerpendicularFileName(index, id1, id2) + ".npy")
        return commonPerpendicular
    except FileNotFoundError:
        return None

def saveCommonPerpendicular(index, id1, id2, commonPerpendicular):
    numpy.save(commonPerpendicularFileName(index, id1, id2), commonPerpendicular)

def targetFileName(index):
    return PATH + "/result-target-{}".format(index)

def loadTarget(index):
    try:
        target = numpy.load(targetFileName(index) + ".npy")
        return target
    except FileNotFoundError:
        return None

def saveTarget(index, target):
    numpy.save(targetFileName(index), target)

def loadTargetResult(index):
    mainLOSResult = loadLOSResult(index, 0)
    i = 1
    subLOSResultList = []
    while True:
        subLOSResult = loadLOSResult(index, i)
        if subLOSResult is not None:
            subLOSResultList.append(subLOSResult)
            i += 1
        else:
            break
    subPrevLOSResultIndex = loadSubLOSResultIndex(index, 0)
    subNextLOSResultIndex = loadSubLOSResultIndex(index, 1)
    subLOSResultIndex = loadSubLOSResultIndex(index, 2)
    commonPerpendicular = loadCommonPerpendicular(index, 0, 1)
    target = loadTarget(index)
    return TargetResult(mainLOSResult, subLOSResultList, subPrevLOSResultIndex, subNextLOSResultIndex, subLOSResultIndex, commonPerpendicular, target)

def saveTargetResult(index, result):
    if result.mainLOSResult is not None:
        saveLOSResult(index, 0, result.mainLOSResult)
    for i in range(len(result.subLOSResultList)):
        saveLOSResult(index, i + 1, result.subLOSResultList[i])

    if result.subPrevLOSResultIndex is not None:
        saveSubLOSResultIndex(index, 0, result.subPrevLOSResultIndex)
    if result.subNextLOSResultIndex is not None:
        saveSubLOSResultIndex(index, 1, result.subNextLOSResultIndex)
    if result.subLOSResultIndex is not None:
        saveSubLOSResultIndex(index, 2, result.subLOSResultIndex)

    if result.commonPerpendicular is not None:
        saveCommonPerpendicular(index, 0, 1, result.commonPerpendicular)
    if result.target is not None:
        saveTarget(index, result.target)
