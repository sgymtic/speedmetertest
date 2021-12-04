from mpl_toolkits.mplot3d import Axes3D
from matplotlib import _pylab_helpers
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d.art3d as art3d
import numpy
import environment

env = environment.load()

# matplotlib の mplot3d を使って各種計算結果・過程を表示する
# デバッグ用
class Drawer:

    figure = None
    ax = None

    def __init__(self):
        self.figure = pyplot.figure()
        self.ax = Axes3D(self.figure)

    def close(self):
        pyplot.close()

    def clearLines(self):
        self.ax.lines = []

    def isActive(self):
        manager = _pylab_helpers.Gcf.get_active()
        return manager is not None

    def pause(self):
        pyplot.pause(0.01)

    def drawTargetResult(self, targetResult, debugSubLOSIndex = None):
        self.setLabels()

        if env.mainCamera is not None:
            if env.mainCamera.position is not None:
                self.drawCamera(env.mainCamera.position)
            if env.mainCamera.position is not None and env.mainCamera.corners is not None:
                self.drawPerspective(env.mainCamera.position, env.mainCamera.corners)

        if env.subCamera is not None:
            if env.subCamera.position is not None:
                self.drawCamera(env.subCamera.position)
            if env.subCamera.position is not None and env.subCamera.corners is not None:
                self.drawPerspective(env.subCamera.position, env.subCamera.corners)

            if targetResult.mainLOSResult is not None and targetResult.mainLOSResult.point is not None:
                self.drawImageAxis(env.mainCamera.position, env.mainCamera.corners, targetResult.mainLOSResult.point, targetResult.mainLOSResult.pointx, targetResult.mainLOSResult.pointy)
                self.drawLineOfSight(env.mainCamera.position, targetResult.mainLOSResult.point)

            if targetResult.subPrevLOSResultIndex is not None:
                subPrevLOSResult = targetResult.subLOSResultList[targetResult.subPrevLOSResultIndex]
                if subPrevLOSResult.point is not None:
                    self.drawImageAxis(env.subCamera.position, env.subCamera.corners, subPrevLOSResult.point, subPrevLOSResult.pointx, subPrevLOSResult.pointy)
                    self.drawLineOfSight(env.subCamera.position, subPrevLOSResult.point)
            if targetResult.subNextLOSResultIndex is not None:
                subNextLOSResult = targetResult.subLOSResultList[targetResult.subNextLOSResultIndex]
                if subNextLOSResult.point is not None:
                    self.drawImageAxis(env.subCamera.position, env.subCamera.corners, subNextLOSResult.point, subNextLOSResult.pointx, subNextLOSResult.pointy)
                    self.drawLineOfSight(env.subCamera.position, subNextLOSResult.point)

            if targetResult.subLOSResultIndex is not None:
                subLOSResult = targetResult.subLOSResultList[targetResult.subLOSResultIndex]
                if subLOSResult.point is not None:
                    self.drawImageAxis(env.subCamera.position, env.subCamera.corners, subLOSResult.point, subLOSResult.pointx, subLOSResult.pointy)
                    self.drawLineOfSight(env.subCamera.position, subLOSResult.point)
            if targetResult.commonPerpendicular is not None:
                self.drawCommonPerpendicular(targetResult.commonPerpendicular[0], targetResult.commonPerpendicular[1])

            if targetResult.target is not None:
                self.drawTarget(targetResult.target)

            if debugSubLOSIndex is not None:
                debugSubLOSResult = targetResult.subLOSResultList[debugSubLOSIndex]
                if debugSubLOSResult.point is not None:
                    self.drawImageAxis(env.subCamera.position, env.subCamera.corners, debugSubLOSResult.point, debugSubLOSResult.pointx, debugSubLOSResult.pointy, color='k')
                    self.drawLineOfSight(env.subCamera.position, debugSubLOSResult.point, color='k')

    def setLabels(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim([env.xmin, env.xmax])
        self.ax.set_ylim([env.ymin, env.ymax])
        self.ax.set_zlim([env.zmin, env.zmax])

    def drawCamera(self, camera):
        self.ax.plot(camera[0], camera[1], camera[2], marker="o",linestyle='None',color='b')

    def drawPerspective(self, camera, corners):
        topLeft, topRight, bottomRight, bottomLeft = corners

        camera2TopLeft = art3d.Line3D([camera[0], topLeft[0]], [camera[1], topLeft[1]], [camera[2] ,topLeft[2]], color='k', linestyle=':')
        self.ax.add_line(camera2TopLeft)
        camera2BottomLeft = art3d.Line3D([camera[0], bottomLeft[0]], [camera[1], bottomLeft[1]], [camera[2] ,bottomLeft[2]], color='k', linestyle=':')
        self.ax.add_line(camera2BottomLeft)
        camera2BottomRight = art3d.Line3D([camera[0], bottomRight[0]], [camera[1], bottomRight[1]], [camera[2] ,bottomRight[2]], color='k', linestyle=':')
        self.ax.add_line(camera2BottomRight)
        camera2TopRight = art3d.Line3D([camera[0], topRight[0]], [camera[1], topRight[1]], [camera[2], topRight[2]], color='k', linestyle=':')
        self.ax.add_line(camera2TopRight)

        frame = art3d.Line3D([topLeft[0], topRight[0], bottomRight[0], bottomLeft[0], topLeft[0]], [topLeft[1], topRight[1], bottomRight[1], bottomLeft[1], topLeft[1]], [topLeft[2], topRight[2], bottomRight[2], bottomLeft[2], topLeft[2]], color='k', linestyle=':')
        self.ax.add_line(frame)

    def drawImageAxis(self, camera, corners, point, pointx, pointy, color='r'):
        topLeft, topRight, bottomRight, bottomLeft = corners
        self.ax.plot(point[0], point[1], point[2], marker="o",linestyle='None', color=color)
        topLeft2X = art3d.Line3D([topLeft[0], pointx[0]], [topLeft[1], pointx[1]], [topLeft[2] ,pointx[2]], color='r', linestyle='-')
        topLeft2Y = art3d.Line3D([topLeft[0], pointy[0]], [topLeft[1], pointy[1]], [topLeft[2] ,pointy[2]], color='r', linestyle='-')
        point2X = art3d.Line3D([point[0], pointx[0]], [point[1], pointx[1]], [point[2] ,pointx[2]], color='r', linestyle=':')
        point2Y = art3d.Line3D([point[0], pointy[0]], [point[1], pointy[1]], [point[2] ,pointy[2]], color='r', linestyle=':')
        self.ax.add_line(topLeft2X)
        self.ax.add_line(topLeft2Y)
        self.ax.add_line(point2X)
        self.ax.add_line(point2Y)

    def drawLineOfSight(self, camera, point, color='y'):
        point2Camera = art3d.Line3D([point[0], camera[0]], [point[1], camera[1]], [point[2] ,camera[2]], color=color, linestyle='-')
        self.ax.add_line(point2Camera)

    def drawCommonPerpendicular(self, point1, point2):
        point2Camera = art3d.Line3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2] ,point2[2]], color='g', linestyle='-')
        self.ax.add_line(point2Camera)

    def drawTarget(self, point):
        self.ax.plot(point[0], point[1], point[2], marker="o",linestyle='None',color='g')
