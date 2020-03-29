import sys, os, math, json

from PySide2 import QtCore, QtWidgets, QtGui
from objects import Box, Sphere, Camera, rot_quat, load_objs
import numpy as np
from enum import Enum

# By what factor should the render output be scaled up?
SCALE_FACTOR = 1
# How far does the camera translate per repeated button press?
TRANSLATION_STEP = 0.1
# How far (in radians) does the camera rotate per repeated button press?
ROTATION_STEP = math.pi / 30
# Outline color used to highlight currently selected rotation pivot in the scene
PIVOT_COLOR = [255, 180, 100]
# Outline color used to highlight object currently selected by the inspector
SELECT_COLOR = [255, 255, 180]

# Types of rendering onto the raster surface
class RasterMode(Enum):
    FRAME = 0,
    RAYTRACE = 1

# Types of file operations
class FileOperation(Enum):
    SAVE = 0,
    LOAD = 1

# Rendering component
class CubeTeaRasterWidget(QtWidgets.QWidget):
    def __init__(self, objs, camera, parent=None):
        super().__init__(parent)
        self.objs = objs
        self.camera = camera
        self.resize(SCALE_FACTOR * self.camera.vdims[0], SCALE_FACTOR * self.camera.vdims[1])
        self.show()
        self.repaint = True
        self.cache = []
        self.mode = RasterMode.FRAME
        self.selectIdx = -1
        self.pivotIdx = -1

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)

        if (self.mode == RasterMode.FRAME):
            frameItems = self.camera.frame_rasterize(self.objs)
            painter.setBrush(QtGui.QColor(*self.camera.color))
            painter.fillRect(QtCore.QRectF(0, 0, SCALE_FACTOR * self.camera.vdims[0],
                                                 SCALE_FACTOR * self.camera.vdims[1]), painter.brush())
            painter.setBrush(QtGui.QColor(0, 0, 0, 0))
            for item in frameItems:
                framePen = painter.pen()
                framePen.setWidth(5.0)
                framePen.setStyle(QtCore.Qt.DashDotDotLine)
                framePen.setColor(QtGui.QColor(*item[2]))
                painter.setPen(framePen)
                if item[3] == "Line":
                    painter.drawLine(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * QtCore.QPointF(*item[1]))
                elif item[3] == "Circle":
                    painter.drawEllipse(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * item[1],
                                        SCALE_FACTOR * item[1])
        elif (self.mode == RasterMode.RAYTRACE):
            if self.repaint:
                raster = np.transpose(self.camera.raytrace(self.objs, False), (1, 0, 2)).copy()
                raster8 = raster.astype(np.uint8, order='C', casting='unsafe')
                image = QtGui.QImage(raster8.data, raster8.shape[1], raster8.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap(image).scaled(
                    SCALE_FACTOR * self.camera.vdims[0],
                    SCALE_FACTOR * self.camera.vdims[1],
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation)
                painter.drawPixmap(QtCore.QPointF(0, 0), pixmap,
                                   QtCore.QRectF(0, 0, SCALE_FACTOR * self.camera.vdims[0],
                                                 SCALE_FACTOR * self.camera.vdims[1]))
                self.cache = pixmap
                self.repaint = False
            else:
                painter.drawPixmap(QtCore.QPointF(0, 0), self.cache,
                                   QtCore.QRectF(0, 0, SCALE_FACTOR * self.camera.vdims[0],
                                                 SCALE_FACTOR * self.camera.vdims[1]))
        if self.pivotIdx != -1:
            painter.setBrush(QtGui.QColor(0, 0, 0, 0))
            pivot = self.objs[self.pivotIdx]
            items = self.camera.frame_rasterize([pivot])
            for item in items:
                framePen = painter.pen()
                framePen.setWidth(2.5)
                framePen.setStyle(QtCore.Qt.DashDotDotLine)
                framePen.setColor(QtGui.QColor(*PIVOT_COLOR))
                painter.setPen(framePen)
                if item[3] == "Line":
                    painter.drawLine(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * QtCore.QPointF(*item[1]))
                elif item[3] == "Circle":
                    painter.drawEllipse(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * item[1],
                                        SCALE_FACTOR * item[1])
        elif self.selectIdx != -1:
            painter.setBrush(QtGui.QColor(0, 0, 0, 0))
            select = self.objs[self.selectIdx]
            items = self.camera.frame_rasterize([select])
            for item in items:
                framePen = painter.pen()
                framePen.setWidth(2.5)
                framePen.setStyle(QtCore.Qt.DashDotDotLine)
                framePen.setColor(QtGui.QColor(*SELECT_COLOR))
                painter.setPen(framePen)
                if item[3] == "Line":
                    painter.drawLine(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * QtCore.QPointF(*item[1]))
                elif item[3] == "Circle":
                    painter.drawEllipse(SCALE_FACTOR * QtCore.QPointF(*item[0]), SCALE_FACTOR * item[1],
                                        SCALE_FACTOR * item[1])
        painter.end()

    def toggleRenderMode(self):
        self.mode = RasterMode.FRAME if self.mode == RasterMode.RAYTRACE else RasterMode.RAYTRACE
        self.repaint = True
        self.update()

    def reselect(self, selectIdx):
        self.pivotIdx = -1
        self.selectIdx = selectIdx
        self.repaint = True
        self.update()

# Object editing component
class CubeTeaInspectorWidget(QtWidgets.QWidget):
    def __init__(self, objs, idx):
        super().__init__()
        self.idx = idx
        self.objs = objs
        self.init = True
        self.allowPropertyCallbacks = True

    def init_inspector(self):
        self.nameLineEdit = QtWidgets.QLineEdit("")
        self.nameLineEdit.setFixedWidth(80)
        self.nameLineEdit.textChanged.connect(self.on_item_name_changed_rev)

        self.nameLabel = QtWidgets.QLabel(self.tr("&Name:"))
        self.nameLabel.setBuddy(self.nameLineEdit)

        doubleValidator = QtGui.QDoubleValidator(bottom=sys.float_info.min, decimals=4, top=sys.float_info.max)

        self.positionXEdit = QtWidgets.QLineEdit("0")
        self.positionXEdit.setValidator(doubleValidator)
        self.positionXEdit.setFixedWidth(80)
        self.positionXEdit.textChanged.connect(self.on_item_property_change("pos", 0))

        self.positionYEdit = QtWidgets.QLineEdit("0")
        self.positionYEdit.setValidator(doubleValidator)
        self.positionYEdit.setFixedWidth(80)
        self.positionYEdit.textChanged.connect(self.on_item_property_change("pos", 1))

        self.positionZEdit = QtWidgets.QLineEdit("0")
        self.positionZEdit.setValidator(doubleValidator)
        self.positionZEdit.setFixedWidth(80)
        self.positionZEdit.textChanged.connect(self.on_item_property_change("pos", 2))

        self.positionLabel = QtWidgets.QLabel(self.tr("&Position:"))
        self.positionLabel.setBuddy(self.positionXEdit)

        degreesValidator = QtGui.QDoubleValidator(bottom=0.0, decimals=4, top=359.999999)

        self.rotationXEdit = QtWidgets.QLineEdit("0")
        self.rotationXEdit.setValidator(degreesValidator)
        self.rotationXEdit.setFixedWidth(80)
        self.rotationXEdit.textChanged.connect(self.on_item_property_change("rot", 0))

        self.rotationYEdit = QtWidgets.QLineEdit("0")
        self.rotationYEdit.setValidator(degreesValidator)
        self.rotationYEdit.setFixedWidth(80)
        self.rotationYEdit.textChanged.connect(self.on_item_property_change("rot", 1))

        self.rotationZEdit = QtWidgets.QLineEdit("0")
        self.rotationZEdit.setValidator(degreesValidator)
        self.rotationZEdit.setFixedWidth(80)
        self.rotationZEdit.textChanged.connect(self.on_item_property_change("rot", 2))

        self.rotationLabel = QtWidgets.QLabel(self.tr("&Rotation:"))
        self.rotationLabel.setBuddy(self.rotationXEdit)

        radiansValidator = QtGui.QDoubleValidator(bottom=0.0, decimals=4, top=2*math.pi - 0.0001)

        self.quaternionWEdit = QtWidgets.QLineEdit("0")
        self.quaternionWEdit.setValidator(radiansValidator)
        self.quaternionWEdit.setFixedWidth(80)
        self.quaternionWEdit.textChanged.connect(self.on_item_property_change("quat", 0))

        self.quaternionXEdit = QtWidgets.QLineEdit("0")
        self.quaternionXEdit.setValidator(doubleValidator)
        self.quaternionXEdit.setFixedWidth(80)
        self.quaternionXEdit.textChanged.connect(self.on_item_property_change("quat", 1))

        self.quaternionYEdit = QtWidgets.QLineEdit("1")
        self.quaternionYEdit.setValidator(doubleValidator)
        self.quaternionYEdit.setFixedWidth(80)
        self.quaternionYEdit.textChanged.connect(self.on_item_property_change("quat", 2))

        self.quaternionZEdit = QtWidgets.QLineEdit("0")
        self.quaternionZEdit.setValidator(doubleValidator)
        self.quaternionZEdit.setFixedWidth(80)
        self.quaternionZEdit.textChanged.connect(self.on_item_property_change("quat", 3))

        self.quaternionLabel = QtWidgets.QLabel(self.tr("&Quaternion:"))
        self.quaternionLabel.setBuddy(self.quaternionWEdit)

        colorValidator = QtGui.QIntValidator(bottom=0, top=255)

        self.colorREdit = QtWidgets.QLineEdit("128")
        self.colorREdit.setValidator(colorValidator)
        self.colorREdit.setFixedWidth(80)
        self.colorREdit.textChanged.connect(self.on_item_property_change("color", 0))

        self.colorGEdit = QtWidgets.QLineEdit("128")
        self.colorGEdit.setValidator(colorValidator)
        self.colorGEdit.setFixedWidth(80)
        self.colorGEdit.textChanged.connect(self.on_item_property_change("color", 1))

        self.colorBEdit = QtWidgets.QLineEdit("128")
        self.colorBEdit.setValidator(colorValidator)
        self.colorBEdit.setFixedWidth(80)
        self.colorBEdit.textChanged.connect(self.on_item_property_change("color", 2))

        self.colorLabel = QtWidgets.QLabel(self.tr("&Color:"))
        self.colorLabel.setBuddy(self.colorREdit)

        self.dimensionXEdit = QtWidgets.QLineEdit("1.0000")
        self.dimensionXEdit.setValidator(doubleValidator)
        self.dimensionXEdit.setFixedWidth(80)
        self.dimensionXEdit.textChanged.connect(self.on_item_property_change("dims", 0))

        self.dimensionYEdit = QtWidgets.QLineEdit("1.0000")
        self.dimensionYEdit.setValidator(doubleValidator)
        self.dimensionYEdit.setFixedWidth(80)
        self.dimensionYEdit.textChanged.connect(self.on_item_property_change("dims", 1))

        self.dimensionZEdit = QtWidgets.QLineEdit("1.0000")
        self.dimensionZEdit.setValidator(doubleValidator)
        self.dimensionZEdit.setFixedWidth(80)
        self.dimensionZEdit.textChanged.connect(self.on_item_property_change("dims", 2))

        self.dimensionLabel = QtWidgets.QLabel(self.tr("&Dimension:"))
        self.dimensionLabel.setBuddy(self.dimensionXEdit)

        self.radiusEdit = QtWidgets.QLineEdit("1.0000")
        self.radiusEdit.setValidator(doubleValidator)
        self.radiusEdit.setFixedWidth(80)
        self.radiusEdit.textChanged.connect(self.on_item_property_change("rad"))

        self.radiusLabel = QtWidgets.QLabel(self.tr("&Radius:"))
        self.radiusLabel.setBuddy(self.radiusEdit)

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(self.nameLabel, 0, 0)
        gridLayout.addWidget(self.nameLineEdit, 0, 1)
        gridLayout.addWidget(self.positionLabel, 1, 0)
        gridLayout.addWidget(self.positionXEdit, 1, 1)
        gridLayout.addWidget(self.positionYEdit, 1, 2)
        gridLayout.addWidget(self.positionZEdit, 1, 3)
        gridLayout.addWidget(self.rotationLabel, 2, 0)
        gridLayout.addWidget(self.rotationXEdit, 2, 1)
        gridLayout.addWidget(self.rotationYEdit, 2, 2)
        gridLayout.addWidget(self.rotationZEdit, 2, 3)
        gridLayout.addWidget(self.quaternionLabel, 3, 0)
        gridLayout.addWidget(self.quaternionWEdit, 3, 1)
        gridLayout.addWidget(self.quaternionXEdit, 3, 2)
        gridLayout.addWidget(self.quaternionYEdit, 3, 3)
        gridLayout.addWidget(self.quaternionZEdit, 3, 4)
        gridLayout.addWidget(self.colorLabel, 4, 0)
        gridLayout.addWidget(self.colorREdit, 4, 1)
        gridLayout.addWidget(self.colorGEdit, 4, 2)
        gridLayout.addWidget(self.colorBEdit, 4, 3)
        gridLayout.setHorizontalSpacing(0)
        for i in range(1, 5):
            gridLayout.setColumnMinimumWidth(i, 20)
        self.setLayout(gridLayout)

    def default_inspector(self):
        self.allowPropertyCallbacks = False
        self.nameLineEdit.setText("")

        self.positionXEdit.setText("0")
        self.positionYEdit.setText("0")
        self.positionZEdit.setText("0")

        self.rotationXEdit.setText("0")
        self.rotationYEdit.setText("0")
        self.rotationZEdit.setText("0")

        self.quaternionWEdit.setText("0")
        self.quaternionXEdit.setText("0")
        self.quaternionYEdit.setText("1")
        self.quaternionZEdit.setText("0")

        self.colorREdit.setText("128")
        self.colorGEdit.setText("128")
        self.colorBEdit.setText("128")

        self.layout().removeWidget(self.dimensionLabel)
        self.layout().removeWidget(self.dimensionXEdit)
        self.layout().removeWidget(self.dimensionYEdit)
        self.layout().removeWidget(self.dimensionZEdit)
        self.dimensionLabel.hide()
        self.dimensionXEdit.hide()
        self.dimensionYEdit.hide()
        self.dimensionZEdit.hide()

        self.layout().removeWidget(self.radiusLabel)
        self.layout().removeWidget(self.radiusEdit)
        self.radiusLabel.hide()
        self.radiusEdit.hide()
        self.allowPropertyCallbacks = True

    def update_inspector(self):

        focus = self.objs[self.idx]
        self.nameLineEdit.setText(focus.name)

        pos = np.round(focus.position, decimals=4)
        self.positionXEdit.setText(str(pos[0]))
        self.positionYEdit.setText(str(pos[1]))
        self.positionZEdit.setText(str(pos[2]))

        euler = focus.get_euler()

        rot = np.round(euler, decimals=4)
        self.rotationXEdit.setText(str(rot[0]))
        self.rotationYEdit.setText(str(rot[1]))
        self.rotationZEdit.setText(str(rot[2]))

        rot2 = np.round(focus.quaternion, decimals=4)
        self.quaternionWEdit.setText(str(rot2[0]))
        self.quaternionXEdit.setText(str(rot2[1]))
        self.quaternionYEdit.setText(str(rot2[2]))
        self.quaternionZEdit.setText(str(rot2[3]))

        self.colorREdit.setText(str(focus.color[0]))
        self.colorGEdit.setText(str(focus.color[1]))
        self.colorBEdit.setText(str(focus.color[2]))

        if isinstance(focus, Box):
            dims = np.round(focus.dims, decimals=4)
            self.dimensionXEdit.setText(str(dims[0]))
            self.dimensionYEdit.setText(str(dims[1]))
            self.dimensionZEdit.setText(str(dims[2]))
            self.layout().addWidget(self.dimensionLabel, 5, 0)
            self.layout().addWidget(self.dimensionXEdit, 5, 1)
            self.layout().addWidget(self.dimensionYEdit, 5, 2)
            self.layout().addWidget(self.dimensionZEdit, 5, 3)
            self.dimensionLabel.show()
            self.dimensionXEdit.show()
            self.dimensionYEdit.show()
            self.dimensionZEdit.show()
        else:
            self.layout().removeWidget(self.dimensionLabel)
            self.layout().removeWidget(self.dimensionXEdit)
            self.layout().removeWidget(self.dimensionYEdit)
            self.layout().removeWidget(self.dimensionZEdit)
            self.dimensionLabel.hide()
            self.dimensionXEdit.hide()
            self.dimensionYEdit.hide()
            self.dimensionZEdit.hide()

        if isinstance(focus, Sphere):
            self.radiusEdit.setText(str(round(focus.radius, 5)))
            self.layout().addWidget(self.radiusLabel, 5, 0)
            self.layout().addWidget(self.radiusEdit, 5, 1)
            self.radiusLabel.show()
            self.radiusEdit.show()
        else:
            self.layout().removeWidget(self.radiusLabel)
            self.layout().removeWidget(self.radiusEdit)
            self.radiusLabel.hide()
            self.radiusEdit.hide()

    def on_obj_entry_clicked(self, idx):
        prev_idx = self.idx
        self.idx = idx
        if prev_idx == -1 and self.idx != -1 and self.init:
            self.init = False
            self.init_inspector()
        if self.idx != -1:
            self.update_inspector()
        elif not self.init:
            self.default_inspector()

    def on_item_name_changed(self):
        if self.idx != -1:
            self.update_inspector()
        else:
            self.default_inspector()

    def on_item_name_changed_rev(self, text):
        if self.idx != -1:
            self.objs[self.idx].name = text
            self.parentWidget().on_item_name_changed_rev(self.idx)

    def on_item_property_change(self, tag, idx=-1):
        def inner_callback(text):
            if text == "" or text == "-" or not self.allowPropertyCallbacks:
                return
            obj = self.objs[self.idx]
            if tag == "pos":
                obj.position[idx] = float(text)
            elif tag == "rot":
                euler = obj.get_euler()
                euler[idx] = float(text)
                obj.set_euler(np.array(euler))
                rot2 = np.round(obj.quaternion, decimals=4)

                self.quaternionWEdit.textChanged.disconnect()
                self.quaternionWEdit.setText(str(rot2[0]))
                self.quaternionWEdit.textChanged.connect(self.on_item_property_change("quat", 0))

                self.quaternionXEdit.textChanged.disconnect()
                self.quaternionXEdit.setText(str(rot2[1]))
                self.quaternionXEdit.textChanged.connect(self.on_item_property_change("quat", 1))

                self.quaternionYEdit.textChanged.disconnect()
                self.quaternionYEdit.setText(str(rot2[2]))
                self.quaternionYEdit.textChanged.connect(self.on_item_property_change("quat", 2))

                self.quaternionZEdit.textChanged.disconnect()
                self.quaternionZEdit.setText(str(rot2[3]))
                self.quaternionZEdit.textChanged.connect(self.on_item_property_change("quat", 3))
            elif tag == "quat":
                obj.quaternion[idx] = float(text)
                rot = np.round(obj.get_euler(), decimals=4)

                self.rotationXEdit.textChanged.disconnect()
                self.rotationXEdit.setText(str(rot[0]))
                self.rotationXEdit.textChanged.connect(self.on_item_property_change("rot", 0))

                self.rotationYEdit.textChanged.disconnect()
                self.rotationYEdit.setText(str(rot[1]))
                self.rotationYEdit.textChanged.connect(self.on_item_property_change("rot", 1))

                self.rotationZEdit.textChanged.disconnect()
                self.rotationZEdit.setText(str(rot[2]))
                self.rotationZEdit.textChanged.connect(self.on_item_property_change("rot", 2))
            elif tag == "color":
                obj.color[idx] = int(text)
                obj.update_colors()
            elif tag == "dims":
                obj.dims[idx] = float(text)
            elif tag == "rad":
                obj.radius = float(text)
            self.parentWidget().update_render()
        return inner_callback

    def get_pivot(self):
        return self.idx

class CubeTeaInspectorDockWidget(QtWidgets.QDockWidget):
    def __init__(self, objs, idx=-1):
        super().__init__()
        self.setWindowTitle("Inspector - Nothing Focused")
        self.inspector = CubeTeaInspectorWidget(objs, idx=idx)
        self.setWidget(self.inspector)
        self.show()

    def on_obj_entry_clicked(self, idx):
        self.setWindowTitle("Inspector" if idx != -1 else "Inspector - Nothing Focused")
        self.inspector.on_obj_entry_clicked(idx)

    def on_item_name_changed(self):
        self.inspector.on_item_name_changed()

    def on_item_name_changed_rev(self, idx):
        self.parentWidget().on_item_name_changed_rev(idx)

    def update_render(self):
        self.parentWidget().update_render()

    def on_new_object_added(self, idx):
        self.on_obj_entry_clicked(idx)

    def on_current_object_deleted(self):
        self.on_obj_entry_clicked(-1)

    def get_pivot(self):
        return self.inspector.get_pivot()

# Object selection component
class CubeTeaHierarchyListWidget(QtWidgets.QListView):
    def __init__(self, objs):
        super().__init__()
        self.model = QtGui.QStandardItemModel(self)
        self.objs = objs
        for obj in objs:
            symbol = "●" if isinstance(obj, Sphere) else "◼"
            item = QtGui.QStandardItem("{0} {1}".format(symbol, obj.name))
            self.model.appendRow(item)
        self.setModel(self.model)
        self.setViewMode(QtWidgets.QListView.ListMode)
        self.show()
        self.clicked.connect(self.on_obj_entry_clicked)
        self.model.itemChanged.connect(self.on_item_changed)
        self.allowCallbacks = True

    def on_obj_entry_clicked(self, index):
        if self.allowCallbacks:
            self.parentWidget().on_obj_entry_clicked(index.row())

    def on_item_changed(self, item):
        if self.allowCallbacks:
            if item.index().row() >= len(self.objs):
                return
            obj = self.objs[item.index().row()]
            symbol = "● " if isinstance(obj, Sphere) else "◼ "
            if len(item.text()) >= 2 and item.text()[0:2] == symbol:
                obj.name = item.text()[2:]
            else:
                obj.name = ""
                self.model.setItem(item.index().row(), 0, QtGui.QStandardItem(symbol))
            self.parentWidget().on_item_name_changed()

    def on_item_changed_rev(self, idx):
        symbol = "● " if isinstance(self.objs[idx], Sphere) else "◼ "
        self.model.setItem(idx, 0, QtGui.QStandardItem(symbol + self.objs[idx].name))

    def on_new_object_added(self):
        self.allowCallbacks = False
        self.model.clear()
        for obj in self.objs:
            symbol = "●" if isinstance(obj, Sphere) else "◼"
            item = QtGui.QStandardItem("{0} {1}".format(symbol, obj.name))
            self.model.appendRow(item)
        self.allowCallbacks = True

    def on_current_object_deleted(self):
        self.allowCallbacks = False
        self.model.clear()
        for obj in self.objs:
            symbol = "●" if isinstance(obj, Sphere) else "◼"
            item = QtGui.QStandardItem("{0} {1}".format(symbol, obj.name))
            self.model.appendRow(item)
        self.allowCallbacks = True

class CubeTeaHierarchyDockWidget(QtWidgets.QDockWidget):
    def __init__(self, objs):
        super().__init__()
        self.setWindowTitle("Object List")
        self.hierarchy = CubeTeaHierarchyListWidget(objs=objs)
        self.setWidget(self.hierarchy)
        self.show()

    def on_obj_entry_clicked(self, idx):
        self.parentWidget().on_obj_entry_clicked(idx)

    def on_item_name_changed(self):
        self.parentWidget().on_item_name_changed()

    def on_item_name_changed_rev(self, idx):
        self.hierarchy.on_item_changed_rev(idx)

    def on_new_object_added(self):
        self.hierarchy.on_new_object_added()

    def on_current_object_deleted(self):
        self.hierarchy.on_current_object_deleted()

# Object addition/removal component
class CubeTeaHierarchyMenuDockWidget(QtWidgets.QDockWidget):
    def __init__(self, objs):
        super().__init__()
        self.setWindowTitle("List Actions")
        self.hierarchyMenu = CubeTeaHierarchyMenuWidget(objs)
        self.setWidget(self.hierarchyMenu)

    def on_new_object_added(self):
        self.parentWidget().on_new_object_added()

    def on_obj_entry_clicked(self, idx):
        self.hierarchyMenu.on_obj_entry_clicked(idx)

    def on_current_object_deleted(self):
        self.parentWidget().on_current_object_deleted()

class CubeTeaHierarchyMenuWidget(QtWidgets.QWidget):
    def __init__(self, objs):
        super().__init__()
        self.objs = objs
        self.idx = -1
        gridLayout = QtWidgets.QGridLayout()
        self.addBoxButton = QtWidgets.QPushButton("&Add Box", self)
        self.addBoxButton.setFixedHeight(30)
        self.addBoxButton.setContentsMargins(30, 5, 30, 5)
        self.addBoxButton.clicked.connect(self.add_box)
        self.addSphereButton = QtWidgets.QPushButton("&Add Sphere", self)
        self.addSphereButton.setFixedHeight(30)
        self.addSphereButton.setContentsMargins(30, 5, 30, 5)
        self.addSphereButton.clicked.connect(self.add_sphere)
        self.deleteButton = QtWidgets.QPushButton("&Delete", self)
        self.deleteButton.setFixedHeight(30)
        self.deleteButton.setContentsMargins(30, 5, 30, 5)
        self.deleteButton.clicked.connect(self.delete_current)
        gridLayout.addWidget(self.addBoxButton, 0, 0)
        gridLayout.addWidget(self.addSphereButton, 0, 1)
        gridLayout.addWidget(self.deleteButton, 0, 2)
        self.setLayout(gridLayout)

    def on_obj_entry_clicked(self, idx):
        self.idx = idx
        self.deleteButton.setEnabled(idx != -1)

    def get_primitive_count(self):
        primitives = {
            "Box": 0,
            "Sphere": 0
        }
        for obj in self.objs:
            if isinstance(obj, Box):
                primitives["Box"] += 1
            elif isinstance(obj, Sphere):
                primitives["Sphere"] += 1
        return primitives

    def add_box(self):
        new_name = "box{0}".format(self.get_primitive_count()["Box"] + 1)
        box = Box(position=camera.position.copy() + camera.basis().copy() @ np.array([0, 2, 0]),
                  quaternion=camera.quaternion.copy(), name=new_name)
        self.objs.append(box)
        self.parentWidget().on_new_object_added()

    def add_sphere(self):
        new_name = "sphere{0}".format(self.get_primitive_count()["Sphere"] + 1)
        sphere = Sphere(position=camera.position.copy() + camera.basis().copy() @ np.array([0, 2, 0]),
                        quaternion=camera.quaternion.copy(), name=new_name)
        self.objs.append(sphere)
        self.parentWidget().on_new_object_added()

    def delete_current(self):
        if self.idx != -1 and self.idx < len(self.objs):
            del self.objs[self.idx]
            self.parentWidget().on_current_object_deleted()
            self.idx = -1

# Camera control component
class CubeTeaCameraDockWidget(QtWidgets.QDockWidget):
    def __init__(self, objs, camera):
        super().__init__()
        self.setWindowTitle("Camera Controls")
        self.controls = CubeTeaCameraWidget(objs, camera)
        self.setWidget(self.controls)
        self.show()

    def update_render(self):
        self.parentWidget().update_render()

    def get_pivot(self):
        return self.parentWidget().get_pivot()

    def toggle_raster(self, mode):
        return self.parentWidget().toggle_raster(mode)

    def reset_pivot(self):
        return self.controls.reset_pivot()

class CubeTeaCameraWidget(QtWidgets.QWidget):
    def __init__(self, objs, camera):
        super().__init__()
        gridLayout = QtWidgets.QGridLayout()
        self.objs = objs
        self.camera = camera
        self.pivot = None

        self.upButton = QtWidgets.QPushButton("&Up", self)
        self.upButton.setFixedHeight(30)
        self.upButton.setContentsMargins(30, 5, 30, 5)
        self.upButton.pressed.connect(self.handle_camera_input("translate", "up"))
        self.upButton.setAutoRepeat(True)
        self.forwardButton = QtWidgets.QPushButton("&Forward", self)
        self.forwardButton.setFixedHeight(30)
        self.forwardButton.setContentsMargins(30, 5, 30, 5)
        self.forwardButton.pressed.connect(self.handle_camera_input("translate", "forward"))
        self.forwardButton.setAutoRepeat(True)
        self.downButton = QtWidgets.QPushButton("&Down", self)
        self.downButton.setFixedHeight(30)
        self.downButton.setContentsMargins(30, 5, 30, 5)
        self.downButton.pressed.connect(self.handle_camera_input("translate", "down"))
        self.downButton.setAutoRepeat(True)

        self.leftButton = QtWidgets.QPushButton("&Left", self)
        self.leftButton.setFixedHeight(30)
        self.leftButton.setContentsMargins(30, 5, 30, 5)
        self.leftButton.pressed.connect(self.handle_camera_input("translate", "left"))
        self.leftButton.setAutoRepeat(True)
        self.backwardButton = QtWidgets.QPushButton("&Backward", self)
        self.backwardButton.setFixedHeight(30)
        self.backwardButton.setContentsMargins(30, 5, 30, 5)
        self.backwardButton.pressed.connect(self.handle_camera_input("translate", "backward"))
        self.backwardButton.setAutoRepeat(True)
        self.rightButton = QtWidgets.QPushButton("&Right", self)
        self.rightButton.setFixedHeight(30)
        self.rightButton.setContentsMargins(30, 5, 30, 5)
        self.rightButton.pressed.connect(self.handle_camera_input("translate", "right"))
        self.rightButton.setAutoRepeat(True)

        self.rollLeftButton = QtWidgets.QPushButton("&Roll Left", self)
        self.rollLeftButton.setFixedHeight(30)
        self.rollLeftButton.setContentsMargins(30, 5, 30, 5)
        self.rollLeftButton.pressed.connect(self.handle_camera_input("rotation", "rollL"))
        self.rollLeftButton.setAutoRepeat(True)
        self.pitchUpButton = QtWidgets.QPushButton("&Pitch Up", self)
        self.pitchUpButton.setFixedHeight(30)
        self.pitchUpButton.setContentsMargins(30, 5, 30, 5)
        self.pitchUpButton.pressed.connect(self.handle_camera_input("rotation", "pitchU"))
        self.pitchUpButton.setAutoRepeat(True)
        self.rollRightButton = QtWidgets.QPushButton("&Roll Right", self)
        self.rollRightButton.setFixedHeight(30)
        self.rollRightButton.setContentsMargins(30, 5, 30, 5)
        self.rollRightButton.pressed.connect(self.handle_camera_input("rotation", "rollR"))
        self.rollRightButton.setAutoRepeat(True)

        self.yawLeftButton = QtWidgets.QPushButton("&Yaw Left", self)
        self.yawLeftButton.setFixedHeight(30)
        self.yawLeftButton.setContentsMargins(30, 5, 30, 5)
        self.yawLeftButton.pressed.connect(self.handle_camera_input("rotation", "yawL"))
        self.yawLeftButton.setAutoRepeat(True)
        self.pitchDownButton = QtWidgets.QPushButton("&Pitch Down", self)
        self.pitchDownButton.setFixedHeight(30)
        self.pitchDownButton.setContentsMargins(30, 5, 30, 5)
        self.pitchDownButton.pressed.connect(self.handle_camera_input("rotation", "pitchD"))
        self.pitchDownButton.setAutoRepeat(True)
        self.yawRightButton = QtWidgets.QPushButton("&Yaw Right", self)
        self.yawRightButton.setFixedHeight(30)
        self.yawRightButton.setContentsMargins(30, 5, 30, 5)
        self.yawRightButton.pressed.connect(self.handle_camera_input("rotation", "yawR"))
        self.yawRightButton.setAutoRepeat(True)

        self.resetButton = QtWidgets.QPushButton("&Reset", self)
        self.resetButton.setFixedHeight(30)
        self.resetButton.setContentsMargins(30, 5, 30, 5)
        self.resetButton.clicked.connect(self.handle_camera_input("misc", "reset"))
        self.rasterModeBox = QtWidgets.QCheckBox("Use Raytracing", self)
        self.rasterModeBox.setCheckState(QtCore.Qt.Unchecked)
        self.rasterModeBox.stateChanged.connect(self.handle_camera_input("misc", "raster"))
        self.pivotButton = QtWidgets.QPushButton("&Pivot", self)
        self.pivotButton.setFixedHeight(30)
        self.pivotButton.setContentsMargins(30, 5, 30, 5)
        self.pivotButton.clicked.connect(self.handle_camera_input("misc", "pivot"))

        self.translationLabel = QtWidgets.QLabel(self.tr("Translation"))
        self.rotationLabel = QtWidgets.QLabel(self.tr("Rotation"))
        self.miscLabel = QtWidgets.QLabel(self.tr("Miscellaneous"))
        self.translationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.translationLabel.setFixedHeight(20)
        self.rotationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.rotationLabel.setFixedHeight(20)
        self.miscLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.miscLabel.setFixedHeight(20)

        gridLayout.addWidget(self.translationLabel, 0, 1)
        gridLayout.addWidget(self.upButton, 1, 0)
        gridLayout.addWidget(self.forwardButton, 1, 1)
        gridLayout.addWidget(self.downButton, 1, 2)
        gridLayout.addWidget(self.leftButton, 2, 0)
        gridLayout.addWidget(self.backwardButton, 2, 1)
        gridLayout.addWidget(self.rightButton, 2, 2)

        gridLayout.addWidget(self.rotationLabel, 3, 1)
        gridLayout.addWidget(self.rollLeftButton, 4, 0)
        gridLayout.addWidget(self.pitchUpButton, 4, 1)
        gridLayout.addWidget(self.rollRightButton, 4, 2)
        gridLayout.addWidget(self.yawLeftButton, 5, 0)
        gridLayout.addWidget(self.pitchDownButton, 5, 1)
        gridLayout.addWidget(self.yawRightButton, 5, 2)

        gridLayout.addWidget(self.miscLabel, 6, 1)
        gridLayout.addWidget(self.resetButton, 7, 0)
        gridLayout.addWidget(self.rasterModeBox, 7, 1)
        gridLayout.addWidget(self.pivotButton, 7, 2)

        self.setLayout(gridLayout)

    def handle_camera_input(self, tag1, tag2):
        def inner_callback():
            camera = self.camera
            if tag1 == "translate":
                delta = [0, 0, 0]
                if tag2 == "up":
                    delta[2] = -1
                elif tag2 == "down":
                    delta[2] = 1
                elif tag2 == "forward":
                    delta[1] = 1
                elif tag2 == "backward":
                    delta[1] = -1
                elif tag2 == "right":
                    delta[0] = 1
                elif tag2 == "left":
                    delta[0] = -1
                camera.position = camera.position.copy() + TRANSLATION_STEP * (camera.basis().copy() @ np.array(delta))
                self.parentWidget().update_render()
            elif tag1 == "rotation":
                delta = [0, 0, 0]
                if tag2 == "rollR":
                    delta[1] = 1
                elif tag2 == "rollL":
                    delta[1] = -1
                elif tag2 == "pitchU":
                    delta[0] = -1
                elif tag2 == "pitchD":
                    delta[0] = 1
                elif tag2 == "yawR":
                    delta[2] = -1
                elif tag2 == "yawL":
                    delta[2] = 1
                pivotPos = self.pivot.position if self.pivot is not None else None
                camera.rotate(rot_quat(np.array(delta), ROTATION_STEP), pivot=pivotPos)
                self.parentWidget().update_render()
            elif tag1 == "misc":
                if tag2 == "reset":
                    camera.position = np.array([0, -1, 0])
                    camera.quaternion = np.array([0, 0, 1, 0])
                    self.parentWidget().update_render()
                elif tag2 == "raster":
                    if self.rasterModeBox.isChecked():
                        self.parentWidget().toggle_raster(RasterMode.RAYTRACE)
                    else:
                        self.parentWidget().toggle_raster(RasterMode.FRAME)
                elif tag2 == "pivot":
                    pivotIdx = self.parentWidget().get_pivot()
                    self.pivot = self.objs[pivotIdx] if pivotIdx != -1 else None
                    self.parentWidget().update_render()
        return inner_callback

    def reset_pivot(self):
        self.pivot = None

# File operation component
class CubeTeaFileMenuDockWidget(QtWidgets.QDockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Actions")
        self.filemenu = CubeTeaFileMenuWidget()
        self.setWidget(self.filemenu)
        self.show()

    def on_file_operation(self, operation, loc):
        self.parentWidget().on_file_operation(operation, loc)

    def on_new_file(self):
        self.parentWidget().on_new_file()

class CubeTeaFileMenuWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        gridLayout = QtWidgets.QGridLayout()
        self.newButton = QtWidgets.QPushButton("&New Scene", self)
        self.newButton.setFixedHeight(30)
        self.newButton.setContentsMargins(30, 5, 30, 5)
        self.newButton.clicked.connect(self.on_new_file)
        self.loadButton = QtWidgets.QPushButton("&Load Scene", self)
        self.loadButton.setFixedHeight(30)
        self.loadButton.setContentsMargins(30, 5, 30, 5)
        self.loadButton.clicked.connect(self.on_file_load)
        self.saveButton = QtWidgets.QPushButton("&Save Scene", self)
        self.saveButton.setFixedHeight(30)
        self.saveButton.setContentsMargins(30, 5, 30, 5)
        self.saveButton.clicked.connect(self.on_file_save)
        gridLayout.addWidget(self.newButton, 0, 0)
        gridLayout.addWidget(self.loadButton, 0, 1)
        gridLayout.addWidget(self.saveButton, 0, 2)
        self.setLayout(gridLayout)

    def on_new_file(self):
        self.parentWidget().on_new_file()

    def on_file_load(self):
        load_file = QtWidgets.QFileDialog.getOpenFileName(self,
                    self.tr("Open Scene"), "~/", self.tr("JSON Files (*.json)"))
        self.parentWidget().on_file_operation(FileOperation.LOAD, load_file[0])

    def on_file_save(self):
        save_file = QtWidgets.QFileDialog.getSaveFileName(self,
                    self.tr("Save Scene"), "~/", self.tr("JSON Files (*.json)"))
        self.parentWidget().on_file_operation(FileOperation.SAVE, save_file[0])

# Main application
class CubeTeaWidget(QtWidgets.QMainWindow):
    def __init__(self, objs, camera=None):
        super().__init__()

        self.objs = objs
        self.camera = Camera() if camera is None else camera

        self.setWindowTitle("CubeTea")
        self.viewport = CubeTeaRasterWidget(self.objs, self.camera)
        self.setCentralWidget(self.viewport)

        # Add left dock widgets
        self.fileMenuDock = CubeTeaFileMenuDockWidget()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.fileMenuDock)
        self.hierarchyMenuDock = CubeTeaHierarchyMenuDockWidget(objs=self.objs)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.hierarchyMenuDock)
        self.hierarchyDock = CubeTeaHierarchyDockWidget(objs=self.objs)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.hierarchyDock)

        # Add right dock widgets
        self.inspectorDock = CubeTeaInspectorDockWidget(objs=self.objs)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.inspectorDock)
        self.cameraDock = CubeTeaCameraDockWidget(self.objs, self.camera)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.cameraDock)

        # Status Bar
        self.status = self.statusBar()
        self.status.showMessage("3D viewer initialized")

        self.init_load()

        self.show()

    # Signal chains
    def on_obj_entry_clicked(self, idx):
        self.cameraDock.reset_pivot()
        self.viewport.reselect(idx)
        self.inspectorDock.on_obj_entry_clicked(idx)
        self.hierarchyMenuDock.on_obj_entry_clicked(idx)
        if idx == -1:
            self.status.showMessage("Object focus lost.")
        else:
            self.status.showMessage("Focused on object {0}".format(self.objs[idx].name))

    def on_item_name_changed(self):
        self.inspectorDock.on_item_name_changed()
        self.autosave()

    def on_item_name_changed_rev(self, idx):
        self.hierarchyDock.on_item_name_changed_rev(idx)
        self.autosave()

    def update_render(self):
        self.viewport.repaint = True
        self.viewport.update()
        self.autosave()

    def on_new_object_added(self):
        self.viewport.reselect(len(self.objs)-1)
        self.inspectorDock.on_new_object_added(len(self.objs)-1)
        self.hierarchyDock.on_new_object_added()
        self.autosave()
        self.status.showMessage("New object {0} added.".format(self.objs[len(self.objs)-1].name))

    def on_current_object_deleted(self):
        self.viewport.reselect(-1)
        self.inspectorDock.on_current_object_deleted()
        self.hierarchyDock.on_current_object_deleted()
        self.autosave()
        self.status.showMessage("")

    def toggle_raster(self, mode):
        if mode != self.viewport.mode:
            self.viewport.toggleRenderMode()
            self.autosave()
            if mode == RasterMode.FRAME:
                self.status.showMessage("Entered frame mode.")
            elif mode == RasterMode.RAYTRACE:
                self.status.showMessage("Entered raytrace mode.")

    def on_file_operation(self, operation, loc):
        if operation == FileOperation.SAVE:
            self.save(loc)
        elif operation == FileOperation.LOAD:
            self.load(loc)

    def on_new_file(self):
        self.new_file()

    def get_pivot(self):
        pivotIdx = self.inspectorDock.get_pivot();
        self.viewport.pivotIdx = pivotIdx
        return pivotIdx

    # Automatically saves current editor state to a JSON file in local storage
    def autosave(self):
        loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.TempLocation)
        self.save(loc, "cubetea_AUTO.json", True)

    # Saves current editor state as a JSON file
    def save(self, loc, name=None, auto=False):
        saveState = {"objs": [obj.dict() for obj in self.objs] + [camera.dict()]}
        saveData = json.dumps(saveState)
        file_ptr = open("{0}/{1}".format(loc, name) if name is not None else loc, "w+")
        file_ptr.truncate()
        file_ptr.write(saveData)
        file_ptr.close()
        if not auto:
            self.status.showMessage("Saved file {0} successfully.".format(
                ("{0}/{1}".format(loc, name) if name is not None else loc)))

    # Looks for autosave and loads it
    def init_load(self):
        # look for autosave
        loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.TempLocation)
        autosave = "{0}/{1}".format(loc, "cubetea_AUTO.json")
        if (os.path.exists(autosave)):
            self.load(autosave, True)
            self.status.showMessage("Successfully loaded auto-save.")
        else:
            self.new_file()

    # Loads a file at loc
    def load(self, loc, auto=False):
        try:
            file_ptr = open(loc, "r")
            saveState = json.loads(file_ptr.read())
            try:
                new_camera, objs = load_objs(saveState["objs"])
            except KeyError:
                raise TypeError
            self.camera.position = new_camera.position
            self.camera.dims = new_camera.dims
            self.camera.vdims = new_camera.vdims
            self.camera.quaternion = new_camera.quaternion
            self.objs.clear()
            for obj in objs:
                self.objs.append(obj)
            self.reset_UI()
            if not auto:
                self.autosave()
        except TypeError:
            self.status.showMessage("JSON file found is invalid!")

    # Creates a new, default scene
    def new_file(self):
        # Add primitives
        self.camera.position = np.array([0, -1, 0])
        self.camera.quaternion = np.array([0, 0, 1, 0])
        self.camera.dims = np.array([10, 10])
        self.camera.vdims = np.array([480, 480])
        box1 = Box(np.array([0, 2, 0]), name="box1", dims=np.array([2, 1, 3]), color=np.array([0, 128, 0]))
        box1.rotate(rot_quat(axis=np.array([0, 1, 1]), ang=math.pi / 4))
        sphere1 = Sphere(np.array([1, 4, 1]), name="sphere1", radius=3, color=np.array([0, 40, 160]))
        self.objs.clear()
        self.objs.append(box1)
        self.objs.append(sphere1)
        self.reset_UI()
        self.status.showMessage("New scene created.")
        self.autosave()

    # Resets UI components
    def reset_UI(self):
        self.viewport.reselect(-1)
        self.inspectorDock.on_current_object_deleted()
        self.hierarchyDock.on_current_object_deleted()
        self.hierarchyMenuDock.on_obj_entry_clicked(-1)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle('plastique')

    camera = camera=Camera(position=np.array([0, -1, 0]), dims=np.array([10, 10]), viewport_dims=np.array([480, 480]))
    widget = CubeTeaWidget(objs=[], camera=camera)
    # Window dimensions
    geometry = app.desktop().availableGeometry(widget)
    widget.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.8)
    widget.show()

    sys.exit(app.exec_())