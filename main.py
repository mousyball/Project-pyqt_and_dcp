# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QApplication, QFileDialog, QPushButton, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from functools import partial
import numpy as np
import sys
import cv2
import ctypes

class dcp_param(ctypes.Structure):

	def __init__(self):
		self.min_radius = 7
		self.box_radius = 100
		self.eps = 0.1
		self.edgemode = Tile
		self.omega = 0.95
		self.airlight_clip_value = 256
		self.airlight_offset = 80
		self.t0 = 0.1
		self.scaling_factor = 1.0
		self.airlight_diff = 1
		self.sky_offset = 0
		self.al_lower_bound = 0
		self.al_lambda = 0.95
		self.sky_intensity = 150
		self.sky_var = 0.6
		self.t1 = 0.4

	pass

dcp_param._fields_ = [
        ("min_radius", ctypes.c_int),
        ("box_radius", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("Edge", ctypes.c_int),
        ("omega", ctypes.c_float),
        ("airlight_clip_value", ctypes.c_float),
        ("airlight_offset", ctypes.c_int),
        ("t0", ctypes.c_float),
        ("scaling_factor", ctypes.c_float),
        ("airlight_diff", ctypes.c_float),
        ("sky_offset", ctypes.c_float),
        ("al_lower_bound", ctypes.c_int),
        ("al_lambda", ctypes.c_float),
        ("sky_intensity", ctypes.c_int),
        ("sky_var", ctypes.c_float),
        ("t1", ctypes.c_float),
        ("airlight_highest", ctypes.c_float),
        ("airlight_curr", ctypes.c_float * 4),
        ("airlight_prev", ctypes.c_float * 4)
]

(Tile, Smear) = (0, 1)
DEPTH_8U = 1

class Example(QWidget):

    MIN_RADIUS = 7
    BOX_RADIUS = 28
    OMEGA = 0.95
    VARIANCE = 0.6
    T0 = 0.1
    T1 = 0.3
    AL_OFFSET = 80


    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.dcp_param = dcp_param()
        self.dcp_lib = ctypes.cdll.LoadLibrary('./dcp.so')

        self.sw = self.Switcher()

        myScaledPixmap1, myScaledPixmap2 = self.loadInitPixmap()
        self.dcpInit(self.img_load)

        my_layout = self.myLayout(myScaledPixmap1, myScaledPixmap2)
        self.setLayout(my_layout)

        self.setFixedSize(1250,600)

        self.move(300, 200)
        self.setWindowTitle('Dark Channel Prior with Sky Preservation')
        self.show()

    def dcpInit(self, img):
        # Get image attribute
        hei, wid, channel = img.shape
        wid_down = int(wid * self.dcp_param.scaling_factor + 0.5)
        hei_down = int(hei * self.dcp_param.scaling_factor + 0.5)

        # Initialize the DCP parameters
        self.dcp_lib.init_dcp_param.argtypes = [ctypes.POINTER(dcp_param)]
        self.dcp_lib.init_dcp_param.restype = None
        self.dcp_lib.init_dcp_param(self.dcp_param)

        # Initialize parameter, pthread, LUT, and image spaces.
        self.dcp_lib.init_frame_param.restype = None
        self.dcp_lib.init_frame_param.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.dcp_lib.init_frame_param(wid, hei, self.dcp_param.scaling_factor);
        self.dcp_lib.init_pthread_param.restype = None
        self.dcp_lib.init_pthread_param.argtypes = None
        self.dcp_lib.init_pthread_param()
        self.dcp_lib.init_LUT.restype = None
        self.dcp_lib.init_LUT.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.dcp_lib.init_LUT(wid_down, hei_down, wid, hei, self.dcp_param.scaling_factor)
        self.dcp_lib.init_IMAGE.restype = None
        self.dcp_lib.init_IMAGE.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.dcp_lib.init_IMAGE(wid_down, hei_down, wid, hei, DEPTH_8U)

        # Create image spaces
        self.img_input = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.img_space = (ctypes.c_ubyte * wid * hei * channel)()
        self.img_output = ctypes.cast(self.img_space, ctypes.POINTER(ctypes.c_ubyte))

        # Create buffer for reading image data
        self.buff = (ctypes.c_ubyte * hei * wid * channel).from_address(ctypes.addressof(self.img_output.contents))

    def dcpProcess(self, img):
        hei, wid, channel = img.shape

        # Call the DCP process
        self.dcp_lib.DCP_process.restype = None
        self.dcp_lib.DCP_process.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(dcp_param)]
        self.dcp_lib.DCP_process(self.img_input, self.img_output, self.dcp_param)

        # Read the buff(self.img_output) to np array
        data = np.ndarray(buffer=self.buff, dtype=np.uint8, shape=(hei, wid, channel))

        return data

    def dcpFree(self):
        # Delete memories
        del self.img_load
        del self.img_input
        del self.img_space
        del self.img_output
        del self.buff
        self.dcp_lib.Free_LUT()
        self.dcp_lib.Free_TMatrix()
        self.dcp_lib.Free_pthread_param()


    def myLayout(self, myScaledPixmap1, myScaledPixmap2):

        # Input Dialog
        self.lbl_fileIn = QPushButton("Load Image", self)
        self.lbl_fileIn.clicked.connect(self.getInputImg)
        self.lbl_fileIn.resize(self.lbl_fileIn.sizeHint())

        # Place the compared images
        self.lbl_in = QLabel(self)
        self.lbl_in.setPixmap(myScaledPixmap1)
        self.lbl_out = QLabel(self)
        self.lbl_out.setPixmap(myScaledPixmap2)

        # Labels for text
        #====================#
        txt_min_radius = QLabel('Min Radius', self)
        val_min_radius = QLabel(str(Example.MIN_RADIUS), self)
        val_min_radius.setFixedWidth(30)
        #====================#
        txt_box_radius = QLabel('Box Radius', self)
        val_box_radius = QLabel(str(Example.BOX_RADIUS), self)
        val_box_radius.setFixedWidth(30)
        #====================#
        txt_omega = QLabel('Omega', self)
        val_omega = QLabel(str(Example.OMEGA), self)
        val_omega.setFixedWidth(30)
        #====================#
        txt_var = QLabel('Variance', self)
        val_var = QLabel(str(Example.VARIANCE), self)
        val_var.setFixedWidth(30)
        #====================#
        txt_t0 = QLabel('T0', self)
        val_t0 = QLabel(str(Example.T0), self)
        val_t0.setFixedWidth(30)
        #====================#
        txt_t1 = QLabel('T1', self)
        val_t1 = QLabel(str(Example.T1), self)
        val_t1.setFixedWidth(30)
        #====================#
        txt_al_offset = QLabel('AL Offset', self)
        val_al_offset = QLabel(str(Example.AL_OFFSET), self)
        val_al_offset.setFixedWidth(30)
        #====================#


        # Place the slider
        #====================#
        sld_min_radius = QSlider(Qt.Horizontal, self)
        sld_min_radius.setFocusPolicy(Qt.NoFocus)
        sld_min_radius.setRange(7, 107)
        sld_min_radius.setValue(Example.MIN_RADIUS)
        sld_min_radius.setTickPosition(QSlider.TicksBelow)
        sld_min_radius.valueChanged.connect(partial(self.setIntSlider, sld_min_radius, val_min_radius))
        sld_min_radius.valueChanged.connect(partial(self.setDcpParam, sld_min_radius, 'sld_min_radius'))
        #====================#
        sld_box_radius = QSlider(Qt.Horizontal, self)
        sld_box_radius.setFocusPolicy(Qt.NoFocus)
        sld_box_radius.setRange(7, 107)
        sld_box_radius.setValue(Example.BOX_RADIUS)
        sld_box_radius.setTickPosition(QSlider.TicksBelow)
        sld_box_radius.valueChanged.connect(partial(self.setIntSlider, sld_box_radius, val_box_radius))
        sld_box_radius.valueChanged.connect(partial(self.setDcpParam, sld_box_radius, 'sld_box_radius'))
        #====================#
        sld_omega = QSlider(Qt.Horizontal, self)
        sld_omega.start = 0.01
        sld_omega.end = 1
        sld_omega.step = 100
        sld_omega.setFocusPolicy(Qt.NoFocus)
        sld_omega.setRange(sld_omega.start*sld_omega.step, sld_omega.end*sld_omega.step)
        sld_omega.setValue(Example.OMEGA*sld_omega.step)
        sld_omega.setTickPosition(QSlider.TicksBelow)
        sld_omega.valueChanged.connect(partial(self.setFloatSlider, sld_omega, val_omega))
        sld_omega.valueChanged.connect(partial(self.setDcpParam, sld_omega, 'sld_omega'))
        #====================#
        sld_var = QSlider(Qt.Horizontal, self)
        sld_var.start = 0.01
        sld_var.end = 2
        sld_var.step = 100
        sld_var.setTickInterval(20)
        sld_var.setFocusPolicy(Qt.NoFocus)
        sld_var.setRange(sld_var.start*sld_var.step, sld_var.end*sld_var.step)
        sld_var.setValue(Example.VARIANCE*sld_var.step)
        sld_var.setTickPosition(QSlider.TicksBelow)
        sld_var.valueChanged.connect(partial(self.setFloatSlider, sld_var, val_var))
        sld_var.valueChanged.connect(partial(self.setDcpParam, sld_var, 'sld_var'))
        #====================#
        sld_t0 = QSlider(Qt.Horizontal, self)
        sld_t0.start = 0.01
        sld_t0.end = 1
        sld_t0.step = 100
        sld_t0.setFocusPolicy(Qt.NoFocus)
        sld_t0.setRange(sld_t0.start*sld_var.step, sld_t0.end*sld_var.step)
        sld_t0.setValue(Example.T0*sld_t0.step)
        sld_t0.setTickPosition(QSlider.TicksBelow)
        sld_t0.valueChanged.connect(partial(self.setFloatSlider, sld_t0, val_t0))
        sld_t0.valueChanged.connect(partial(self.setDcpParam, sld_t0, 'sld_t0'))
        #====================#
        sld_t1 = QSlider(Qt.Horizontal, self)
        sld_t1.start = 0.01
        sld_t1.end = 1
        sld_t1.step = 100
        sld_t1.setFocusPolicy(Qt.NoFocus)
        sld_t1.setRange(sld_t1.start*sld_t1.step, sld_t1.end*sld_t1.step)
        sld_t1.setValue(Example.T1*sld_t1.step)
        sld_t1.setTickPosition(QSlider.TicksBelow)
        sld_t1.valueChanged.connect(partial(self.setFloatSlider, sld_t1, val_t1))
        sld_t1.valueChanged.connect(partial(self.setDcpParam, sld_t1, 'sld_t1'))
        #====================#
        sld_al_offset = QSlider(Qt.Horizontal, self)
        sld_al_offset.setFocusPolicy(Qt.NoFocus)
        sld_al_offset.setRange(1, 256)
        sld_al_offset.setValue(Example.AL_OFFSET)
        sld_al_offset.setTickPosition(QSlider.TicksBelow)
        sld_al_offset.valueChanged.connect(partial(self.setIntSlider, sld_al_offset, val_al_offset))
        sld_al_offset.valueChanged.connect(partial(self.setDcpParam, sld_al_offset, 'sld_al_offset'))
        #====================#


        ## Layout Setting
        # H1 - Images
        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(self.lbl_in)
        hbox1.setAlignment(self.lbl_in, Qt.AlignHCenter | Qt.AlignVCenter)
        #hbox1.addSpacing(10)
        hbox1.addStretch(1)
        hbox1.addWidget(self.lbl_out)
        hbox1.setAlignment(self.lbl_out, Qt.AlignHCenter | Qt.AlignVCenter)
        hbox1.addStretch(1)

        # H2 - Load buttons, silder, and texts.
        hbox2 = QHBoxLayout()
        hbox2.addSpacing(5)
        hbox2.addWidget(self.lbl_fileIn)
        hbox2.addSpacing(5)
        hbox2.addWidget(txt_min_radius)
        hbox2.addSpacing(5)
        hbox2.addWidget(val_min_radius)
        hbox2.addSpacing(5)
        hbox2.addWidget(sld_min_radius)
        hbox2.addSpacing(50)
        hbox2.addWidget(txt_box_radius)
        hbox2.addSpacing(5)
        hbox2.addWidget(val_box_radius)
        hbox2.addSpacing(5)
        hbox2.addWidget(sld_box_radius)
        hbox2.addSpacing(100)

        # H3 - silder, and texts.
        hbox3 = QHBoxLayout()
        hbox3.addSpacing(113)
        hbox3.addWidget(txt_omega)
        hbox3.addSpacing(30)
        hbox3.addWidget(val_omega)
        hbox3.addSpacing(5)
        hbox3.addWidget(sld_omega)
        hbox3.addSpacing(52)
        hbox3.addWidget(txt_var)
        hbox3.addSpacing(20)
        hbox3.addWidget(val_var)
        hbox3.addSpacing(5)
        hbox3.addWidget(sld_var)
        hbox3.addSpacing(100)

        # H4 - silder, and texts.
        hbox4 = QHBoxLayout()
        hbox4.addSpacing(113)
        hbox4.addWidget(txt_t0)
        hbox4.addSpacing(62)
        hbox4.addWidget(val_t0)
        hbox4.addSpacing(5)
        hbox4.addWidget(sld_t0)
        hbox4.addSpacing(53)
        hbox4.addWidget(txt_t1)
        hbox4.addSpacing(62)
        hbox4.addWidget(val_t1)
        hbox4.addSpacing(5)
        hbox4.addWidget(sld_t1)
        hbox4.addSpacing(100)

        # H5 - silder, and texts.
        hbox5 = QHBoxLayout()
        hbox5.addSpacing(113)
        hbox5.addWidget(txt_al_offset)
        hbox5.addSpacing(12)
        hbox5.addWidget(val_al_offset)
        hbox5.addSpacing(5)
        hbox5.addWidget(sld_al_offset)
        hbox5.addSpacing(638)

        # V1 - Overall
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addStretch(1)

        final_layout = vbox

        return final_layout

    def loadInitPixmap(self):
        # Load the default images
        img_input1 = cv2.imread('tiananmen_input.png', 1)
        img_input2 = cv2.imread('tiananmen_output.png', 1)
        myScaledPixmap1 = Example.npArrToPixmap(img_input1)
        myScaledPixmap2 = Example.npArrToPixmap(img_input2)

        self.img_load = img_input1

        return myScaledPixmap1, myScaledPixmap2

    def getInputImg(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', './', "Images (*.png *.xpm *.jpg *.bmp);;Text files (*.txt);;XML files (*.xml);;ALL (*)")

        if fname[0]:
	        # FREE the previous memories
            self.dcpFree()

            # Fail due to chinese encoding
            #self.img_load = cv2.imread('C:/Users/Desktop/新增資料夾/layout/canon3.png', 1)
            # Resovle by the following code, or use English path only.
            # np.fromfile - Construct an array from data in an image.
            # cv2.imdecode - Decode the data in the array.
            self.img_load = cv2.imdecode(np.fromfile(fname[0],dtype=np.uint8),-1)

            # DCP process
            self.dcpInit(self.img_load)
            img_dcp = self.dcpProcess(self.img_load)

            pix_input = Example.npArrToPixmap(self.img_load)
            pix_output = Example.npArrToPixmap(img_dcp)

            self.lbl_in.clear()
            self.lbl_out.clear()
            self.lbl_in.setPixmap(pix_input)
            self.lbl_out.setPixmap(pix_output)
        else:
            print("Loading file is canceled.")


    def setIntSlider(self, sld, val, sender):
        val.setText(str(sld.value()))

    def setFloatSlider(self, sld, val_txt, sender):
        val_float = Example.setValue(sld.value())
        val_txt.setText(str('{0:.2f}'.format(val_float)))

    def setDcpParam(self, sld, sender):
        self.sw.setter(sender, sld.value(), self)

        img_dcp = self.dcpProcess(self.img_load)
        pix_output = Example.npArrToPixmap(img_dcp)

        self.lbl_out.clear()
        self.lbl_out.setPixmap(pix_output)

    @staticmethod
    def npArrToPixmap(npArr):

        imgRGB = cv2.cvtColor(npArr, cv2.COLOR_BGR2RGB)
        height, width, channel = npArr.shape
        bytesPerLine = 3 * width
        qImg = QImage(imgRGB, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        myScaledPixmap = Example.limitSize(pixmap)

        # Delete
        del imgRGB
        del qImg

        return myScaledPixmap

    @staticmethod
    def setValue(value, _max_int=100):
        return value  / _max_int

    @staticmethod
    def limitSize(pixmap):
        _WID = 600
        _HEI = 450

        wid = pixmap.width()
        hei = pixmap.height()

        diff_wid = abs(wid - _WID)
        diff_hei = abs(hei - _HEI)

        if wid >= _WID:
            if hei >= _HEI:
                if diff_wid >= diff_hei:
                    wid = _WID
                    pixmap_s = pixmap.scaledToWidth(_WID, Qt.SmoothTransformation)
                else:
                    hei = _HEI
                    pixmap_s = pixmap.scaledToHeight(_HEI, Qt.SmoothTransformation)
            else:
                wid = _WID
                pixmap_s = pixmap.scaledToWidth(_WID, Qt.SmoothTransformation)
        else:
            hei = _HEI
            pixmap_s = pixmap.scaledToHeight(_HEI, Qt.SmoothTransformation)


        return pixmap_s

    class Switcher(object):
        def __init__(self):
            pass

        def setter(self, argument, value, outer):
            """Dispatch method"""
            method = getattr(self, argument, lambda: "nothing")
            return method(value, outer)

        def sld_min_radius(self, value, outer):
            outer.dcp_param.min_radius = value
            # return print("sld_min_radius: ", value)

        def sld_box_radius(self, value, outer):
            outer.dcp_param.box_radius = value
            # return print("sld_box_radius: ", value)

        def sld_omega(self, value, outer):
            outer.dcp_param.omega = Example.setValue(value)
            # return print("sld_omega: ", Example.setValue(value))

        def sld_var(self, value, outer):
            outer.dcp_param.sky_var = Example.setValue(value)
            # return print("sld_var: ", Example.setValue(value))

        def sld_t0(self, value, outer):
            outer.dcp_param.t0 = Example.setValue(value)
            # return print("sld_t0: ", Example.setValue(value))

        def sld_t1(self, value, outer):
            outer.dcp_param.t1 = Example.setValue(value)
            # return print("sld_t1: ", Example.setValue(value))

        def sld_al_offset(self, value, outer):
            outer.dcp_param.airlight_offset = value
            # return print("sld_al_offset: ", value)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ex = Example()
    app.exec_()
