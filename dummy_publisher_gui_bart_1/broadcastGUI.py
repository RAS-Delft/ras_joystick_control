'''
 Developed in Researchlab Autonomous Shipping (RAS) Delft
 Department Maritime and Transport Technology of faculty 3mE, TU Delft. 
 https://rasdelft.nl/nl/

 Bart Boogmans
 bartboogmans@hotmail.com
'''

import sys
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import rclpy
import math
import os 
import numpy as np
import time
from PyQt5.QtCore import Qt, QTimer, QPointF, QPoint
from PyQt5.QtGui import QPixmap, QColor, QPolygonF, QPen, QBrush, QPainter, QPolygon, QIcon
from PyQt5.QtWidgets import (
	QApplication,
	QLabel,
	QMainWindow,
	QPushButton,
	QVBoxLayout,
	QHBoxLayout,
	QWidget,
	QSlider,
	QGraphicsView,
	QGraphicsScene,
	QLineEdit,
	QCheckBox,
)

import ras_ros_core_control_modules.tools.geometry_tools as geometry_tools
import ras_ros_core_control_modules.tools.titoneri_parameters as titoneri_parameters

DRAWSCALE = 250 # pixels per meter

class plotColorPalette():
	pen_x = QPen(QColor(255,0,0),2)
	pen_y = QPen(QColor(0,255,0),2)
	pen_z = QPen(QColor(0,0,255),2)

	RAS_TN_DB = QPen(QColor(0, 96, 186),3)
	RAS_TN_GR = QPen(QColor(44, 171, 5),3)
	RAS_TN_YE = QPen(QColor(235, 227, 0),3)
	RAS_TN_PU = QPen(QColor(206, 0, 224),3)
	RAS_TN_LB = QPen(QColor(28, 164, 255),3)
	RAS_TN_OR = QPen(QColor(255, 149, 0),3)

	default_vessel_hull = QPen(QColor(0, 0, 0),3)
	vessel_hull_disabled = QPen(QColor(20, 20, 20),3)

	thrusters = QPen(QColor(0, 0, 0),2)

class plotTree2d():
	""" Class to assist in drawing 2d objects in a tree structure. 
		The root of the tree should not have a parent.
	"""
	def __init__(self, line:np.ndarray=None,parent:'plotTree2d'=None, brush:QBrush=None, pen:QPen=None, translation:np.ndarray=np.array([0.0,0.0]), rotation:float=0.0,inheritLayout:'plotTree2d'=None,name:str=None):
		self.children = []
		self.name = name
		self.parent = parent

		# Set default layout
		self.line = None
		self.brush = None
		self.pen = QPen(QColor(0,0,0))

		# Inherit layout from referenced object if given
		if inheritLayout is not None:
			self.line = inheritLayout.line
			self.brush = inheritLayout.brush
			self.pen = inheritLayout.pen
		
		# Set specified layout
		if line is not None:
			self.line = line
		if brush is not None:
			self.brush = brush
		if pen is not None:
			self.pen = pen

		# Set translation and rotation
		self.translation = translation
		self.rotation = rotation

		if parent is not None:
			parent.addChild(self)

	def addChild(self, child:'plotTree2d'):
		# Check if the object added is not the root of itself
		if self.getRoot() is child:
			raise ValueError("Cannot add a root to child of itself to avoid recursive plotting")
		else:
			# chech if child is not already a child
			if child in self.children:
				raise ValueError("Cannot add a child that is already a child")
			else:
				# Check if child has a parent
				if child.parent is not None:
					# Remove child from old parent
					if child in child.parent.children:
						child.parent.children.remove(child)
				self.children.append(child)
				child.parent = self

	def getRoot(self):
		if self.parent is None:
			return self
		else:
			return self.parent.getRoot()

	def draw(self, painter:QPainter):
		# Draw self
		if self.line is not None:

			# Set the pen and brush
			painter.setPen(self.pen)
			if self.brush is not None:
				painter.setBrush(self.brush)
			else:
				painter.setBrush(QBrush(Qt.NoBrush))

			# Rotate the hull outline, translate and scale to pixel coordinates
			outline = (np.matmul(rotation_matrix_2d(self.getGlobalRotation()),self.line)+self.getGlobalTranslation()[:, np.newaxis])*DRAWSCALE

			# make a list of QPoint objects and translate to center
			outline_qpoint = []
			for i in range(len(outline[0])):
				point = QPoint(outline[0][i],outline[1][i])
				outline_qpoint.append(point)
			
			# Draw the outline
			painter.drawPolygon(QPolygon(outline_qpoint))

		# Draw children (if any)
		for child in self.children:
			child.draw(painter)

	def getGlobalTranslation(self):
		if self.parent is not None:
			# My coordinate system is expressed in my parent's local coordinate system
			return self.parent.getGlobalTranslation() + np.matmul(rotation_matrix_2d(self.parent.getGlobalRotation()),self.translation)
		else:
			# I am root, thus my coordinate system is expressed in the global coordinate system
			return self.translation
	
	def getGlobalRotation(self):
		if self.parent is not None:
			# My coordinate system is expressed in my parent's local coordinate system
			return self.parent.getGlobalRotation() + self.rotation
		else:
			# I am root, thus my coordinate system is expressed in the global coordinate system
			return self.rotation

def rotation_matrix_2d(theta):
	""" Returns a 2D rotation matrix. """
	return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

class GuiNode(Node):
	""" Manages ROS2 communication."""

	def __init__(self,parent_,pub_frequency=10.0):
		self.parent = parent_
		super().__init__('manual_control_gui_node')
		self.pub_actuation = None
		self.subscriber1 = None
		self.timer_actuation = self.create_timer(1.0/pub_frequency, self.timer_callback1)
		self.num_msgs_received=0

	def timer_callback1(self):
		# If there is a publisher
		if self.pub_actuation:
			msg = Float32MultiArray()
			msg.data = [float(self.parent.slider_rpm_SB.value()), float(self.parent.slider_rpm_PS.value()), float(self.parent.slider_angle_SB.value()), float(self.parent.slider_angle_PS.value()), float(self.parent.slider_bow.value())]
			self.pub_actuation.publish(msg)
		
	def subscriber_callback1(self, msg):
		self.num_msgs_received+=1

	def startActuationBroadcast(self, vesselID):
		self.pub_actuation= self.create_publisher(Float32MultiArray, vesselID + '/reference/actuation', 10)

	def stopActuationBroadcast(self):
		self.destroy_publisher(self.pub_actuation)
		self.pub_actuation = None

class Vesselplotter():
	"""Some tools that help drawing a vessel and n actuators on a QGraphicsScene."""
	def __init__(self,targetWidget:QMainWindow,  type='titoneri'):
		# case specific parameters
		self.target = targetWidget

		# Load vessel parameters
		self.vessel_outline = titoneri_parameters.vessel_outline()
		self.thruster_outlines = titoneri_parameters.thruster_outlines()
		self.thruster_positions = titoneri_parameters.thruster_locations()
		thruster_angle0 = titoneri_parameters.angles_zero()

		#self.actuation2force = titoneri_parameters.titoNeriThrusterRelation('ThrustToActuator')

		self.vessel_rotation = -np.pi/2
		
		self.hullplotter = plotTree2d(line=self.vessel_outline,rotation=float(self.vessel_rotation),name='hull',pen=plotColorPalette.vessel_hull_disabled,brush=QBrush(QColor(110, 110, 110)))

		self.thrusterSBplotter = plotTree2d(line=self.thruster_outlines[0],translation=self.thruster_positions[0],rotation=float(thruster_angle0[0]),parent=self.hullplotter,name='thrusterSB',pen=plotColorPalette.thrusters,brush=QBrush(QColor(100, 100, 100)))
		self.thrusterPSplotter = plotTree2d(line=self.thruster_outlines[1],translation=self.thruster_positions[1],rotation=float(thruster_angle0[1]),parent=self.hullplotter,name='thrusterPS',pen=plotColorPalette.thrusters,brush=QBrush(QColor(100, 100, 100)))
		self.bowthrusterplotter = plotTree2d(line=self.thruster_outlines[2],translation=self.thruster_positions[2],rotation=float(thruster_angle0[2]),parent=self.hullplotter,name='bowthruster',pen=plotColorPalette.thrusters,brush=QBrush(QColor(100, 100, 100)))

		self.hullXaxisplotter = plotTree2d(line=np.array([[0,0.1],[0,0]]),parent=self.hullplotter,pen=plotColorPalette.pen_x,name='hullXaxis')
		self.hullYaxisplotter = plotTree2d(line=np.array([[0,0],[0,0.1]]),parent=self.hullplotter,pen=plotColorPalette.pen_y,name='hullYaxis')

		self.thrusterSBXaxisplotter = plotTree2d(line=np.array([[0,0.05],[0,0]]),parent=self.thrusterSBplotter,pen=plotColorPalette.pen_x,name='thrusterSBXaxis')
		self.thrusterSBYaxisplotter = plotTree2d(line=np.array([[0,0],[0,0.05]]),parent=self.thrusterSBplotter,pen=plotColorPalette.pen_y,name='thrusterSBYaxis')
		self.thusterPSXaxisplotter = plotTree2d(line=np.array([[0,0.05],[0,0]]),parent=self.thrusterPSplotter,pen=plotColorPalette.pen_x,name='thrusterPSXaxis')
		self.thusterPSYaxisplotter = plotTree2d(line=np.array([[0,0],[0,0.05]]),parent=self.thrusterPSplotter,pen=plotColorPalette.pen_y,name='thrusterPSYaxis')

		self.draw_boundary_offsets = [[50,-50],[36,-110]] # [[dx_min, dx_max],[dy_min, dy_max]] away from left, right, top and bottom of the window
		self.set_draw_boundaries()
	
	def set_draw_boundaries(self):
		""" Sets the boundaries of the drawing area. """
		self.draw_boundaries = [[self.draw_boundary_offsets[0][0],self.target.width() + self.draw_boundary_offsets[0][1]],[self.draw_boundary_offsets[1][0],self.target.height()+self.draw_boundary_offsets[1][1]]]
	
	def drawVessel(self, u:np.ndarray,a:np.ndarray,painter:QPainter,main_window:QMainWindow):
		""" Draws the vessel and actuators in the screen. 
			u: np.ndarray([aft_rpm_sb, _aft_rpm_ps, bowthruster_usage])
			a: np.ndarray([aft_sb_angle, aft_ps_angle, bow_angle])
			* u and a need to be of the length of the number of actuators. """

		# Set the vessel rotation
		self.hullplotter.rotation = self.vessel_rotation

		# Set the thruster rotation
		self.thrusterSBplotter.rotation = np.radians(a[0])
		self.thrusterPSplotter.rotation = np.radians(a[1])

		# plot the vessel
		self.hullplotter.draw(painter)

class Window(QMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.clicksCount = 0
		self.setupUi()
		self.vesselplotter = Vesselplotter(self)
		
		rclpy.init(args=None)
		self.node = GuiNode(self)
	
		self.spinROSTimer = QTimer()
		self.spinROSTimer.setInterval(1) # 1 ms = 1000 Hz (the minimum is 1ms)
		# Note that the total maximum amount of ros events that can be processed per second is limited by the spinROSTimer interval.
		self.spinROSTimer.timeout.connect(self.spinROSTimerCallback)
		self.spinROSTimer.start()
		
		self.drawTimer = QTimer()
		self.drawTimer.setInterval(50) # 50 ms = 20 Hz
		self.drawTimer.timeout.connect(self.drawTimedCallback)
		self.drawTimer.start()

		self.pen = QPen(QColor(0,0,0))					  # set lineColor
		self.pen.setWidth(3)											# set lineWidth
		self.brush = QBrush(QColor(255,255,255,255))		# set fillColor  

	def keyPressEvent(self, event):
		""" check if enter has been pressed while in the vessel ID field.
		If so, start/stop the actuation broadcast and toggle the button text. """
		if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
			self.startBtnClicked()

	def resizeEvent(self, event):
		self.vesselplotter.set_draw_boundaries()
		vesselplotcenter = np.array([(self.vesselplotter.draw_boundaries[0][0]+self.vesselplotter.draw_boundaries[0][1])/2,(self.vesselplotter.draw_boundaries[1][0]+self.vesselplotter.draw_boundaries[1][1])/2])/DRAWSCALE
		self.vesselplotter.hullplotter.translation = vesselplotcenter
		QMainWindow.resizeEvent(self, event)

	def checkVesselColor(self):
		vesselID = self.vesselIDField.text()
	 	# check if class plotColorPalette has a pen with the name of the vesselID
		if hasattr(plotColorPalette, vesselID):
			self.vesselplotter.hullplotter.pen = getattr(plotColorPalette, vesselID)
		else:
			self.vesselplotter.hullplotter.pen = plotColorPalette.default_vessel_hull

	def drawTimedCallback(self):
		self.update()
	
	def paintEvent(self, event):
		painter = QPainter(self)
		painter.setPen(QPen(Qt.black,  2, Qt.SolidLine))
		#painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
		cornerpoints = [
			QPoint(self.vesselplotter.draw_boundaries[0][0], self.vesselplotter.draw_boundaries[1][0]),
			QPoint(self.vesselplotter.draw_boundaries[0][1], self.vesselplotter.draw_boundaries[1][0]),
			QPoint(self.vesselplotter.draw_boundaries[0][1], self.vesselplotter.draw_boundaries[1][1]),
			QPoint(self.vesselplotter.draw_boundaries[0][0], self.vesselplotter.draw_boundaries[1][1]),
			]
		#poly = QPolygon(cornerpoints)
		#painter.drawPolygon(poly)

		self.vesselplotter.drawVessel(np.array([self.slider_rpm_SB.value(), self.slider_rpm_PS.value(),self.slider_bow.value()]),np.array([self.slider_angle_SB.value(), self.slider_angle_PS.value(),0]),painter,self)
	
	def slidervalue_changed_by_user(self,slider:QSlider, label: QLabel):
		""" Sets the label value to the slider value. """
		# If the slider is within 1.5% if the total range, snap to zero
		sliderRange = slider.maximum() - slider.minimum()
		if abs(slider.value()) < sliderRange*0.015:
			slider.setValue(0)

		# Set the label value to the slider value
		label.setText(str(slider.value()))

	def joysticktoggle_changed_by_user(self):
		""" Toggles joystick control on and off. """
		if self.joysticktoggle.checkState() == Qt.Checked:
			print("Joystick control is on")
			# Joystick control is on
			self.slider_rpm_SB.setEnabled(False)
			self.slider_rpm_PS.setEnabled(False)
			self.slider_angle_SB.setEnabled(False)
			self.slider_angle_PS.setEnabled(False)
			self.slider_bow.setEnabled(False)
		else:
			print("Joystick control is off")
			# Joystick control is off
			self.slider_rpm_SB.setEnabled(True)
			self.slider_rpm_PS.setEnabled(True)
			self.slider_angle_SB.setEnabled(True)
			self.slider_angle_PS.setEnabled(True)
			self.slider_bow.setEnabled(True)

	def setupUi(self):
		self.setWindowTitle("ROS2 manual control interface")
		self.resize(450, 535)
		self.centralWidget = QWidget()
		self.setCentralWidget(self.centralWidget)

		# Set minimum window size
		self.setMinimumSize(350,417)

		# Set component color schemes
		self.backgroundcolor = QColor()
		self.backgroundcolor.setRgb(70, 70, 70)
		self.fieldcolor = QColor()
		self.fieldcolor.setRgb(85, 85, 85)
		self.rasYellow = QColor()
		self.rasYellow.setRgb(255, 228, 54)
		
		# Set background color
		p = self.palette()
		p.setColor(self.backgroundRole(), self.backgroundcolor)
		self.setPalette(p)

		# Create sliders and set minimum and maximum properties
		self.slider_rpm_SB = QSlider(Qt.Vertical, self)
		self.slider_rpm_SB.setTickInterval(1200)
		self.slider_rpm_SB.setTickPosition(QSlider.TicksRight)
		self.slider_rpm_SB.setFixedWidth(40)

		self.slider_rpm_PS = QSlider(Qt.Vertical, self)
		self.slider_rpm_PS.setTickInterval(1200)
		self.slider_rpm_PS.setTickPosition(QSlider.TicksLeft)
		self.slider_rpm_PS.setFixedWidth(40)
		
		self.slider_angle_SB = QSlider(Qt.Horizontal, self)
		self.slider_angle_SB.setTickInterval(10)

		self.slider_angle_PS = QSlider(Qt.Horizontal, self)
		self.slider_angle_PS.setTickInterval(10)
		self.slider_bow = QSlider(Qt.Horizontal, self)
		self.slider_bow.setTickInterval(20)
		self.slider_bow.setFixedWidth(150)

		# Create labels for slider intensities
		self.label_rpm_SB = QLabel("0", self)
		self.label_rpm_SB.setAlignment(Qt.AlignLeft)
		self.label_rpm_PS = QLabel("0", self)
		self.label_rpm_PS.setAlignment(Qt.AlignRight)
		self.label_angle_SB = QLabel("0", self)
		self.label_angle_SB.setAlignment(Qt.AlignLeft)
		self.label_angle_PS = QLabel("0", self)
		self.label_angle_PS.setAlignment(Qt.AlignRight)
		self.label_bow = QLabel("0", self)
		self.label_bow.setAlignment(Qt.AlignLeft)
		self.label_rpm_SB.setMinimumWidth(40)
		self.label_rpm_PS.setMinimumWidth(40)
		self.label_angle_SB.setMinimumWidth(30)
		self.label_angle_PS.setMinimumWidth(30)


		# Align slider bow to the top
		self.layout_slideronly = QVBoxLayout()
		self.layout_slideronly.addWidget(self.slider_bow)
		self.layout_slideronly.setAlignment(Qt.AlignTop)

		# Set slider-label layout combinations
		layout_rpm_SB = QVBoxLayout()
		layout_rpm_SB.addWidget(self.label_rpm_SB)
		layout_rpm_SB.addWidget(self.slider_rpm_SB)
		layout_rpm_PS = QVBoxLayout()
		layout_rpm_PS.addWidget(self.label_rpm_PS)
		layout_rpm_PS.addWidget(self.slider_rpm_PS)
		layout_bow = QHBoxLayout()
		spacer1 = QLabel("", self)
		spacer1.setMinimumWidth(40)
		layout_bow.addWidget(QLabel("", self)) # Spacer
		layout_bow.addLayout(self.layout_slideronly)
		layout_bow.addWidget(self.label_bow)
		self.label_bow.setMinimumWidth(40)

		# Set slider limits
		rpm_max = 2400 # rpm
		angle_max = 120 # degrees
		bow_pwr_max = 100 # percent
		self.slider_rpm_SB.setMinimum(-rpm_max)
		self.slider_rpm_SB.setMaximum(rpm_max)
		self.slider_rpm_PS.setMinimum(-rpm_max)
		self.slider_rpm_PS.setMaximum(rpm_max)
		self.slider_angle_SB.setMinimum(-angle_max)
		self.slider_angle_SB.setMaximum(angle_max)
		self.slider_angle_PS.setMinimum(-angle_max)
		self.slider_angle_PS.setMaximum(angle_max)
		self.slider_bow.setMinimum(-bow_pwr_max)
		self.slider_bow.setMaximum(bow_pwr_max)

		# Set slider value changed callback
		self.slider_rpm_SB.valueChanged.connect(lambda: self.slidervalue_changed_by_user(self.slider_rpm_SB, self.label_rpm_SB))
		self.slider_rpm_PS.valueChanged.connect(lambda: self.slidervalue_changed_by_user(self.slider_rpm_PS, self.label_rpm_PS))
		self.slider_angle_SB.valueChanged.connect(lambda: self.slidervalue_changed_by_user(self.slider_angle_SB, self.label_angle_SB))
		self.slider_angle_PS.valueChanged.connect(lambda: self.slidervalue_changed_by_user(self.slider_angle_PS, self.label_angle_PS))
		self.slider_bow.valueChanged.connect(lambda: self.slidervalue_changed_by_user(self.slider_bow, self.label_bow))

		# Make vessel drawing area
		self.vessel_scene = QGraphicsScene(0, 0, 300, 400)
		self.vessel_view = QGraphicsView(self.vessel_scene, self)

		# Hide vessel drawing scene fully
		self.vessel_view.setVisible(False)

		# Set minimal size of vessel drawing area
		self.vessel_view.setMinimumSize(340, 404)
		
		# Make vessel ID section components
		self.vesselIDField = QLineEdit("RAS_TN_DB", self)
		self.btnStart = QPushButton("Start", self)
		self.btnStart.clicked.connect(self.startBtnClicked)
		self.joysticktoggle = QCheckBox("Joystick", self)
		self.joysticktoggle.stateChanged.connect(self.joysticktoggle_changed_by_user)
		self.IDlabel = QLabel("Vessel ID:", self)

		# Make notes and logo section components
		self.statuslabel = QLabel("", self)
		self.rasLogo = QLabel(self)
		self.rasLogo.setPixmap(QPixmap(os.path.dirname(os.path.realpath(__file__)) + "/raslogo1.png").scaledToHeight(30))
		self.rasLogo.setAlignment(Qt.AlignRight)

		# Set background of drawing area and textboxes
		self.vessel_scene.setBackgroundBrush(self.fieldcolor)
		self.vesselIDField.setStyleSheet("background-color: rgb(85, 85, 85); color: rgb(255, 255, 255);")
		self.btnStart.setStyleSheet("background-color: rgb(85, 85, 85); color: rgb(255, 255, 255);")

		# Set the layout
		layout = QVBoxLayout() # Main layout
		layout_tophalf = QHBoxLayout() # Top half of main layout
		layout_bottomhalf = QVBoxLayout() # Bottom half of main layout
		layout.addLayout(layout_tophalf)
		layout.addLayout(layout_bottomhalf)
		layout_visual_bowsliderbox = QVBoxLayout() # Layout for vessel drawing and bow slider
		layout_bowsliderspacer = QHBoxLayout() # Layout for bow slider and spacer
		layout_bowsliderspacer.addLayout(layout_bow)
		layout_visual_bowsliderbox.addLayout(layout_bowsliderspacer)
		layout_visual_bowsliderbox.addWidget(self.vessel_view)
		layout_tophalf.addLayout(layout_rpm_PS)
		layout_tophalf.addLayout(layout_visual_bowsliderbox)
		layout_tophalf.addLayout(layout_rpm_SB)
		layout_rudderhorizontal = QHBoxLayout() # Layout for rudder sliders and labels
		layout_rudderhorizontal.addWidget(self.label_angle_PS)
		layout_rudderhorizontal.addWidget(self.slider_angle_PS)
		layout_rudderhorizontal.addWidget(self.slider_angle_SB)
		layout_rudderhorizontal.addWidget(self.label_angle_SB)
		layout_bottomhalf.addLayout(layout_rudderhorizontal)
		layout_vesselIDsection = QHBoxLayout() # Layout for vessel ID field and start button
		layout_vesselIDsection.addWidget(self.IDlabel)
		layout_vesselIDsection.addWidget(self.vesselIDField)
		layout_vesselIDsection.addWidget(self.btnStart)
		layout_vesselIDsection.addWidget(self.joysticktoggle)
		layout_bottomhalf.addLayout(layout_vesselIDsection)
		layout_notes_and_logo = QHBoxLayout() # Layout for status label and logo
		layout_notes_and_logo.addWidget(self.statuslabel)
		layout_notes_and_logo.addWidget(self.rasLogo)
		layout_bottomhalf.addLayout(layout_notes_and_logo)
		self.centralWidget.setLayout(layout)

	def startBtnClicked(self):
		""" Toggles between on and off state when clicked."""
		if self.btnStart.text() == "Start":
			self.startROS()
		else:
			self.stopROS()

	def startROS(self):
		self.btnStart.setText("Stop")
		self.btnStart.setStyleSheet("background-color: rgb(115, 115, 115); color: rgb(255, 255, 255);")
		self.vesselIDField.setStyleSheet("background-color: rgb(85, 85, 85); color: rgb(160, 160, 160);")
		self.vesselIDField.setDisabled(True)
		self.checkVesselColor()
		self.node.startActuationBroadcast(self.vesselIDField.text())
	
	def stopROS(self):
		self.btnStart.setText("Start")
		self.btnStart.setStyleSheet("background-color: rgb(85, 85, 85); color: rgb(255, 255, 255);")
		self.vesselIDField.setStyleSheet("background-color: rgb(85, 85, 85); color: rgb(255, 255, 255);")
		self.vesselIDField.setDisabled(False)
		self.node.stopActuationBroadcast()
		self.vesselplotter.hullplotter.pen = plotColorPalette.vessel_hull_disabled
	
	def spinROSTimerCallback(self):
		""" This function is called every time the spinROSTimer times out.
			It is used to process ros events. """
		rclpy.spin_once(self.node, timeout_sec=0.0001)
		# Note that the timeout here is important to not let the program wait until the next ros event, but move on if there are none. 
		# Is spin_until_future_complete better? https://docs.ros2.org/foxy/api/rclpy/api/scheduler.html#rclpy.scheduler.spin_until_future_complete

def main(args=None):
	app = QApplication(sys.argv)
	win = Window()
	win.show()
	shutdownstate = app.exec() # This causes the app to run until the window is closed
	rclpy.shutdown() # This causes the ros node to shutdown, avoiding errors when the program is closed
	sys.exit(shutdownstate) # IDK what the shutdownstate is good for. It is not necessary to pass it to sys.exit() but others do it so I do it too. Baah Im a sheep.

if __name__ == "__main__":
	main()