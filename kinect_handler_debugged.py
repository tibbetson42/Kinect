
import thread
import itertools
import ctypes

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

KINECTEVENT = pygame.USEREVENT
#DEPTH_WINSIZE = 320,240
DEPTH_WINSIZE = 640,480
VIDEO_WINSIZE = 640,480
pygame.init()
# Colors #---------------------------------------------
SKELETON_COLORS = [ THECOLORS["red"],   THECOLORS["blue"],  THECOLORS["green"],
                    THECOLORS["orange"],THECOLORS["purple"],THECOLORS["yellow"]
                  ]
# Body Parts #-----------------------------------------
SPINE = (JointId.HipCenter, JointId.Spine, JointId.ShoulderCenter, JointId.Head)

LEFT_ARM =  (JointId.ShoulderCenter, JointId.ShoulderLeft, JointId.ElbowLeft,
             JointId.WristLeft, JointId.HandLeft)

RIGHT_ARM = (JointId.ShoulderCenter, JointId.ShoulderRight, JointId.ElbowRight,
             JointId.WristRight, JointId.HandRight)

LEFT_LEG =  (JointId.HipCenter, JointId.HipLeft, JointId.KneeLeft,
             JointId.AnkleLeft, JointId.FootLeft)

RIGHT_LEG = (JointId.HipCenter, JointId.HipRight, JointId.KneeRight,
             JointId.AnkleRight, JointId.FootRight)
print('initialization complete')

class kinect_handler:
    def __init__(self,screen):
        # Flags and data
        self.VIDEO = False
        self.DEPTH = True
        self.DRAW_SKELETONS = True
        self.SEATED = True
        self.info = pygame.display.Info()
        self.thread = thread.allocate()
        self.screen = screen
        self.skeletons = None
        self.done = False
        # Buffer information/address functions for self.copy_to_screen
        # recipe to get address of surface(or screen): http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
        if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
            print('32bit')
            self.Py_ssize_t = ctypes.c_int
        elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
            print('64bit')
            self.Py_ssize_t = ctypes.c_int64
        else:
           raise TypeError("Cannot determine type of Py_ssize_t")
        self._PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
        self._PyObject_AsWriteBuffer.restype = ctypes.c_int
        self._PyObject_AsWriteBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(self.Py_ssize_t)]

    def copy_to_screen(self,frame):
       buffer_interface = self.screen.get_buffer()
       address = ctypes.c_void_p()
       size = self.Py_ssize_t()
       self._PyObject_AsWriteBuffer(buffer_interface, ctypes.byref(address), ctypes.byref(size))
       #print(size.value)
       bytes = (ctypes.c_byte * size.value).from_address(address.value)
       bytes.object = buffer_interface
       frame.image.copy_bits(bytes)

    def draw_skeletons(self):
        for idx,skeleton in enumerate(self.skeletons):
            if skeleton.tracking_state:
                print(idx),
                print(skeleton.get_skeleton_positions()[JointId.Head]),
                print(' ')
                color =  SKELETON_COLORS[idx]
                self.draw_bodypart(skeleton,  SPINE,     color)
                self.draw_bodypart(skeleton,  LEFT_ARM,  color)
                self.draw_bodypart(skeleton,  RIGHT_ARM, color)
                if not self.SEATED:
                    self.draw_bodypart(skeleton,  LEFT_LEG,  color)
                    self.draw_bodypart(skeleton,  RIGHT_LEG, color)
            else:
                print(idx),
                print(' not tracked'),
                print(' ')

    def draw_bodypart(self,skeleton, Joint_Indexes, color, width = 3):
        thisJointXYZ = skeleton.SkeletonPositions[Joint_Indexes[0]]
        if Joint_Indexes[0] == JointId.Head:
            pygame.draw.circle(self.screen, color, thisJointPxls,10*width,0)
        for joint in itertools.islice(Joint_Indexes,1,None):
            nextJointXYZ = skeleton.SkeletonPositions[joint.value]
            thisJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(thisJointXYZ, self.info.current_w, self.info.current_h)
            nextJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(nextJointXYZ, self.info.current_w, self.info.current_h)
            pygame.draw.line(self.screen, color, thisJointPxls,nextJointPxls,width)
            if joint.value == JointId.Head:
                pygame.draw.circle(self.screen, color, (int(nextJointPxls[0]),int(nextJointPxls[1])),10*width,0)
            thisJointXYZ = nextJointXYZ

    def depth_frame_ready(self,frame):
        if self.VIDEO:
            return
        with self.thread:
            self.copy_to_screen(frame)
            if self.skeletons is not None and self.DRAW_SKELETONS:
                #import pdb; pdb.set_trace()
                self.draw_skeletons()
                #pass
            pygame.display.update()

    def video_frame_ready(self,frame):
        if self.DEPTH:
            return
        with self.thread:
            self.copy_to_screen(frame)
            if self.skeletons is not None and self.DRAW_SKELETONS:
                self.draw_skeletons()
                #pass
            pygame.display.update()

    def post_skeleton_frame(self,frame):
        try:
            #print('post frame')
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons = frame.SkeletonData))
        except:
            print('error: event queue full')
            pass

    def event_decision(self,event):
        if event.type == pygame.QUIT:
            print('quit',event.type)
            self.done = True
        elif event.type == KINECTEVENT:
            #print('kinect frame',event.type)
            self.skeletons = event.skeletons
            #if self.DRAW_SKELETONS:
                #self.draw_skeletons()
                #pygame.display.update()
        elif event.type == KEYDOWN:
            print('quit',event.type)
            if event.key == K_ESCAPE:
                self.done = True
            elif event.key == K_d:
                print('depth',event.type)
                with self.thread:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    self.DEPTH = True
                    self.VIDEO = False
            elif event.key == K_v:
                print('video',event.type)
                with self.thread:
                    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,16)
                    self.DEPTH = False
                    self.VIDEO = True
            elif event.key == K_s:
                print('skeletons {}'.format(not self.DRAW_SKELETONS,event.type))
                self.DRAW_SKELETONS = not self.DRAW_SKELETONS
            elif event.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif event.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif event.key == K_x:
                kinect.camera.elevation_angle = 2

if __name__ == '__main__':
    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
    handler = kinect_handler(screen)
    pygame.display.set_caption('Python Kinect Demo')
    screen.fill(THECOLORS["black"])
    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    kinect.skeleton_frame_ready += handler.post_skeleton_frame
    kinect.depth_frame_ready += handler.depth_frame_ready
    kinect.video_frame_ready += handler.video_frame_ready
    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)

    # print('Controls: ')
    # print('     h - Show this message')
    # print('     d - Switch to depth view')
    # print('     v - Switch to video view')
    # print('     s - Toggle displaing of the skeleton')
    # print('     u - Increase elevation angle')
    # print('     j - Decrease elevation angle')

    # main game loop
    while not handler.done:
        e = pygame.event.wait()
        handler.info = pygame.display.Info()
        handler.event_decision(e)
