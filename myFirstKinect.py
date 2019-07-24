############ BROKKKKKEN

# DO NOT USE
############ BROKKKKKEN
############ BROKKKKKEN
############ BROKKKKKEN
############ BROKKKKKEN
############ BROKKKKKEN
############ BROKKKKKEN
############ BROKKKKKEN

import thread
import itertools
import ctypes
# Import PyKinect Library
import pykinect
from pykinect import nui
from pykinect.nui import JointId # Id constants for various limbs. 0 = Hip center

import pygame #?????
from pygame.color import THECOLORS # color dictionary
from pygame.locals import * #???????????
#Import my Initialization
#import myFirstKinect_config as cf
exec(open('myFirstKinect_INIT.py').read())
###============================================================================
## Function Defintions ##--------------------------------------------
# Resolution #-----------------------------------------
class kinect_handler:
    def __init__(self):#,screen,kinect = nui.Runtime()):
        self.VIDEO_COLOR = False
        self.VIDEO_DEPTH = True
        self.DRAW_SKELETONS = True
        #
        self.thread = thread.allocate()
        #
        self.screen = pygame.display.set_mode(DEPTH_RESOLUTION,0,16)
        self.screen.fill(THECOLORS["black"])
        self.skeletons = None
        #
        self.kinect = nui.Runtime()

        self.kinect.skeleton_engine_enabled = True
        self.info = pygame.display.Info()

        self.kinect.skeleton_frame_ready += self.post_frame
        self.kinect.depth_frame_ready += self.depth_frame_ready
        self.kinect.video_frame_ready += self.color_frame_ready

        if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
            print('32bit')
            self.Py_ssize_t = ctypes.c_int
        elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
            print('64bit')
            self.Py_ssize_t = ctypes.c_int64
        else:
           raise TypeError("Cannot determine type of Py_ssize_t")
        #print(self.Py_ssize_t())
        self._PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
        self._PyObject_AsWriteBuffer.restype = ctypes.c_int
        self._PyObject_AsWriteBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(self.Py_ssize_t)]


    def copy_to_screen(self,frame):
        #print(frame.image.pitch)
        buffer = self.screen.get_buffer()
        #print(buffer)
        address = ctypes.c_void_p()
        #print(self.Py_ssize_t())
        size = self.Py_ssize_t()
        self._PyObject_AsWriteBuffer(buffer,ctypes.byref(address),ctypes.byref(size))
        bytes = (ctypes.c_byte * size.value).from_address(address.value)
        bytes.object = buffer
        frame.image.copy_bits(bytes)
        return

    def draw_skeletons(self,skeletons):
        print('skele draw')
        for idx,skeleton in enumerate(skeletons):
            color =  SKELETON_COLORS[idx]
            self.draw_bodypart(skeleton,  SPINE,     color)
            self.draw_bodypart(skeleton,  LEFT_ARM,  color)
            self.draw_bodypart(skeleton,  RIGHT_ARM, color)
            self.draw_bodypart(skeleton,  LEFT_LEG,  color)
            self.draw_bodypart(skeleton,  RIGHT_LEG, color)

    def draw_bodypart(self, skeleton, joint_Indexes, color, width = 3):
        thisJointXYZ = skeleton.SkeletonPositions[joint_Indexes[0]]
        if joint_Indexes[0] == JointId.Head:
            pygame.draw.circle(self.screen, color, thisJointPxls,10*width,0)
        for joint in itertools.islice(Joint_Indexes,1,None):
            nextJointXYZ = skeleton.SkeletonPositions[joint.value]
            thisJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(thisJointXYZ,self.info.current_w,self.info.current_h)
            nextJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(nextJointXYZ,self.info.current_w,self.info.current_h)
            pygame.draw.line(self.screen, color, thisJointPxls,nextJointPxls,width)
            if joint.value == JointId.Head:
                pygame.draw.circle(self.screen, color, thisJointPxls,10*width,0)
            thisJointXYZ = nextJointXYZ

    def depth_frame_ready(self,frame):
        #import pdb; pdb.set_trace()
        if not self.VIDEO_DEPTH:
            return
        print('depth printing')
        with self.thread:
            self.copy_to_screen(frame)
            print(self.skeletons,self.DRAW_SKELETONS)
            if self.skeletons is not None and self.DRAW_SKELETONS:
                self.draw_skeletons(self.skeletons)
            pygame.display.update()

    def color_frame_ready(self,frame):
        if not self.VIDEO_COLOR:
            return
        print('video printing')
        with self.thread:
            self.copy_to_screen(frame)
            print(self.skeletons,self.DRAW_SKELETONS)
            if self.skeletons is not None and self.DRAW_SKELETONS:
                self.draw_skeletons(self.skeletons)
            pygame.display.update()

    def post_frame(self,frame):
        print('post frame')
        #try:
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, skeletons = frame.SkeletonData))
        #except:
            #print('post frame exception')
            # event queue full
            #pass

if __name__ == '__main__':
    if True:
        #screen = pygame.display.set_mode(DEPTH_RESOLUTION,0,16)
        pygame.display.set_caption('Handler Test')
        #kinect = nui.Runtime()
        #kinect.skeleton_engine_enabled = True
        handler = kinect_handler()
        handler.kinect.video_stream.open(nui.ImageStreamType.Video, 2,  COLOR_RESOLUTION_NUI_ID, nui.ImageType.Color)
        handler.kinect.depth_stream.open(nui.ImageStreamType.Depth, 2,  DEPTH_RESOLUTION_NUI_ID, nui.ImageType.Depth)

        print('Controls: ')
        print('     d - Switch to depth view')
        print('     v - Switch to video view')
        print('     s - Toggle displaing of the skeleton')
        print('     u - Increase elevation angle')
        print('     j - Decrease elevation angle')

    # main game loop
    done = False
    while not done:
        e = pygame.event.wait()
        handler.info = pygame.display.Info()
        if e.type == pygame.QUIT:
            print('quit',e.type)
            done = True
            break
        elif e.type == KINECTEVENT:
            print('kinect frame',e.type)
            handler.skeletons = e.skeletons
            if handler.DRAW_SKELETONS:
                handler.draw_skeletons(handler.skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                print('quit',e.type)
                done = True
                break
            elif e.key == K_d:
                print('depth',e.type)
                with handler.thread:
                    handler.screen = pygame.display.set_mode(DEPTH_RESOLUTION,0,16)
                    handler.VIDEO_COLOR = False
                    handler.VIDEO_DEPTH = True
            elif e.key == K_v:
                print('video',e.type)
                with handler.thread:
                    handler.screen = pygame.display.set_mode(COLOR_RESOLUTION,0,32)
                    handler.VIDEO_COLOR = True
                    handler.VIDEO_DEPTH = False
            elif e.key == K_s:
                print('skeletons {}'.format(not handler.DRAW_SKELETONS),e.type)
                handler.DRAW_SKELETONS = not handler.DRAW_SKELETONS
            elif e.key == K_u:
                handler.kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                handler.kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                handler.kinect.camera.elevation_angle = 2
