import thread
import itertools
import ctypes
# Import PyKinect Library
import pykinect
from pykinect import nui
from pykinect.nui import JointId # Id constants for various limbs. 0 = Hip center

print('initializing')
import pygame #?????
from pygame.color import THECOLORS # color dictionary
from pygame.locals import * #???????????
#Import my Initialization
import myFirstKinect_config as cf
###============================================================================
## Initialization ##-------------------------------------------------
# Resolution #-----------------------------------------
# note that many pykinect functions like depth_to_skeleton are built for 320x240
DEPTH_RESOLUTION = (640,480)
DEPTH_RESOLUTION_NUI_ID = nui.ImageResolution.Resolution640x480
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
###============================================================================
## Function Defintions ##--------------------------------------------
# Resolution #-----------------------------------------
class kinect_handler:
    def __init__(self,kinect,screen):
        self.kinect = kinect
        self.kinect.skeleton_engine_enabled = True
        self.screen = screen
        self.info = pygame.display.Info()
        self.VIDEO_COLOR = True
        self.VIDEO_DEPTH = False
        self.skeletons = None
        self.thread = thread.allocate()
        if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
           Py_ssize_t = ctypes.c_int
        elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
           Py_ssize_t = ctypes.c_int64
        else:
           raise TypeError("Cannot determine type of Py_ssize_t")
        self._PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
        self._PyObject_AsWriteBuffer.restype = ctypes.c_int
        self._PyObject_AsWriteBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(Py_ssize_t)]

    def copy_to_screen(self,frame):
        with self.thread:
            buffer = self.screen.get_buffer()
            address = ctypes.c_void_p()
            size = self.Py_ssize_t()
            self._PyObject_AsWriteBuffer(screen.get_buffer(),ctypes.byref(address),ctypes.byref(size))
            bytes = (ctypes.c_byte * size.value).from_address(address.value)
            bytes.object = buffer_interface
            frame.image.copy_bits(bytes)
            return

    def draw_skeletons(self,skeletons):
        for idx,skeleton in enumerate(skeletons):
            color =  SKELETON_COLORS[idx]
            # HeadXYZ = skeleton.SkeletonPositions[JointId.Head]
            # print(HeadXYZ)
            # HeadPxls = nui.SkeletonEngine.skeleton_to_depth_image(HeadXYZ,dispInfo.current_w,dispInfo.current_h)
            # pygame.draw.circle(screen, color, (int(HeadPxls[0]), int(HeadPxls[1])), 20, 0)
            self.draw_bodypart(skeleton,  SPINE,     color)
            self.draw_bodypart(skeleton,  LEFT_ARM,  color)
            self.draw_bodypart(skeleton,  RIGHT_ARM, color)
            self.draw_bodypart(skeleton,  LEFT_LEG,  color)
            self.draw_bodypart(skeleton,  RIGHT_LEG, color)

    def draw_bodypart(self, skeleton, joint_Indexes, color, width = 3):
        thisJointXYZ = skeleton.SkeletonPositions[joint_Indexes[0]]
        if joint_Indexes[0] == JointId.Head:
            pygame.draw.circle(screen, color, thisJointPxls,10*width,0)
        for joint in itertools.islice(Joint_Indexes,1,None):
            nextJointXYZ = skeleton.SkeletonPositions[joint.value]
            thisJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(thisJointXYZ)
            nextJointPxls = nui.SkeletonEngine.skeleton_to_depth_image(nextJointXYZ)
            pygame.draw.line(screen, color, thisJointPxls,nextJointPxls,width)
            if joint.value == JointId.Head:
                pygame.draw.circle(screen, color, thisJointPxls,10*width,0)
            thisJointXYZ = nextJointXYZ

    def depth_frame_ready(self,frame):
        if self.VIDEO_DEPTH:
            return
        self.copy_to_screen(frame)
        if self.skeletons is not None and self.DRAW_SKELETONS:
            self.draw_skeletons(self)
        pygame.display.update()

    def color_frame_ready(self,frame):
        if self.VIDEO_COLOR:
            return
        self.copy_to_screen(frame)
        if self.skeletons is not None and self.DRAW_SKELETONS:
            self.draw_skeletons(self)
        pygame.display.update()


def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT, skeletons = frame.SkeletonData))
            #print(frame.SkeletonData)
        except:
            # event queue full
            pass

if __name__ == '__main__':
    print('asodbnaosd')
    screen = pygame.display.set_mode( DEPTH_RESOLUTION,0,16)
    pygame.display.set_caption('Handler Test')
    kinect = nui.Runtime()
    handler = kinect_handler(kinect,screen)

    kinect.skeleton_frame_ready += post_frame
    print(handler.kinect.skeleton_frame_ready)
    kinect.depth_frame_ready += handler.depth_frame_ready
    kinect.video_frame_ready += handler.video_frame_ready
    kinect.video_stream.open(nui.ImageStreamType.Video, 2,  DEPTH_RESOLUTION_NUI_ID, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2,  DEPTH_RESOLUTION_NUI_ID, nui.ImageType.Depth)
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
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == pygame.USEREVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                draw_skeletons(skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    video_display = False
            elif e.key == K_v:
                with screen_lock:
                    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,16)
                    video_display = True
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2
