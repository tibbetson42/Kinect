import thread
import itertools
import ctypes
# Import PyKinect Library
import pykinect
from pykinect import nui
from pykinect.nui import JointId # Id constants for various limbs. 0 = Hip center

# Import pygame
import pygame #?????
from pygame.color import THECOLORS # color dictionary
from pygame.locals import * #???????????


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
