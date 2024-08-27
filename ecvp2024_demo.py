"""
Created on Mon Aug 26 15:59:44 2024
@author: lynnschmittwilken
"""

from PIL import Image
import numpy as np
from hrl import HRL
import stimuli
from stimupy.utils import pad_dict_to_shape, flip_dict, stack_dicts


# %% Prepare
WIDTH, HEIGHT = 1920, 1080             # monitor specs
coords = [WIDTH / 2.0, HEIGHT / 2.0]   # center coords
rate = 60                              # frame rate
nframes = int(1. * rate)              # duration of transitions in frames
fade = np.linspace(0, 1, nframes)      # linear fading

# Create HRL object
hrl = HRL(
    graphics='gpu',
    inputs='keyboard',
    wdth=WIDTH,
    hght=HEIGHT,
    bg=stimuli.INTENSITY_BACKGROUND,
    # fs=False,
    )


# %% Stimuli

############ Natural image
s0 = np.array(Image.open("natural_scene_kingdom2011.png").convert("L")) / 255

############# Simplify
# sbc black
s1 = stimuli.sbc(contexts=["black","background"], intensity_contexts={
    "black": 0.0,
    "background": stimuli.INTENSITY_BACKGROUND,
    "white": 1.0,
}, intensity_targets=(0.5, stimuli.INTENSITY_BACKGROUND))
s1 = pad_dict_to_shape(s1, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)


# sbc white
s2 = stimuli.sbc(contexts=["background","white"], intensity_contexts={
    "black": 0.0,
    "background": stimuli.INTENSITY_BACKGROUND,
    "white": 1.0,
}, intensity_targets=(stimuli.INTENSITY_BACKGROUND, 0.5))
s2 = pad_dict_to_shape(s2, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)


# sbc full
s3 = pad_dict_to_shape(stimuli.sbc(), shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)


############# Context effects
# sbc small
s4 = stimuli.sbc_separate()
s4 = pad_dict_to_shape(s4, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s5 = stimuli.sbc_smallest()
s5 = pad_dict_to_shape(s5, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

# bullseye
s6 = stimuli.bullseye_separate()
s6 = pad_dict_to_shape(s6, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s7 = stimuli.bullseye_high_freq()
s7 = pad_dict_to_shape(s7, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

# checkerboard
s7_ = stimuli.cross()
s7_ = pad_dict_to_shape(s7_, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s8 = stimuli.checkerboard_smallest()
s8 = pad_dict_to_shape(s8, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s9 = stimuli.checkerboard_separate()
s9 = pad_dict_to_shape(s9, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s10 = stimuli.checkerboard()
s10 = pad_dict_to_shape(s10, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

# White's
s11 = stimuli.cross_polarity()
s11 = pad_dict_to_shape(s11, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s12 = stimuli.whites_separate()
s12= pad_dict_to_shape(s12, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s14 = stimuli.whites()
s14 = pad_dict_to_shape(s14, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s15 = stimuli.whitesLong()
s15 = pad_dict_to_shape(s15, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

s16 = stimuli.whiteHowe()
s16 = pad_dict_to_shape(s16, shape=(768, 1024), pad_value=stimuli.INTENSITY_BACKGROUND)

sStack1 = stack_dicts(stack_dicts(s3, s7, keys="img"), s10, keys="img")
sStack2 = stack_dicts(stack_dicts(s14, s15, keys="img"), s16, keys="img")
sStack = stack_dicts(sStack1, sStack2, "vertical", keys="img")



# %% Functions
def starter():
    stimTex = presentStim(s0)
    return stimTex

def finisher():
    stimTex = presentStim(sStack["img"][::2, ::2])
    return stimTex

############# SBCs
def sbcBlack():
    stimTex = presentStim(s1["img"])
    return stimTex

def sbcWhite():
    stimTex = presentStim(s2["img"])
    return stimTex

def sbcFull():
    stimTex = presentStim(s3["img"])
    return stimTex

def sbcSmaller():
    for t in range(int(nframes)):
        stim = s3["img"] * (1-fade[t]) + s4["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def sbcSmallest():
    for t in range(int(nframes)):
        stim = s4["img"] * (1-fade[t]) + s5["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex


############# Bullseyes
def bullseyeSmall():
    for t in range(int(nframes)):
        stim = s5["img"] * (1-fade[t]) + s6["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def bullseye():
    for t in range(int(nframes)):
        stim = s6["img"] * (1-fade[t]) + s7["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def bullseyeSmallest():
    for t in range(int(nframes)):
        stim = s7["img"] * (1-fade[t]) + s5["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex


############# Checkerboard
def cross():
    for t in range(int(nframes)):
        stim = s5["img"] * (1-fade[t]) + s7_["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def checkSmall():
    for t in range(int(nframes)):
        stim = s7_["img"] * (1-fade[t]) + s8["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def check():
    for t in range(int(nframes)):
        stim = s8["img"] * (1-fade[t]) + s9["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def checkFull():
    for t in range(int(nframes)):
        stim = s9["img"] * (1-fade[t]) + s10["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex


############# Whites
def cross2():
    for t in range(int(nframes)):
        stim = s10["img"] * (1-fade[t]) + s7_["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def crossPol():
    for t in range(int(nframes)):
        stim = s7_["img"] * (1-fade[t]) + s11["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def white():
    for t in range(int(nframes)):
        stim = s11["img"] * (1-fade[t]) + s12["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def whiteFull():
    for t in range(int(nframes)):
        stim = s12["img"] * (1-fade[t]) + s14["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def whiteLong():
    for t in range(int(nframes)):
        stim = s14["img"] * (1-fade[t]) + s15["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def whiteHowe():
    for t in range(int(nframes)):
        stim = s15["img"] * (1-fade[t]) + s16["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex

def sbcSmall():
    for t in range(int(nframes)):
        stim = s16["img"] * (1-fade[t]) + flip_dict(s5)["img"] * fade[t]
        stimTex = presentStim(stim)
    return stimTex



############# Helper functions
def presentStim(stimArr):
    # Create texture
    stimTex = hrl.graphics.newTexture(stimArr)
    
    # Draw texture
    stimTex.draw((coords[0] - stimTex.wdth/2, coords[1] - stimTex.hght/2))
    hrl.graphics.flip(clr=True)
    return stimTex


# %% Execute
if __name__ == '__main__':
    stimFuncs = [
        starter,
        sbcBlack, sbcWhite, sbcFull, sbcSmaller, sbcSmallest,  # sbc
        bullseyeSmall, bullseye, bullseyeSmallest,             # bullseye
        cross, checkSmall, check, checkFull,                   # checkerboard
        cross2, crossPol, white, whiteFull, whiteLong,         # Whites
        whiteHowe, sbcSmall,                                   # return
        finisher, starter,
        ]
    idx = 0
    
    # Display first stimulus
    stimTex = stimFuncs[idx]()
    
    # Continue displaying
    while True:
        (btn,t1) = hrl.inputs.readButton()
        
        if btn == 'Space':
            break
        
        elif btn == 'Right':
            idx += 1
            idx = idx % (len(stimFuncs)-1)
            stimTex.delete()
            stimTex = stimFuncs[idx]()
            
        elif btn == 'Left':
            idx -= 1
            idx = idx % (len(stimFuncs)-1)
            stimTex.delete()
            stimTex = stimFuncs[idx]()


    stimTex.delete()
    hrl.close()


