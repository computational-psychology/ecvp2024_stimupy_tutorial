"""
Created on Mon Aug 26 15:59:44 2024
@author: lynnschmittwilken
"""

import numpy as np
from hrl import HRL


# Monitor specs / center of screen
WIDTH, HEIGHT = 1920, 1080
coords = [WIDTH / 2.0, HEIGHT / 2.0]   # center coords
rate = 60                              # frame rate
nframes = 3 * rate                     # duration of effects

bg = 0.5

# Create HRL object
hrl = HRL(
    graphics='gpu',
    inputs='keyboard',
    wdth=WIDTH,
    hght=HEIGHT,
    bg=bg,
    # fs=False,
    )

def stimFunc1():
    stimArr = np.ones((10, 10))
    
    for t in range(nframes):
        stimTex = presentStim(stimArr * t/nframes)
    return stimTex
    


def presentStim(stimArr):
    # Create texture
    stimTex = hrl.graphics.newTexture(stimArr)
    
    # Draw texture
    stimTex.draw((coords[0] - stimTex.wdth/2, coords[1] - stimTex.hght/2))
    hrl.graphics.flip(clr=True)
    return stimTex


if __name__ == '__main__':
    stimFuncs = [stimFunc1, stimFunc1, stimFunc1]
    idx = 0
    
    # Display first stimulus
    stimTex = stimFuncs[idx]()
    
    # Continue displaying
    while True:
        (btn,t1) = hrl.inputs.readButton()
        
        if btn == 'Space':
            break
        
        elif btn == 'Right' and idx < (len(stimFuncs)-1):
            idx += 1
            stimTex.delete()
            stimTex = stimFuncs[idx]()
            
        elif btn == 'Left' and idx > 0:
            idx -= 1
            stimTex.delete()
            stimTex = stimFuncs[idx]()


    stimTex.delete()
    hrl.close()


