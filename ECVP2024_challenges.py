# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: stimupy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Open and FAIR stimulus creation with `stimupy`

# %%
import stimupy

# %reload_ext stimupy

# %% [markdown]
# ## Challenges

# %% [markdown]
# ### Challenge 0: generate exact existing stimulus

# %%
import stimupy.papers.RHS2007

stim_0 = stimupy.papers.RHS2007.WE_zigzag()

stimupy.utils.plot_stim(stim_0)

# %% [markdown]
# ### Challenge I, a: reproduce stimulus, from description

# %%
stim_Ia1 = stimupy.stimuli.gabors.gabor(
    visual_size = (2, 2),
    frequency=2,
    sigma=0.5,
    ppd = 32,
    rotation=45,
)

stimupy.utils.plot_stim(stim_Ia1)

# %%
stim_Ia2 = stimupy.stimuli.sbcs.basic_two_sided(
    visual_size=(8, 16),
    shape=(512,1024),
    intensity_background=(0.0, 1.0),
    intensity_target=0.5,
    target_size=4
)

stimupy.utils.plot_stim(stim_Ia2)

# %% [markdown]
# ### Challenge I, b: reprodue stimulus, from image

# %%
stim_Ib2 = stimupy.stimuli.todorovics.cross(
    visual_size=8,
    ppd=24,
    cross_size=6,
    cross_thickness=1,
    covers_size=3
)

stimupy.utils.plot_stim(stim_Ib2)

# %%
stim_Ib3 = stimupy.stimuli.todorovics.rectangle(
    visual_size=8,
    ppd=24,
    target_size=6,
    covers_size=3,
    covers_offset=(2,2),
)

stimupy.utils.plot_stim(stim_Ib3)

# %% [markdown]
# ## Challenge II: vary stimulus parametrization

# %%
stim_II = stimupy.checkerboards.checkerboard(
    check_visual_size=2,
    board_shape=(5,10),
    ppd=24
)

stimupy.utils.plot_stim(stim_II)

# %%
param_space = stimupy.utils.permutate_params(
    {
        "check_visual_size": [1, 2, 4],
        "ppd": [24],
        "board_shape": [(5,10),(7, 12), (3, 5)]
    }
)
param_space

# %%
stims_II = stimupy.utils.create_stimspace_stimuli(stimupy.stimuli.checkerboards.checkerboard, param_space)

stimupy.utils.plot_stimuli(stims_II)

# %% [markdown]
# ### Challenge III: compose a stimulus

# %%
WE_h = stimupy.stimuli.whites.white(
    visual_size=(3, 3),
    n_bars=6,
    target_indices=(2, 5),
    ppd=24,
    target_heights=.5,
)
WE_v = stimupy.utils.rotate_dict(WE_h, 1)

WE_h = stimupy.utils.pad_dict_to_visual_size(WE_h, visual_size=(10,5), ppd=24, pad_value=0.5)

WE_v = stimupy.utils.pad_dict_to_visual_size(WE_v, visual_size=(10,5), ppd=24, pad_value=0.5)

stim_III = stimupy.utils.stack_dicts(WE_h, WE_v)

stimupy.utils.plot_stim(stim_III)

# %% [markdown]
# ### Challenge IV: compose a new stimulus

# %%
import numpy as np

segments1 = stimupy.stimuli.waves.square_angular(
    visual_size=(20,20),
    ppd=32,
    n_segments=20,
    rotation=(360/20/2)
)
segments2 = stimupy.stimuli.waves.square_angular(
    visual_size=(20,20),
    ppd=32,
    n_segments=20,
    rotation=-(360/20/2)
)

rings = stimupy.stimuli.bullseyes.circular(
    visual_size=(20,20),
    ppd=32,
    n_rings=20
)

img = np.where(rings["ring_mask"] < 3, rings["img"], segments1["img"])
img = np.where(rings["ring_mask"] == 10, segments2["img"], img)
img = np.where(rings["ring_mask"] == 20, segments2["img"], img)



stim_IV = {"img": img, "visual_size": rings["visual_size"]}

stimupy.utils.plot_stim(stim_IV)
