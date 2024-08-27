from copy import deepcopy

import numpy as np
import stimupy

__all__ = [
    "sbc",
    "sbc_separate",
    "sbc_smallest",
    "bullseye",
    "bullseye_high_freq",
    "bullseye_separate",
    "whites",
    "whites_narrow",
    "whites_separate",
    "strip",
    "checkerboard",
    "checkerboard_separate",
    "checkerboard_narrow",
]

# Defaults
TARGET_SIZE = 0.5  # deg. visual angle, square
ASPECT_RATIO = 2  # width : height
N_SURROUNDS = 5

PPD = 72
INTENSITY_BACKGROUND = 0.3


INTENSITY_CONTEXT = {
    "black": 0.0,
    "white": 1.0,
}


# Helpers
def VISUAL_SIZE(target_size, n_surrounds):
    return np.array((1, ASPECT_RATIO)) * (n_surrounds * 2 + 1) * target_size


def radii(target_size, n_surrounds):
    return (np.arange(n_surrounds * 2 + 1) + 1) * (target_size / 2)


def separation_mask(ppd=PPD, target_size=TARGET_SIZE, n_surrounds=N_SURROUNDS):
    stim = bullseye_high_freq(ppd=ppd, target_size=target_size)

    # Mask frames that need to be kept
    N_frames = len(radii(target_size, n_surrounds=n_surrounds))
    separate_mask = np.zeros_like(stim["frame_mask"])
    for frame in range(N_frames // 2):
        separate_mask = np.where(stim["frame_mask"] == frame + 1, 1, separate_mask)
        separate_mask = np.where(stim["frame_mask"] == N_frames + frame + 1, 1, separate_mask)

    return separate_mask


# %%      BULLSEYEs          #
# -------------------------- #
def bullseye(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    visual_size = VISUAL_SIZE(target_size, n_surrounds)

    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[0])
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=ppd,
        visual_size=(visual_size[0], visual_size[1] / 2),
        radii=radii(target_size, n_surrounds)[::2],
        intensity_frames=(*intensities.values(), intensity_surround),
        target_indices=1,
        intensity_target=intensity_targets[0],
    )

    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[1])
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=ppd,
        visual_size=(visual_size[0], visual_size[1] / 2),
        radii=radii(target_size, n_surrounds)[::2],
        intensity_frames=(*intensities.values(), intensity_surround),
        target_indices=1,
        intensity_target=intensity_targets[1],
    )
    stim = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    return stim


def bullseye_high_freq(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    visual_size = VISUAL_SIZE(target_size, n_surrounds)

    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[0])
    left = stimupy.stimuli.rings.rectangular_generalized(
        ppd=ppd,
        visual_size=(visual_size[0], visual_size[1] / 2),
        radii=radii(target_size, n_surrounds),
        intensity_frames=(*intensities.values(), intensity_surround),
        target_indices=1,
        intensity_target=intensity_targets[0],
    )

    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[1])
    right = stimupy.stimuli.rings.rectangular_generalized(
        ppd=ppd,
        visual_size=(visual_size[0], visual_size[1] / 2),
        radii=radii(target_size, n_surrounds),
        intensity_frames=(*intensities.values(), intensity_surround),
        target_indices=1,
        intensity_target=intensity_targets[1],
    )
    stim = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    return stim


def bullseye_separate(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    # High-freq, separated
    stim = bullseye_high_freq(
        ppd=ppd,
        contexts=contexts,
        intensity_contexts=intensity_contexts,
        intensity_targets=intensity_targets,
        target_size=target_size,
    )

    # Mask frames to keep
    separate_mask = separation_mask(ppd=ppd, target_size=target_size)
    stim["img"] = np.where(separate_mask, stim["img"], intensity_background)

    return stim


# %%         SBCs            #
# -------------------------- #
def sbc(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    return stimupy.stimuli.sbcs.square_two_sided(
        ppd=ppd,
        visual_size=VISUAL_SIZE(target_size, n_surrounds),
        target_radius=target_size / 2,
        surround_radius=VISUAL_SIZE(target_size, n_surrounds)[0] / 2,
        intensity_target=intensity_targets,
        intensity_surround=(intensity_contexts[contexts[0]], intensity_contexts[contexts[1]]),
        intensity_background=intensity_background,
    )


def sbc_separate(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    return stimupy.stimuli.sbcs.square_two_sided(
        ppd=ppd,
        visual_size=VISUAL_SIZE(target_size, n_surrounds),
        target_radius=target_size / 2,
        surround_radius=radii(target_size, n_surrounds)[len(radii(target_size, n_surrounds)) // 2],
        intensity_target=intensity_targets,
        intensity_surround=(intensity_contexts[contexts[0]], intensity_contexts[contexts[1]]),
        intensity_background=intensity_background,
    )


def sbc_smallest(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    stim = stimupy.stimuli.sbcs.square_two_sided(
        ppd=ppd,
        visual_size=VISUAL_SIZE(target_size, n_surrounds),
        target_radius=target_size / 2,
        surround_radius=radii(target_size, n_surrounds)[1],
        intensity_target=intensity_targets,
        intensity_surround=(intensity_contexts[contexts[0]], intensity_contexts[contexts[1]]),
        intensity_background=intensity_background,
    )

    return stim


# %%    CHECKERBOARDS        #
# -------------------------- #
def checkerboard_target_cols(n_surrounds=N_SURROUNDS):
    return {
        "black": (n_surrounds, (4 * n_surrounds + 2) - (n_surrounds + 1) - 1),
        "white": (n_surrounds + 1, (4 * n_surrounds + 2) - (n_surrounds + 1)),
    }


def checkerboard(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    target_row = n_surrounds
    target_indices = [
        (target_row, checkerboard_target_cols(n_surrounds)[context][i])
        for i, context in enumerate(contexts)
    ]

    return stimupy.stimuli.checkerboards.checkerboard(
        ppd=ppd,
        visual_size=VISUAL_SIZE(target_size, n_surrounds),
        check_visual_size=target_size,
        target_indices=target_indices,
        intensity_checks=[*reversed(intensity_contexts.values())],
        intensity_target=intensity_targets,
    )


def checkerboard_narrow(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    vis_size = VISUAL_SIZE(target_size, n_surrounds)

    target_row = n_surrounds // 2
    target_indices = [
        (target_row, checkerboard_target_cols(n_surrounds)[context][i])
        for i, context in enumerate(contexts)
    ]

    intensity_checks = [*reversed(intensity_contexts.values())]
    if ((n_surrounds // 2) - 1) % 2:
        intensity_checks = [*reversed(intensity_checks)]

    checkerboard_narrow = stimupy.stimuli.checkerboards.checkerboard(
        ppd=ppd,
        visual_size=(n_surrounds * target_size, vis_size[1]),
        check_visual_size=target_size,
        target_indices=target_indices,
        intensity_checks=intensity_checks,
        intensity_target=intensity_targets,
    )

    return stimupy.utils.pad_dict_to_visual_size(
        dct=checkerboard_narrow, ppd=ppd, visual_size=vis_size, pad_value=intensity_background
    )


def checkerboard_separate(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    vis_size = VISUAL_SIZE(target_size, n_surrounds)

    # Generate left checkerboard
    if contexts[0] == "white":
        intensity_checks = [*intensity_contexts.values()]
    elif contexts[0] == "black":
        intensity_checks = [*reversed(intensity_contexts.values())]

    left = stimupy.stimuli.checkerboards.checkerboard(
        ppd=ppd,
        board_shape=(n_surrounds,) * 2,
        check_visual_size=target_size,
        target_indices=[((n_surrounds // 2), (n_surrounds // 2))],
        intensity_checks=intensity_checks,
        intensity_target=intensity_targets[0],
    )
    left = stimupy.utils.pad_dict_to_visual_size(
        dct=left,
        ppd=ppd,
        visual_size=(vis_size[0], vis_size[1] / 2),
        pad_value=intensity_background,
    )

    # Generate right checkerboard
    if contexts[1] == "white":
        intensity_checks = [*intensity_contexts.values()]
    elif contexts[1] == "black":
        intensity_checks = [*reversed(intensity_contexts.values())]

    right = stimupy.stimuli.checkerboards.checkerboard(
        ppd=ppd,
        check_visual_size=target_size,
        board_shape=(n_surrounds,) * 2,
        target_indices=[((n_surrounds // 2), (n_surrounds // 2))],
        intensity_checks=intensity_checks,
        intensity_target=intensity_targets[1],
    )
    right = stimupy.utils.pad_dict_to_visual_size(
        dct=right,
        ppd=ppd,
        visual_size=(vis_size[0], vis_size[1] / 2),
        pad_value=intensity_background,
    )

    # Combine
    stim = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    return stim


def checkerboard_smallest(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    # Generate checkerboard(s)
    stim = checkerboard_separate(
        ppd=ppd,
        intensity_targets=intensity_targets,
        contexts=contexts,
        intensity_contexts=intensity_contexts,
        target_size=target_size,
        n_surrounds=n_surrounds,
        intensity_background=intensity_background,
    )

    # Mask only inner rings
    x = bullseye_high_freq(
        ppd=ppd,
        intensity_targets=intensity_targets,
        contexts=contexts,
        intensity_contexts=intensity_contexts,
        target_size=target_size,
        n_surrounds=n_surrounds,
        intensity_background=intensity_background,
    )
    inner_ring_mask = np.where(x["frame_mask"] < 4, 1, 0)
    inner_ring_mask = np.where(
        np.logical_and(x["frame_mask"] > 12, x["frame_mask"] < 15), 2, inner_ring_mask
    )
    inner_ring_mask = np.where(stim["target_mask"], 0, inner_ring_mask)

    # Replace
    stim["img"] = np.where(
        np.logical_or(inner_ring_mask, stim["target_mask"]), stim["img"], intensity_background
    )

    return stim


def cross(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    stim = checkerboard_smallest(
        ppd=ppd,
        intensity_targets=intensity_targets,
        contexts=contexts,
        intensity_contexts=intensity_contexts,
        target_size=target_size,
        n_surrounds=n_surrounds,
        intensity_background=intensity_background,
    )

    # Corners mask
    corner_idcs_l = [7, 9, 17, 19]
    corners_mask = np.zeros_like(stim["checker_mask"])
    for idx in corner_idcs_l:
        corners_mask = np.where(stim["checker_mask"] == idx, 1, corners_mask)
    corner_idcs_r = [32, 34, 42, 44]
    for idx in corner_idcs_r:
        corners_mask = np.where(stim["checker_mask"] == idx, 2, corners_mask)

    # Remove
    stim["img"] = np.where(corners_mask, intensity_background, stim["img"])

    return stim


def cross_polarity(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    stim = cross(
        ppd=ppd,
        intensity_targets=intensity_targets,
        contexts=contexts,
        intensity_contexts=intensity_contexts,
        target_size=target_size,
        n_surrounds=n_surrounds,
        intensity_background=intensity_background,
    )

    # Flankers mask
    flankers_h_mask = np.zeros_like(stim["checker_mask"])
    flankers_h_l_idcs = [12, 14]
    for idx in flankers_h_l_idcs:
        flankers_h_mask = np.where(stim["checker_mask"] == idx, 1, flankers_h_mask)
    flankers_h_r_idcs = [37, 39]
    for idx in flankers_h_r_idcs:
        flankers_h_mask = np.where(stim["checker_mask"] == idx, 2, flankers_h_mask)

    # Switch polarity
    switched_contexts = (contexts[1], contexts[0])
    for idx, context in enumerate(switched_contexts):
        stim["img"] = np.where(
            flankers_h_mask == idx + 1, intensity_contexts[context], stim["img"]
        )

    return stim


# %%       WHITEs            #
# -------------------------- #
def whites_target_indices(N_surrounds=N_SURROUNDS):
    return {
        "black": (N_surrounds + 1, -(N_surrounds + 1)),
        "white": (N_surrounds + 2, -N_surrounds),
    }


def whites(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    target_indices = (
        whites_target_indices(n_surrounds)[contexts[0]][0],
        whites_target_indices(n_surrounds)[contexts[1]][1],
    )

    if n_surrounds % 2:
        intensity_bars = [*intensity_contexts.values()]
    else:
        intensity_bars = [*reversed(intensity_contexts.values())]

    return stimupy.stimuli.whites.white(
        ppd=ppd,
        visual_size=VISUAL_SIZE(target_size, n_surrounds),
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=intensity_bars,
        intensity_target=intensity_targets,
    )


def whites_narrow(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    vis_size = VISUAL_SIZE(target_size, n_surrounds)

    target_indices = (
        whites_target_indices(n_surrounds)[contexts[0]][0],
        whites_target_indices(n_surrounds)[contexts[1]][1],
    )

    if n_surrounds % 2:
        intensity_bars = [*intensity_contexts.values()]
    else:
        intensity_bars = [*reversed(intensity_contexts.values())]

    whites_narrow = stimupy.stimuli.whites.white(
        ppd=ppd,
        visual_size=(n_surrounds * target_size, vis_size[1]),
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=intensity_bars,
        intensity_target=intensity_targets,
    )

    return stimupy.utils.pad_dict_to_visual_size(
        dct=whites_narrow, ppd=ppd, visual_size=vis_size, pad_value=intensity_background
    )


def whites_separate(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    vis_size = VISUAL_SIZE(target_size, n_surrounds)

    # Generate left White's
    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[0])
    if ((n_surrounds - 1) // 2) % 2:
        intensity_bars = [intensity_surround, *intensities.values()]
    else:
        intensity_bars = [*intensities.values(), intensity_surround]

    left = stimupy.stimuli.whites.white(
        ppd=ppd,
        n_bars=n_surrounds,
        bar_width=target_size,
        target_indices=(((n_surrounds - 1) // 2) + 1,),
        target_heights=target_size,
        intensity_bars=intensity_bars,
        intensity_target=intensity_targets,
    )
    left = stimupy.utils.pad_dict_to_visual_size(
        dct=left,
        ppd=ppd,
        visual_size=(vis_size[0], vis_size[1] / 2),
        pad_value=intensity_background,
    )

    # Generate right White's
    intensities = deepcopy(intensity_contexts)
    intensity_surround = intensities.pop(contexts[1])
    if ((n_surrounds - 1) // 2) % 2:
        intensity_bars = [intensity_surround, *intensities.values()]
    else:
        intensity_bars = [*intensities.values(), intensity_surround]

    right = stimupy.stimuli.whites.white(
        ppd=ppd,
        n_bars=n_surrounds,
        bar_width=target_size,
        target_indices=(((n_surrounds - 1) // 2) + 1,),
        target_heights=target_size,
        intensity_bars=intensity_bars,
        intensity_target=intensity_targets,
    )
    right = stimupy.utils.pad_dict_to_visual_size(
        dct=right,
        ppd=ppd,
        visual_size=(vis_size[0], vis_size[1] / 2),
        pad_value=intensity_background,
    )
    # Combine
    stim = stimupy.utils.stack_dicts(left, right, direction="horizontal")

    return stim


# %%        STRIP            #
# -------------------------- #
def strip(
    ppd=PPD,
    intensity_targets=(0.5, 0.5),
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    vis_size = VISUAL_SIZE(target_size, n_surrounds)

    target_indices = (
        whites_target_indices(n_surrounds)[contexts[0]][0],
        whites_target_indices(n_surrounds)[contexts[1]][1],
    )

    if n_surrounds % 2:
        intensity_bars = [*reversed(intensity_contexts.values())]
    else:
        intensity_bars = [*intensity_contexts.values()]

    whites_narrow = stimupy.stimuli.whites.white(
        ppd=ppd,
        visual_size=(target_size, vis_size[1]),
        bar_width=target_size,
        target_indices=target_indices,
        target_heights=target_size,
        intensity_bars=intensity_bars,
        intensity_target=intensity_targets,
    )

    return stimupy.utils.pad_dict_to_visual_size(
        dct=whites_narrow, ppd=ppd, visual_size=vis_size, pad_value=intensity_background
    )


def gen_all(
    ppd=PPD,
    contexts=("black", "white"),
    intensity_contexts=INTENSITY_CONTEXT,
    intensity_targets=(0.5, 0.5),
    target_size=TARGET_SIZE,
    n_surrounds=N_SURROUNDS,
    intensity_background=INTENSITY_BACKGROUND,
):
    # Initialize empty dict to hold all stims
    stims = {}

    for stim_name in __all__:
        stims[stim_name] = globals()[stim_name](
            ppd=ppd,
            contexts=contexts,
            intensity_contexts=intensity_contexts,
            target_size=target_size,
            n_surrounds=n_surrounds,
            intensity_targets=intensity_targets,
            intensity_background=intensity_background,
        )

    return stims


if __name__ == "__main__":
    stim = cross_polarity()
    stimupy.utils.plot_stim(stim)
    # s = gen_all(contexts=("black", "white"), n_surrounds=N_SURROUNDS)

    # stimupy.utils.plot_stimuli(s)
