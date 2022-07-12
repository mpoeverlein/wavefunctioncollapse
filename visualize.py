#!/usr/bin/python3

'''
This file contains plotting functionalities.
'''

import propagate as prop

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_canvas(tile_array, colors):
    canvas = np.zeros((tile_array.shape[0], tile_array.shape[1], 3))
    for iy, ix in np.ndindex(tile_array.shape):
        canvas[iy,ix,:] = colors[tile_array[iy,ix].get_state()][:3]

    return canvas/255

def plot_single_grid(tile_array, colors, interpolation='none', cmap=None):
    fig, ax = plt.subplots()
    ax.imshow(make_canvas(tile_array, colors), interpolation=interpolation, cmap=cmap)
    plt.show()

def plot_annotated_entropy(tile_array):
    fig, ax = plt.subplots()
    sns.heatmap(prop.vectorized_entropy(tile_array).astype(float), annot=True, fmt=".2f")
    plt.show()


def plot_evolution(tile_array_list, interpolation='none', cmap=None, fps=30):
    pass

