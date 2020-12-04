#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 22:17:02 2020

@filename:    pronun.py
@author:      jeffchen
"""

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import nengo, nengo_gui, nengo_spa as spa

# $$$$$$$$$$$$$$$$$$$ CONSTANTS $$$$$$$$$$$$$$$$$$$$ #
seed = 1126
dimension = 29

# $$$$$$$$$$$$$$$$ MODEL DEFINITION $$$$$$$$$$$$$$$$ #
model = spa.Network(
    label="Pronunciation parsing", seed=seed)

# $$$$$$$$$$$$ DEFINE INPUT DATA SOURCE $$$$$$$$$$$$ #
def func_plotter(x, end=10):
    x_pts = np.linspace(0, end, num=2000)
    plt.plot(x_pts, [x(i) for i in x_pts])

def option_input(option_list, period=3):
    len_lst = len(option_list)
    random_list = np.random.randint(
        0, len_lst, 10 * len_lst)
    def func(t):
        return option_list[random_list[
            int(t // (period / len_lst) % len(random_list))]]
    return func

consa_inp = option_input(["P", "T", "K"], 1)
vowel_inp = option_input(["A", "U", "I"], 1)

with model:
    consa_inp = spa.Transcode(
        consa_inp, output_vocab=dimension)
    vowel_inp = spa.Transcode(
        vowel_inp, output_vocab=dimension)
