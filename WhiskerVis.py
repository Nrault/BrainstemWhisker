#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:39:00 2023

@author: nicolas
"""

from brian2 import *
import numpy as np
import random as rng
import pandas as pd

prefs.codegen.target = 'numpy'

defaultclock.dt = 1 * ms

def import_force_moment():
    
    force_x = pd.read_csv("./dynamics/Fx.csv", header=None)
    force_x = force_x[:][0]
    force_y = pd.read_csv("./dynamics/Fy.csv", header=None)
    force_y = force_y[:][0]
    force_z = pd.read_csv("./dynamics/Fz.csv", header=None)
    force_z = force_z[:][0]
    bp_force = []
    
    moment_x = pd.read_csv("./dynamics/Mx.csv", header=None)
    moment_x = moment_x[:][0]
    moment_y = pd.read_csv("./dynamics/My.csv", header=None)
    moment_y = moment_y[:][0]
    moment_z = pd.read_csv("./dynamics/Mz.csv", header=None)
    moment_z = moment_z[:][0]
    bp_moment = []
    
    for x, y, z in zip(force_x, force_y, force_z):
        bp_force.append([x, y, z])
    for x, y, z in zip(moment_x, moment_y, moment_z):
        bp_moment.append([x, y, z])
    bp_force = np.asarray(bp_force)
    bp_moment = np.asarray(bp_moment)
    
    return bp_force, bp_moment


def import_whisker_pos(duration):
    
    pos_x = pd.read_csv("./kinematics/x/LB1.csv", header=None)
    pos_x = np.asarray(pos_x)
    pos_y = pd.read_csv("./kinematics/y/LB1.csv", header=None)
    pos_y = np.asarray(pos_y)
    pos_z = pd.read_csv("./kinematics/z/LB1.csv", header=None)
    pos_z = np.asarray(pos_z)
    
    return pos_x, pos_y, pos_z