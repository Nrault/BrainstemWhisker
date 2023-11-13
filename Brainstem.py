# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from brian2 import *
import numpy as np
import random as rng
import pandas as pd
import socket
import json
import time

prefs.codegen.target = 'numpy'

defaultclock.dt = 1 * ms


''' convert_input_to_time_array
Take an array and turns it into a Timedarray in which each cell's value 
correspond to a timestep
'''
def convert_input_to_time_array(input_data):
    return TimedArray(input_data, dt = 1 * ms)

def unit_vec(vec):
    return vec/np.linalg.norm(vec)

def get_direction(force):
    force = unit_vec(force)
    if np.isnan(force[0]):
        return 0
    return np.rad2deg(np.arccos(np.dot(force, np.array([0, 0, 1]))))

def force_to_time_array(force_array):
    direction = np.zeros(shape(force_array)[0])
    amplitude = np.zeros(shape(force_array)[0])
    amplitudeOnOff = np.zeros(shape(force_array)[0])
    for idx, force in enumerate(force_array):
        direction[idx] = get_direction(force)
        amplitude[idx] = np.linalg.norm(force) * 100
        amplitudeOnOff[idx] = amplitude[idx] > 3
    
    direction = TimedArray(direction, dt = 1*ms)
    amplitude = TimedArray(amplitude, dt = 1 * ms)
    amplitudeOnOff = TimedArray(amplitudeOnOff, dt = 1 * ms)
    
    return direction, amplitude, amplitudeOnOff
    
def force_to_time_array_per_timestep(force_array):
    direction = np.zeros((1, 1))
    amplitude = np.zeros((1, 1))
    amplitudeOnOff = np.zeros((1, 1))
    for idx, force in enumerate(force_array):
        direction[idx] = get_direction(force)
        amplitude[idx] = np.linalg.norm(force) * 100
        amplitudeOnOff[idx] = amplitude[idx] > 3
    
    direction = TimedArray(direction, dt = 1*ms)
    amplitude = TimedArray(amplitude, dt = 1 * ms)
    amplitudeOnOff = TimedArray(amplitudeOnOff, dt = 1 * ms)
    
    return direction, amplitude, amplitudeOnOff

''' create_ramp_and_hold
'''

def create_ramp_and_hold(start_time, end_time, max_amp, min_amp, angle, 
                         ramp_duration, duration):
    direction = np.ones((duration)) * angle
    amplitude = np.ones((duration)) * min_amp
    amp_on = np.zeros((duration))
    
    for i in range(1, duration):
        if i >= start_time and i <= end_time:
            amp_on[i] = 1
            if i < (end_time - ramp_duration):
                if amplitude[i-1] < max_amp:
                    amplitude[i] = amplitude[i-1] + max_amp/ramp_duration
                else :
                    amplitude[i] = max_amp
            else:
                if i >= (end_time - ramp_duration) and amplitude[i-1] > min_amp:
                    amplitude[i] = amplitude[i-1] - max_amp/ramp_duration
    return direction, amplitude, amp_on

def create_random_stim(duration):
    pass

def import_data():
    
    force_x = pd.read_csv("./dynamics/Fx.csv", header=None)
    force_x = force_x[:][0]
    force_y = pd.read_csv("./dynamics/Fy.csv", header=None)
    force_y = force_y[:][0]
    force_z = pd.read_csv("./dynamics/Fz.csv", header=None)
    force_z = force_z[:][0]
    bp_force = []
    
    for x, y, z in zip(force_x, force_y, force_z):
        bp_force.append([x, y, z])
    bp_force = np.asarray(bp_force)
    
    return bp_force

''' define_favorite_angles
'''

def define_favorite_angles(forward_low, forward_high, backward_low, 
                           backward_high, pop_size):
    forward = [rng.randrange(forward_low, forward_high, 1) for i in range(0, int(pop_size/2))]
    backward = [rng.randrange(backward_low, backward_high, 1) for i in range(0, int(pop_size/2))]
    return np.concatenate((np.asarray(forward), np.asarray(backward)))

def define_favorite_angles_gauss(mu_forward, mu_backward, sigma, pop_size):
    forward = [rng.gauss(mu_forward, sigma) for i in range(0, int(pop_size/2))]
    backward = [rng.gauss(mu_backward, sigma) for i in range(0, int(pop_size/2))]
    return np.concatenate((np.asarray(forward), np.asarray(backward)))

def define_favorite_angles_uniform(pop_size):
    return [rng.randrange(0, 360, 1) for i in range(0, int(pop_size))]

SA1 = '''
Iapp = I * (amplitudeOnOff(t)) * ((1 + cos((direction(t) - fav_direction) * (pi/180)))/2) : amp
dISRA_SA1/dt = (a_SA1 * (v - El) - ISRA_SA1)/tau_SRA1 : amp
Il = Gl * (El - v + delta_th * exp((v - Vth)/delta_th)) : amp
dv/dt = (Il - ISRA_SA1 + Iapp)/Cm : volt

fav_direction : 1
I : amp
'''

SA2 = '''
Iapp = I * (amplitude(t)) * ((1 + cos((direction(t) - fav_direction) * (pi/180)))/2) : amp
dISRA_SA2/dt = (a_SA2 * (v - El) - ISRA_SA2)/tau_SRA2 : amp
Il = Gl * (El - v + delta_th * exp((v - Vth)/delta_th)) : amp
dv/dt = (Il - ISRA_SA2 + Iapp)/Cm : volt

fav_direction : 1
I : amp
'''

RA = '''
Iapp = I * (amplitudeOnOff(t)) * log(1 + ((amplitude(t) - amplitude(t-(dt))))**2) * ((1 + cos((direction(t) - fav_direction) * (pi/180)))/2) : amp
dISRA_RA/dt = (0 * nS * (v - El) - ISRA_RA)/tau_RA : amp
Il = Gl * (El - v + delta_th * exp((v - Vth)/delta_th)) : amp
dv/dt = (Il - ISRA_RA + Iapp)/Cm : volt

fav_direction : 1
I : amp
'''

bar = '''
Il = Gl_bar * (El_bar - v + delta_th * exp((v - Vth_bar)/delta_th)) : amp
dv/dt = (Il)/Cm : volt
'''

tg = '''
Il = Gl_tg * (El_tg - v + delta_th * exp((v - Vth_tg)/delta_th)) : amp
dv/dt = (Il)/Cm : volt
'''

def compute_spontaneous_activity(spike_times, duration):
    spontaneous_activity = np.zeros((duration, 1))
    spike_times = np.round(spike_times/ms)
    for i in range(duration):
        spontaneous_activity[i] = sum(spike_times==i)
    return spontaneous_activity

def model(a_SA1, b_SA1, tau_SRA1, a_SA2, b_SA2, tau_SRA2, a_RA, b_RA, tau_RA,
          direction, amplitude, amplitudeOnOff, nb_input_neuron, fav_direction,
          duration):
    
    start_scope()
    
    El_bar = -60 * mV
    Gl_bar = 1/(633 * 10 ** 6) * siemens
    Vth_bar = -40 * mV
    Vreset_bar = -55 * mV
    
    El = El_tg = -50 * mV
    Gl = Gl_tg = 10 * nS
    Cm = 0.1 * nF
    Vth = Vth_tg = -30 * mV
    Vreset_tg = -45 * mV
    Vreset = -65 * mV
    delta_th = 5 * mV
    
    SA1_pop = NeuronGroup(75, SA1, method = 'euler', threshold='v > Vth_tg',
                          reset = '''v = Vreset_tg 
                          ISRA_SA1 = ISRA_SA1 + b_SA1''')
    SA2_pop = NeuronGroup(75, SA2, method = 'euler', threshold='v > Vth_tg',
                          reset = '''v = Vreset_tg 
                          ISRA_SA2 = ISRA_SA2 + b_SA2''')
    RA_pop = NeuronGroup(50, RA, method = 'euler', threshold='v > Vth_tg',
                          reset = '''v = Vreset_tg 
                          ISRA_RA = ISRA_RA''')
    TG_pop = NeuronGroup(200, tg, method = 'euler', threshold='v > Vth_tg',
                          reset = '''v = Vreset_tg''')
    PSV_pop = NeuronGroup(400, bar, method = 'euler', threshold='v > Vth_bar',
                          reset = '''v = Vreset_bar''')
    
    S1 = Synapses(SA1_pop, TG_pop, model = 'w : volt', on_pre='v += 25*mV')
    S1.connect(condition='i==j')
    S2 = Synapses(SA2_pop, TG_pop, model = 'w : volt', on_pre='v += 25*mV')
    S2.connect(condition='i+75==j+75')
    S3 = Synapses(RA_pop, TG_pop, model = 'w : volt', on_pre='v += 25*mV')
    S3.connect(condition='i+150==j+150')
    
    S4 = Synapses(TG_pop, PSV_pop, model = 'w : volt', on_pre='v += 25*mV')
    S4.connect(p=0.005)
    S5 = Synapses(PSV_pop, PSV_pop, model = 'w : volt', on_pre='v -= 25*mV')
    S5.connect(condition='i!=j', p=0.005)
    
    trace = StateMonitor(SA1_pop, True, record = True)
    spike_mon = SpikeMonitor(PSV_pop)
    
    SA1_pop.v = El
    SA1_pop.I = 900 * pA
    SA1_pop.fav_direction = fav_direction[:75]
    
    SA2_pop.v = El
    SA2_pop.I = 900 * pA
    SA2_pop.fav_direction = fav_direction[75:150]
    
    RA_pop.v = El
    RA_pop.I = 900 * pA
    RA_pop.fav_direction = fav_direction[150:200]
    
    TG_pop.v = El_tg
    PSV_pop.v = El_bar
    
    print(duration)
    run(duration * msecond)
    
    return spike_mon, trace
    
def produce_motor_input():
	return True

def handle_client(client_socket):
    fav_direction = define_favorite_angles_gauss(90, 225, 45, 1000)
    while True:
	response_data = {'Start' : 'True'}
	response_json = json.dumps(response_data)
	client_socket.send(response_json.encode('utf-8'))
    	
    	data = client_socket.recv(1024)
    	if data:
		json_data = json.loads(data.decode('utf-8'))
		print("Received JSON data:", json_data)
		direction, amplitude, amplitudeOnOff = force_to_time_array_per_timestep(json_data['RA1']['force_vector'])
		
		spikes, _ = model(20.23676859919913 * nS, 69.64621029919265 * pA, 44.089198785866415 * ms,
                46.111299501668576 * nS, 3.213718541167184 * pA, 1882.0521336373106 * ms,
                59.10061355050058 * nS, 86.53424254697113 * pA, 1890.103325320332 * ms,
                direction, amplitude, amplitudeOnOff, 1000, fav_direction, 1)

		plt.plot(range(duration), compute_spontaneous_activity(spikes.t, 1))
		time.sleep(1)
		
    
    client_socket.close()
    

def main():
    host = '127.0.0.1'
    port = 12346
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Server listening on", host, "port", port)
    client_socket, client_address = server_socket.accept()
    print("Connected by", client_address)
    handle_client(client_socket)
        
    server_socket.close()
    
if __name__ == "__main__":
    main()

#direction, amplitude, amplitudeOnOff = create_ramp_and_hold(1123, 3677, 90, 0, 
#                                                            90, 100, duration)
#direction = convert_input_to_time_array(direction)
#amplitude = convert_input_to_time_array(amplitude)
#amplitudeOnOff = convert_input_to_time_array(amplitudeOnOff)

bp_force = import_data()
direction, amplitude, amplitudeOnOff = force_to_time_array(bp_force)
duration = len(direction.values)

nb_input_neuron = 1000
#fav_direction = define_favorite_angles(45, 135, 225, 315, 1000)
fav_direction = define_favorite_angles_gauss(90, 225, 45, 1000)
#fav_direction = define_favorite_angles_uniform(1000)

'''
Magic numbers obtained through fitting each population adaptation rate to
data from Sonekatsu and Gu 2020 (not that magic eh ?)
'''
spikes, _ = model(20.23676859919913 * nS, 69.64621029919265 * pA, 44.089198785866415 * ms,
               46.111299501668576 * nS, 3.213718541167184 * pA, 1882.0521336373106 * ms,
               59.10061355050058 * nS, 86.53424254697113 * pA, 1890.103325320332 * ms,
               direction, amplitude, amplitudeOnOff, nb_input_neuron, fav_direction, duration)


plt.plot(range(duration), compute_spontaneous_activity(spikes.t, duration))
#plt.hist(fav_direction, bins=360)
