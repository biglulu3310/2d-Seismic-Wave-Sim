# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:14:44 2024

@author: cdrg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
# Local
from obspy.io.segy.segy import _read_segy
from util import source_time_func, get_dx, get_dt, reciever_signal, wave2d

#============================================================
# Init
vp_dir = "../data/"
vp_filename = "Vp.segy"  

output_path = "../output/trace/"
output_geo_path = "../output/"

SHOT_NUM = 1
SHOT_OFFSET = 5

#============================================================
# Vp
AIR_Z_GRID = 20
vp = np.array([a.data for a in \
               _read_segy(vp_dir + vp_filename).traces]).T
 
vp = np.vstack((np.full([AIR_Z_GRID, vp.shape[1]], 340), vp))
vp_min = np.min(vp)
vp_max = np.max(vp)

x_dim = vp.shape[1]
z_dim = vp.shape[0]

#============================================================
# Source
src_freq = 8
src_time = 2/src_freq
src_amp = 10
src_wave_num = 1
src_grid = 10

src_xind = x_dim//3
src_zind = AIR_Z_GRID + 20

#============================================================
# Space
dx = get_dx(src_freq, src_grid, vp_min)
dz = dx

x_start = 0
z_start = - AIR_Z_GRID * dx

x = np.array([x_start + i * dx for i in range(x_dim)])
z = np.array([z_start + i * dz for i in range(z_dim)])

#============================================================
# Time
dt = get_dt(dx, vp_max, 1)
time_ = np.arange(0, 4000*dt, dt)

#============================================================
# Reciever
reciever_num = 100
reciever_x_offset = 5 #dx
reciever_rate = 2   #dt
reciever_src_offset = 100 #dx

reciever_xind = np.array([src_xind + reciever_src_offset + \
                          i * reciever_x_offset for i in range(reciever_num)])

reciever_zind = np.full(reciever_xind.shape, src_zind)

#============================================================
# Wave init
source = source_time_func(src_freq, src_amp, time_, src_wave_num)
receiver_time_ind, receiver_signal = reciever_signal(reciever_num, time_, reciever_rate)

kappa = (vp * dt) / dx
tau = kappa ** 2

#============================================================
# output geometry
with open(output_geo_path + "geometry.txt", 'w') as f:
    f.write("#" + "*"*80 + "\n")
    f.write("{0:>30}{1:>30}\n".format("dx(m)", str(dx)))
    f.write("{0:>30}{1:>30}\n".format("dt(s)", str(dt)))
    f.write("#" + "*"*80 + "\n")
    f.write("{0:>30}{1:>30}\n".format("reciever_num", str(reciever_num)))
    f.write("{0:>30}{1:>30}\n".format("reciever_x_offset(dx)", str(reciever_x_offset)))
    f.write("{0:>30}{1:>30}\n".format("reciever_src_offset(dx)", str(reciever_src_offset)))
    f.write("{0:>30}{1:>30}\n".format("reciever_zind(dx)", str(src_zind)))
    f.write("{0:>30}{1:>30}\n".format("reciever_datapoints", str(len(receiver_time_ind))))
    f.write("{0:>30}{1:>30}\n".format("reciever_sample_rate(dt)", str(reciever_rate)))
    f.write("#" + "*"*80 + "\n")
    f.write("{0:>30}{1:>30}\n".format("shot_num", str(SHOT_NUM)))
    f.write("{0:>30}{1:>30}\n".format("shot_offset(dx)", str(SHOT_OFFSET)))
    f.write("{0:>30}{1:>30}\n".format("shot_zind(dx)", str(src_zind)))
    
#============================================================
# Wave
for s in range(SHOT_NUM):  
    src_xind = x_dim//3 - SHOT_OFFSET * s
    reciever_xind = np.array([src_xind + reciever_src_offset + \
                              i * reciever_x_offset for i in range(reciever_num)])
      
    u_init = np.zeros((3 ,z_dim, x_dim))
    u, rec = wave2d(u_init, x_dim, z_dim, time_, source, src_xind, src_zind, kappa, tau,
               src_time, receiver_signal, receiver_time_ind, AIR_Z_GRID, reciever_xind, reciever_rate)
      
    u_max = np.max(abs(u[1]))
    
    output_path_src = output_path + str(src_xind) + "/"   
    os.makedirs(output_path_src, exist_ok=True)
    
    #============================================================
    # Output
    fig = plt.figure(figsize=(15, 7))
    ax0 = plt.subplot2grid((1, 1), (0, 0))
    
    divider1 = make_axes_locatable(ax0)
    cax = divider1.append_axes("right", size="1.5%", pad=0.25)
    cax_r = divider1.append_axes("right", size="1.5%", pad=1.3)
    
    u_plot = u[1].copy()
    #u_plot[abs(u[1])<u_max*0.08] = np.nan
    
    vp_contour = ax0.contourf(x, z, vp, 50, cmap='gray_r', alpha=0.7, zorder=0)
    ax0.contour(x, z, vp, 60, colors='k', linewidths=1.1, zorder=1)
    vp_cbar = plt.colorbar(vp_contour, orientation='vertical', cax=cax)
    vp_cbar.set_label("velocity (m/s)", fontsize=17, labelpad=10)
    vp_cbar.ax.tick_params(labelsize=14)
    
    wave_contour = ax0.imshow(u_plot, cmap='coolwarm', vmin=-u_max*0.3, vmax=u_max*0.3, alpha=0.65, \
               origin='lower', extent = (x[0], x[-1], z[0], z[-1]), zorder=2)
        
    wave_contour = plt.colorbar(wave_contour, orientation='vertical', cax=cax_r)
      
    wave_contour.set_label("amplitude", fontsize=17, labelpad=10)
    wave_contour.ax.tick_params(labelsize=14)
    
    ax0.axhline(y=0, c='k', lw=2,zorder=3)
    
    ax0.scatter(x[src_xind], z[src_zind], marker='*', \
                        edgecolor=[0.8, 0.2, 0.2], facecolor=[1, 0.9, 0.2], s=400, lw=1.4, zorder=4)   
    
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5))  
    ax0.xaxis.set_minor_locator(AutoMinorLocator(5))
      
    ax0.set_title("Wave Simulation",fontsize=30,pad=20) 
    ax0.set_xlabel("x-direction (m)",fontsize=17,labelpad=10)
    ax0.set_ylabel("z-direction (m)",fontsize=17,labelpad=10)
    
    ax0.tick_params(axis='both', which='major',direction='out',\
                    bottom=True,top=False,right=False,left=True,\
                    length=10, width=2, labelsize=15,pad=10)
        
    ax0.tick_params(axis='both', which='minor',direction='out',\
                    bottom=True,top=False,right=False,left=True,\
                    length=5, width=1.5, labelsize=15,pad=10)
        
    [ax0.spines[b].set_linewidth(1.5) for b in ['bottom','left','right']]
    [ax0.spines[b].set_color("black") for b in ['bottom','left','right']]
    ax0.spines['top'].set_linewidth(0)  
    
    ax0.invert_yaxis()
    ax0.set_aspect('equal')
    ax0.set_ylim(z[-1], z[0] - 0.15 * z_dim * dz)
    
    plt.tight_layout()
    plt.savefig(output_path_src + "wave.png", dpi=400)
    
    #========================================================
    fig = plt.figure(figsize=(13, 20))
    ax0 = plt.subplot2grid((1, 1), (0, 0))
    
    for i, a in enumerate(rec.T): 
        a = a * 120
        ax0.plot(a + x[reciever_xind[i]], time_[receiver_time_ind], c='k', lw=0.8)
        ax0.fill_betweenx(time_[receiver_time_ind], x[reciever_xind[i]], a + x[reciever_xind[i]], \
                          where=(a + x[reciever_xind[i]]>x[reciever_xind[i]]),color='r')
            
        ax0.fill_betweenx(time_[receiver_time_ind], x[reciever_xind[i]], a + x[reciever_xind[i]], \
                          where=(a + x[reciever_xind[i]]<x[reciever_xind[i]]),color='b')
        
    ax0.yaxis.set_minor_locator(AutoMinorLocator(10))  
    ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax0.set_xlim(x[reciever_xind[0]], x[reciever_xind[-1]]) 
    ax0.set_ylim(time_[receiver_time_ind[0]], time_[receiver_time_ind[-1]]) 

    ax0.tick_params(axis='both', which='major',direction='out',\
                            bottom=True,top=False,right=False,left=True,\
                            length=15, width=3.1, labelsize=20,pad=12)
                
    ax0.tick_params(axis='both', which='minor',direction='out',\
                    bottom=True,top=False,right=False,left=True,\
                    length=9, width=2.5, labelsize=20,pad=12)
        
    [ax0.spines[b].set_linewidth(2.4) for b in ['top', 'bottom','left','right']]
    [ax0.spines[b].set_color("black") for b in ['top', 'bottom','left','right']]

    ax0.set_title("Wiggle Traces",fontsize=45,pad=20) 
    ax0.set_xlabel("x-direction (m)",fontsize=25,labelpad=15)
    ax0.set_ylabel("travel-time (s)",fontsize=25,labelpad=15)

    ax0.invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_path_src + "wiggle.png", dpi=400)

    for i in range(len(reciever_xind)):
        with open(output_path_src+str(reciever_xind[i])+".txt", 'w') as f:
            for t in range(len(receiver_time_ind)):
                f.write(str(time_[receiver_time_ind[t]]) + "    " + str(receiver_signal[t, i]) + "\n")

