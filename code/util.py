# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:16:46 2024

@author: cdrg
"""

import numpy as np
from numba import jit, prange

def source_time_func(freq, amplitude, time, wave_num):
    # sin
    period = 1 / freq
    angular_freq = 2 * np.pi * freq
    source = amplitude * np.sin(angular_freq * time)
    source[time > wave_num * period] = 0
    
    '''  
    # ricker
    tau=np.pi*freq*(time-1/freq)
    source = amplitude*(1.0-2.0*tau**2.0)*np.exp(-tau**2)
    '''
    
    return source


def get_dx(source_freq, source_grid, vp_min):
    return vp_min / (source_freq * source_grid)
    

def get_dt(dx, vp_max, cfl_const=0.5):
    return cfl_const * dx / vp_max   


def reciever_signal(reciever_num, time, sample_rate):
    sample_time = time[::sample_rate]
    sample_index = np.arange(0, len(time), sample_rate)
    
    return sample_index, np.zeros([len(sample_time), reciever_num])


#===================================================================
@jit(nopython=True, cache=True, parallel=True)
def laplace(u0, u1, u2, nz, nx, tau, parallel=True):
    for x in prange(1, nx-1):                                            
        for z in range(1, nz-1):
            u0[z, x] = tau[z, x] * \
                (0.5*(u1[z+1 , x] + u1[z-1 , x] + 
                      u1[z , x+1] + u1[z , x-1]) + 
                 0.25*(u1[z+1 , x+1] + u1[z+1 , x-1] + 
                       u1[z-1 , x+1] + u1[z-1 , x-1]) +
                 -3*u1[z, x])\
                 + 2 * u1[z, x] - u2[z, x]

    return u0               


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def bc(u0, u1, nz, nx, kappa):          
    for x in prange(1, nx-1):
        # bottom
        u0[nz-1, x] = u1[nz-2, x] +\
            (kappa[nz-1, x]-1)/(kappa[nz-1, x]+1) *\
            (u0[nz-2, x] - u1[nz-1, x])
         
        #left
        u0[0, x] = u1[1, x] + \
            (kappa[0, x]-1)/(kappa[0, x]+1) * \
            (u0[1, x] - u1[0, x])
      
    for z in prange(1, nz-1):
        # right
        u0[z, nx-1] = u1[z, nx-2] +\
            (kappa[z, nx-1]-1)/(kappa[z, nx-1]+1) * \
            (u0[z, nx-2] - u1[z, nx-1])
            
        # top
        u0[z, 0] = u1[z, 1] + \
            (kappa[z, 0]-1)/(kappa[z, 0]+1) * \
            (u0[z, 1] - u1[z, 0])
          
    u0[0, 0] = (u1[1, 0] + (kappa[0, 0]-1)/(kappa[0, 0]+1) * (u0[1, 0] - u1[0, 0]) + \
               u1[0, 1] + (kappa[0, 0]-1)/(kappa[0, 0]+1) * (u0[0, 1] - u1[0, 0]))/2
                
    u0[0, nx-1] = (u1[1, nx-1] + (kappa[0, nx-1]-1)/(kappa[0, nx-1]+1) * (u0[1, nx-1] - u1[0, nx-1]) + \
                  u1[0, nx-2] + (kappa[0, nx-1]-1)/(kappa[0, nx-1]+1) * (u0[0, nx-2] - u1[0, nx-1]))/2
    
    u0[nz-1, 0] = (u1[nz-2, 0] + (kappa[nz-1, 0]-1)/(kappa[nz-1, 0]+1) * (u0[nz-2, 0] - u1[nz-1, 0]) + \
                  u1[nz-1, 1] + (kappa[nz-1, 0]-1)/(kappa[nz-1, 0]+1) * (u0[nz-1, 1] - u1[nz-1, 0]))/2
              
    u0[nz-1, nx-1] = (u1[nz-2, nx-1] + (kappa[nz-1, nx-1]-1)/(kappa[nz-1, nx-1]+1) * (u0[nz-2, nx-1] - u1[nz-1, nx-1]) + \
                     u1[nz-1, nx-2] + (kappa[nz-1, nx-1]-1)/(kappa[nz-1, nx-1]+1) * (u0[nz-1, nx-2] - u1[nz-1, nx-1]))/2
                          
    return u0


# wave equation
def wave2d(u, nx, nz, time, src_value, src_xind, src_zind, kappa, tau, src_time,
           receiver_signal, receiver_time_ind, AIR_Z_GRID, reciever_xind, reciever_rate):      
    for t in range(len(time)):
        # source        
        if time[t]<=src_time:
            u[0, src_zind, src_xind] = src_value[t]     
            
        # update  
        u[2] = u[1]
        u[1] = u[0]
        u[0] = laplace(u[0], u[1], u[2], nz, nx, tau)
        
        # Boundary condition
        u[0] = bc(u[0], u[1], nz, nx, kappa)
        
        if t in receiver_time_ind:
            receiver_signal[t//reciever_rate, :] = u[1, src_zind, reciever_xind]

    return u, receiver_signal

 
    
    