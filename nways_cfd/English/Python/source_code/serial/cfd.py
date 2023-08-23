# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
#!/usr/bin/env python
#
# CFD Calculation
# ===============
#
# Simulation of inviscid flow in a 2D box using the Jacobi algorithm.
#
# Python version - uses numpy and loops
#
# EPCC, 2014
#
import sys
import time

# Import numpy
import numpy as np
import math
import sys
import cupy.cuda.nvtx as nvtx
from numba import njit, jit

def main(argv):
    printfreq = 1000 #output frequency
    error = bnorm = 0.0
    tolerance = 0.0 #tolerance for convergence. <=0 means do not check

    error = 0.0
    # Set the minimum size parameters
    mbase = 32
    nbase = 32
    bbase = 10
    hbase = 15
    wbase = 5

    irrotational = 1
    checkerr = 0
    iter = 0


    # Test we have the correct number of arguments
    if len(argv) < 2:
        sys.stdout.write("Usage: cfd.py <scalefactor> <iterations>")
        sys.exit(1)

    # Get the systen parameters from the arguments
    scalefactor = int(argv[0])
    niter = int(argv[1])

    sys.stdout.write("\n2D CFD Simulation\n")
    sys.stdout.write("=================\n")
    sys.stdout.write("Scale factor = {0}\n".format(scalefactor))
    sys.stdout.write("Iterations   = {0}\n".format(niter))

    # do we stop because of tolerance?
    if (tolerance > 0):
        checkerr = 1

    # check command line parameters and parse them
    if (len(argv) < 2 or len(argv) > 3):
        print("Usage: cfd <scale> <numiter> [reynolds]\n")
        return 0

    scalefactor = int(argv[0])
    numiter = int(argv[1])

    if len(argv) == 3:
        re = float(argv[2])
        irrotational = 0
    else:
        re = -1.0

    if not checkerr:
        print("Scale Factor = {}, iterations = {}\n".format(scalefactor, numiter))
    else:
        print("Scale Factor = {}, iterations = {}, tolerance= {}\n".format(scalefactor, numiter, tolerance))

    if (irrotational):
        print("Irrotational flow\n")
    else:
        print("Reynolds number = {}\n".format(re))

    # Set the parameters for boundary conditions
    #Calculate b, h & w and m & n
    b = bbase * scalefactor
    h = hbase * scalefactor
    w = wbase * scalefactor
    m = mbase * scalefactor
    n = nbase * scalefactor

    re = re / float(scalefactor)

    # Write the simulation details
    sys.stdout.write("\nRunning CFD on {0} x {1} grid in serial\n".format(m, n))

    # allocate arrays
    nvtx.RangePush("Initialization")
    psi = np.zeros(((m + 2) * (n + 2)), dtype=np.float64)
    nvtx.RangePop()
    psitmp = np.zeros(psi.size, dtype=np.float64)

    if (not irrotational):
        # allocate arrays
        nvtx.RangePush("Initialization")
        zet = np.zeros(((m + 2) * (n + 2)), dtype=np.float64)
        nvtx.RangePop()
        zettmp = np.zeros(((m + 2) * (n + 2)), dtype=np.float64)

    nvtx.RangePush("Boundary_PSI")
    #set the psi boundary conditions
    psi = boundarypsi(psi, m, n, b, h, w)
    nvtx.RangePop() 

    #compute normalisation factor for error
    bnorm = 0.0
    nvtx.RangePush("Compute_Normalization")
    for i in range(m + 2):
        for j in range(n + 2):
            bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j]
    nvtx.RangePop()
    # boundary set for zet
    if not irrotational:
        zet = boundaryzet(zet, psi, m, n)
        nvtx.RangePush("Compute_Normalization")
        for i in range(m + 2):
            for j in range(n + 2):
                bnorm += zet[i * (m + 2) + j] * zet[i * (m + 2) + j]
        nvtx.RangePop()
    bnorm = math.sqrt(bnorm)

    #begin iterative Jacobi loop
    print("\nStarting main loop...\n\n")
    tstart = time.time()

    nvtx.RangePush("Overall_Iteration")
    for iter in range(1, numiter+1):
        nvtx.RangePush("JacobiStep")
        if (irrotational): #calculate psi for next iteration
            psitmp = jacobistep(psitmp, psi, m, n)
        else:
            psitmp,zettmp = jacobistepvort(zettmp, psitmp, zet, psi, m, n, re)
        nvtx.RangePop() 
        nvtx.RangePush("Calculate_Error")
        #calculate current error if required
        if checkerr or iter == numiter:
            error = deltasq(psitmp, psi, m, n)
            if not irrotational:
                error += deltasq(zettmp, zet, m, n)

            error = math.sqrt(error)
            error = error / bnorm
        nvtx.RangePop()
        #quit early if we have reached required tolerance
        if checkerr:
            if error < tolerance:
                print("Converged on iteration {0}\n".format(iter))
                break
        #copy back
        nvtx.RangePush("Switch_Array")
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                psi[i * (m + 2) + j] = psitmp[i * (m + 2) + j]
        
        if not irrotational:
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    zet[i * (m + 2) + j] = zettmp[i * (m + 2) + j]
        nvtx.RangePop()
        if not irrotational:
            # update zeta BCs that depend on psi
            boundaryzet(zet, psi, m, n)

        # print loop information
        if iter % printfreq == 0:
            if not checkerr:
                print("Completed iteration {0}\n".format(iter))
            else:
                print("Completed iteration {0}, error = {1}\n".format(iter, error))
    nvtx.RangePop()
    if iter > numiter:
        iter=numiter
    tstop = time.time()
    ttot = tstop - tstart
    titer = ttot / float(iter)

    #print out some stats
    print("\n... finished\n")
    print("\nCalculation took {0:.5f}s\n\n".format(ttot))
    print("After {0} iterations, the error is {1}\n".format(niter, error))
    print("Time for {0} iterations was {1} seconds\n".format(niter, ttot))
    print("Each iteration took {0} seconds\n".format(titer))

    # Write the output files for subsequent visualisation
    nvtx.RangePush("output visualization")
    write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat")
    nvtx.RangePop()

    # Finish nicely
    sys.exit(0)


def write_data(m, n, scale, psi, velfile, colfile):

    # Open the specified files
    velout = open(velfile, "w")
    velout.write("{0} {1}\n".format(m/scale, n/scale))
    colout = open(colfile, "w")
    colout.write("{0} {1}\n".format(m, n))

    # Loop over stream function array (excluding boundaries)
    for i in range(0, m):
        for j in range(0, n):

            # Compute velocities and magnitude
            ux =  (psi[(i+1)*(m+2)+j+2]-psi[(i+1)*(m+2)+j])/2.0
            uy = -(psi[(i+2)*(m+2)+j+1]-psi[i*(m+2)+j+1])/2.0
            #umod = (ux**2 + uy**2)
            umod = (ux ** 2 + uy ** 2) ** 0.5

            # We are actually going to output a colour, in which
            # case it is useful to shift values towards a lighter
            # blue (for clarity) via the following kludge...
            hue = umod ** 0.6
            #hue = math.pow(umod, 0.4)
            colout.write("{0:5d} {1:5d} {2:10.5f}\n".format(i, j, hue))

            # Only write velocity vectors every "scale" points
            if (i-1)%scale == (scale-1)/2 and (j-1)%scale == (scale-1)/2:
                velout.write("{0:5d} {1:5d} {2:10.5f} {3:10.5f}\n".format(i-1, j-1, ux, uy))

    velout.close()
    colout.close()

@jit()
def jacobistep(psinew, psi, m, n):
    for i in range(1, m+1):
        for j in range(1, n+1):
            psinew[i * (m + 2) + j]=0.25 * (psi[(i-1) * (m+2)+j]+psi[(i+1) * (m+2)+j]+psi[i * (m+2)+j-1]+psi[i * (m+2)+j+1])
    return psinew

@jit()
def jacobistepvort(zetnew, psinew,zet,psi,m,n,re):
    for i in range(1, m+1):
        for j in range(1, n+1):
            psinew[i * (m + 2) + j]=0.25 * (psi[(i-1) * (m+2)+j]+psi[(i+1) * (m+2)+j]+psi[i * (m+2)+j-1]+psi[i * (m+2)+j+1]- zet[i * (m+2)+j])

    for i in range(1, m+1):
        for j in range(1, n+1):
            zetnew[i * (m + 2) + j] = 0.25 * (zet[(i - 1) * (m + 2) + j] + zet[(i + 1) * (m + 2) + j] + zet[i * (m + 2) + j - 1] + zet[i * (m + 2) + j + 1])
            - re / 16.0 * ((psi[i * (m + 2) + j + 1] - psi[i * (m + 2) + j - 1]) * (zet[(i + 1) * (m + 2) + j] - zet[(i - 1) * (m + 2) + j])
                    - (psi[(i + 1) * (m + 2) + j] - psi[(i - 1) * (m + 2) + j]) * (zet[i * (m + 2) + j + 1] - zet[i * (m + 2) + j - 1]))

    return psinew, zetnew

@jit()
def deltasq (newarr, oldarr, m, n):
    dsq = 0.0
    for i in range(1, m+1):
        for j in range(1, n+1):
            tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
            dsq += tmp * tmp

    return dsq

@jit()
def boundarypsi(psi,m,n,b,h,w):
    # Set the boundary conditions on bottom edge

    for i in range(b+1, b+w):
        psi[i*(m+2)+0] = float(i-b)

    for i in range(b + w, m + 1):
        psi[i*(m+2)+0] = float(w)

    # Set the boundary conditions on right edge
    for j in range(1, h + 1):
        psi[(m+1)*(m+2)+j] = float(w)

    for j in range(h + 1, h + w):
        psi[(m+1)*(m+2)+j] = float(w - j + h)

    return psi

@jit()
def boundaryzet(zet, psi, m, n):
    # set top/bottom BCs:
    for i in range(1, m + 1):
        zet[i * (m + 2) + 0] = 2.0 * (psi[i * (m + 2) + 1] - psi[i * (m + 2) + 0])
        zet[i * (m + 2) + n + 1] = 2.0 * (psi[i * (m + 2) + n] - psi[i * (m + 2) + n + 1])

    # set left BCs:
    for j in range(1, n + 1):
        zet[0 * (m + 2) + j] = 2.0 * (psi[1 * (m + 2) + j] - psi[0 * (m + 2) + j])

    # set right BCs
    for j in range(1, n + 1):
        zet[(m + 1) * (m + 2) + j] = 2.0 * (psi[m * (m + 2) + j] - psi[(m + 1) * (m + 2) + j])

    return  zet

if __name__ == "__main__":
        main(sys.argv[1:])