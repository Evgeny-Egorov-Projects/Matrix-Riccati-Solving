# -*- coding: utf-8 -*-

"""
    File Name : Quasiclassics functions.py

    Purpose : Puttting functions that calculate Green's functions via Ricatti amplitudes in one place. 

    Special note: Step choice (defines your grid and slices qp trajectory over it) and OrderParameter
    (slices OP according to grid) should be specified within the program as they are crucial input for
     functions below, but too special for each given problem.

     Notations used in functions are copied from Matthias Eschrig group material on Ricatti equation solving.

    Creation Date : 06.10.2016

    Author : Eugene Egorov
"""
# Import external libraries
from decimal import *
import numpy as np
import scipy as sp
from scipy import linalg
from numpy import conjugate as conj
import time
from typing import Any


def inverse2by2(matrix):
    # straightforward 2x2 matrix inverse function
    val = (1 / (matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]) *
        np.c_[matrix[1, 1], -matrix[0, 1], -matrix[1, 0], matrix[0, 0]])\
        .reshape(2,2)
    return val


def homogeneous_gamma(energy, delta, delta_tilda):
    e = energy
    val = np.empty((delta.shape[2], 2, 2), dtype='complex')
    for k in np.arange(delta[0, 0, :].size):
        Matrix11 = np.r_[np.c_[-e * np.identity(2), np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex')], [
            np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')]]
        Matrix22 = np.r_[[np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')], np.c_[
            np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex'), e * np.identity(2)]]
        Matrix12 = np.r_[np.c_[np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex'), delta_tilda[:, :, k]], [
            np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')]]
        Matrix21 = np.r_[[np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')], np.c_[
            delta[:, :, k], np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex')]]
        Matrix = Matrix11 + Matrix12 + Matrix21 + Matrix22
        if np.any(np.isnan(Matrix)):
            print(Matrix)
        U = U_inverse = np.identity(4, dtype='complex')
        maximum = 1
        dec = 16
        precision = 5 * 1e-14
        cnt = 0
        while maximum > precision:
            i, j = np.unravel_index(np.argmax(np.abs(sp.linalg.tril(Matrix, -1))), (4, 4))
            m = np.array([[Matrix[j, j], Matrix[i, j]], [Matrix[j, i], Matrix[i, i]]], dtype='complex')
            u = np.diag([1.0 + 0.0j] * 4)
            u_inverse = np.diag([1.0 + 0.0j] * 4)
            epsi = 0.5 * (m[1, 1] - m[0, 0])
            if cnt > 1000:
                print('Im stuck!', maximum)
                print(i, j)
                print(Matrix)
                Matrix = np.around(Matrix, dec)
            if np.absolute(m[0, 1] * m[1, 0]) < precision ** 2:
                if np.absolute(m[0, 1]) < precision:
                    u[i, i] = u_inverse[i, i] = 0.0 + 0.0j
                    u[i, j] = u_inverse[i, j] = 1.0 + 0.0j
                    u[j, i] = u_inverse[j, i] = 1.0 + 0.0j
                    u[j, j] = u_inverse[j, j] = 0.0 + 0.0j
                else:
                    u = np.identity(4, dtype='complex')
                    u_inverse = np.identity(4, dtype='complex')
            else:
                sq = np.around(epsi + 1j * np.sqrt(-1.0 * m[0, 1] * m[1, 0] - epsi * epsi), dec)
                if np.abs(sq) == 0:
                    print(m)
                    print('sq= ', sq)
                else:
                    t1 = -1.0 * np.around(m[1, 0] / sq, dec)
                    t2 = -1.0 * np.around(m[0, 1] / sq, dec)
                    coef = np.around(1.0 / np.sqrt(1.0 + t1 * t2), dec)
                    u[i, i] = u_inverse[i, i] = u[j, j] = u_inverse[j, j] = coef
                    u[j, i] = np.around(t1 * coef, dec)
                    u_inverse[j, i] = -1.0 * np.around(t1 * coef, dec)
                    u[i, j] = -1.0 * np.around(t2 * coef, dec)
                    u_inverse[i, j] = np.around(t2 * coef, dec)
            cnt = cnt + 1
            Matrix = np.dot(u, np.dot(Matrix, u_inverse))
            maximum = np.amax(np.abs(sp.linalg.tril(Matrix, -1)))
            U = np.dot(u, U)
            U_inverse = np.dot(U_inverse, u_inverse)
        A = np.array(
            [[U[2, 2], 0, U[2, 3], 0], [0, U[2, 2], 0, U[2, 3]], [U[3, 2], 0, U[3, 3], 0], [0, U[3, 2], 0, U[3, 3]]],
            dtype='complex')
        if np.any(np.isnan(A)):
            print(A)
            print(Matrix)
        b = -U[2:, :2].reshape(4)
        gam = np.linalg.solve(A, b)
        val[k] = gam.reshape(2, 2)
    return val


def homogeneous_greens_function(energy, delta, delta_tilda):
    e = energy
    val = np.empty((delta.shape[2], 2, 2), dtype='complex')
    # valt=np.empty((delta.shape[2],2,2),dtype='complex')
    Mone = np.identity(2, dtype='complex')
    for k in np.arange(delta[0, 0, :].size):
        Matrix11 = np.r_[np.c_[e * Mone, np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex')], [
            np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')]]
        Matrix22 = np.r_[[np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')], np.c_[
            np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex'), -e * Mone]]
        Matrix12 = np.r_[np.c_[np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex'), -delta[:, :, k]], [
            np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')]]
        Matrix21 = np.r_[[np.zeros(4, dtype='complex')], [np.zeros(4, dtype='complex')], np.c_[
            -delta_tilda[:, :, k], np.zeros(2, dtype='complex'), np.zeros(2, dtype='complex')]]
        Matrix = Matrix11 + Matrix12 + Matrix21 + Matrix22
        evalues = sp.linalg.eig(Matrix)
        gamma_homogeneous = -1j * np.pi * evalues[1].dot(sp.diag(np.sign(evalues[0].imag))).dot(sp.linalg.inv(evalues[1]))
        val[k] = sp.linalg.inv(gamma_homogeneous[:2, :2] - 1j * np.pi * Mone).dot(gamma_homogeneous[:2, 2:])
        # val[k]=sp.linalg.inv(GH[:2,:2]-1j*np.pi*Mone,check_finite=False).dot(GH[:2,2:])
        # valt[k]=conj(-sp.linalg.inv(1j*np.pi*np.identity(2)+GH[2:,2:]).dot(GH[2:,:2]))
        # direct conjugate for backwards trajectory
    return val  # ,valt


def omegas(energy, intersections, gammahom, delta_tilda):
    gh = sp.linalg.block_diag(*gammahom)
    # times=1j*np.diag(intersections)
    times = 0.5 * 1j * np.diag(intersections)
    # 0.5 for double usage in calculate routines
    delta_tilda4by4 = sp.linalg.block_diag(*delta_tilda.T.reshape(gammahom.shape[0], 2, 2))
    en = energy * np.diag(np.ones(intersections.size))
    solution = np.empty((gammahom.shape[0], 2, 2), dtype='complex')
    omega1 = en - gh.dot(delta_tilda4by4)
    exponent1 = times.dot(omega1)
    omega2 = en - delta_tilda4by4.dot(gh)
    exponent2 = times.dot(omega2)
    mu1 = 0.5 * (sp.diag(exponent1)[::2] + sp.diag(exponent1)[1::2])
    mu2 = 0.5 * (sp.diag(exponent2)[::2] + sp.diag(exponent2)[1::2])
    Mone= np.diag([1.0 + 0.0j] * 2) #identity matrix of given size - creates it faster than standard numpy
    for k in np.arange(delta_tilda.shape[2]):
        b = delta_tilda4by4[2 * k:2 * k + 2, 2 * k:2 * k + 2].reshape(4)
        A = sp.linalg.block_diag(omega1[2 * k:2 * k + 2, 2 * k:2 * k + 2].T, omega1[2 * k:2 * k + 2, 2 * k:2 * k + 2].T)\
            + np.c_[np.array([omega2[2 * k, 2 * k], 0, omega2[2 * k, 2 * k + 1], 0]),\
                    np.array([0, omega2[2 * k, 2 * k], 0, omega2[2 * k, 2 * k + 1]]), \
                    np.array([omega2[2 * k + 1, 2 * k], 0, omega2[2 * k + 1, 2 * k + 1], 0]),\
                    np.array([0, omega2[2 * k + 1, 2 * k], 0, omega2[2 * k + 1, 2 * k + 1]])]
        solution[k] = np.linalg.solve(A, b).reshape(2, 2)
        omega1power = exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] - mu1[k] * Mone
        omega2power = exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] - mu2[k] * Mone
        root1 = np.sqrt(omega1power[0, 0] * omega1power[1, 1] - omega1power[1, 0] * omega1power[0, 1])
        root2 = np.sqrt(omega2power[0, 0] * omega2power[1, 1] - omega2power[1, 0] * omega2power[0, 1])
        if np.isnan(float(np.abs(np.cos(root1)))):
            #If something numerically wrong
            print(omega1power, omega2power)
            print(mu1, mu2)
            print('cos, sin', np.cos(root1), np.sin(root2))
        #here is used fast way to obtain matrix exponential for 2x2 matrices
        if np.abs(root1) == 0.0:
            exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu1[k]) * (Mone + omega1power)
        else:
            exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu1[k]) * (np.cos(root1) * Mone + np.sin(root1) *
                                                                            omega1power / root1)
        if np.abs(root2) == 0.0:
            exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu2[k]) * (Mone + omega2power)
        else:
            exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu2[k]) * (np.cos(root2) * Mone + np.sin(root2)
                                                                            * omega2power / root2)
        # solution[k]=sp.linalg.solve_sylvester(Om2[2*k:2*k+2,2*k:2*k+2],Om1[2*k:2*k+2,2*k:2*k+2],DT[2*k:2*k+2,2*k:2*k+2])
    w = sp.linalg.block_diag(*solution)
    U_homogeneous = sp.linalg.block_diag(exponent1)
    V_homogeneous = sp.linalg.block_diag(exponent2)
    W_homogeneous = V_homogeneous.dot(w).dot(U_homogeneous) - w
    val = U_homogeneous, V_homogeneous, W_homogeneous
    return val


def omegas_for_real_energies(energy, intersections, gammahom, delta_tilda):
    gh = sp.linalg.block_diag(*gammahom)
    times = 1j * np.diag(intersections)
    # times=0.5*1j*np.diag(intersections)
    # 0.5 for double usage in calculate routines
    delta_tilda4by4 = sp.linalg.block_diag(*delta_tilda.T.reshape(gammahom.shape[0], 2, 2))
    en = energy * np.diag(np.ones(intersections.size))
    solution = np.empty((gammahom.shape[0], 2, 2), dtype='complex')
    omega1 = en - gh.dot(delta_tilda4by4)
    exponent1 = times.dot(omega1)
    omega2 = en - delta_tilda4by4.dot(gh)
    exponent2 = times.dot(omega2)
    mu1 = 0.5 * (sp.diag(exponent1)[::2] + sp.diag(exponent1)[1::2])
    mu2 = 0.5 * (sp.diag(exponent2)[::2] + sp.diag(exponent2)[1::2])
    Mone= np.diag([1.0 + 0.0j] * 2)
    # identity matrix of given size - creates it faster than standard numpy
    for k in np.arange(delta_tilda.shape[2]):
        b = delta_tilda4by4[2 * k:2 * k + 2, 2 * k:2 * k + 2].reshape(4)
        A = sp.linalg.block_diag(omega1[2 * k:2 * k + 2, 2 * k:2 * k + 2].T, omega1[2 * k:2 * k + 2, 2 * k:2 * k + 2].T)\
            + np.c_[np.array([omega2[2 * k, 2 * k], 0, omega2[2 * k, 2 * k + 1], 0]),\
                    np.array([0, omega2[2 * k, 2 * k], 0, omega2[2 * k, 2 * k + 1]]), \
                    np.array([omega2[2 * k + 1, 2 * k], 0, omega2[2 * k + 1, 2 * k + 1], 0]),\
                    np.array([0, omega2[2 * k + 1, 2 * k], 0, omega2[2 * k + 1, 2 * k + 1]])]
        solution[k] = np.linalg.solve(A, b).reshape(2, 2)
        omega1power = exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] - mu1[k] * Mone
        omega2power = exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] - mu2[k] * Mone
        root1 = np.sqrt(omega1power[0, 0] * omega1power[1, 1] - omega1power[1, 0] * omega1power[0, 1])
        root2 = np.sqrt(omega2power[0, 0] * omega2power[1, 1] - omega2power[1, 0] * omega2power[0, 1])
        if np.isnan(float(np.abs(np.cos(root1)))):
            # If something numerically wrong
            print(omega1power, omega2power)
            print(mu1, mu2)
            print('cos, sin', np.cos(root1), np.sin(root2))
        # here is used fast way to obtain matrix exponential for 2x2 matrices
        if np.abs(root1) == 0.0:
            exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu1[k]) * (Mone + omega1power)
        else:
            exponent1[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu1[k]) * (
                        np.cos(root1) * Mone + np.sin(root1) *
                        omega1power / root1)
        if np.abs(root2) == 0.0:
            exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu2[k]) * (Mone + omega2power)
        else:
            exponent2[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.exp(mu2[k]) * (
                        np.cos(root2) * Mone + np.sin(root2)
                        * omega2power / root2)
        # solution[k]=sp.linalg.solve_sylvester(Om2[2*k:2*k+2,2*k:2*k+2],Om1[2*k:2*k+2,2*k:2*k+2],DT[2*k:2*k+2,2*k:2*k+2])
    w = sp.linalg.block_diag(*solution)
    U_homogeneous = sp.linalg.block_diag(exponent1)
    V_homogeneous = sp.linalg.block_diag(exponent2)
    W_homogeneous = V_homogeneous.dot(w).dot(U_homogeneous) - w
    val = U_homogeneous, V_homogeneous, W_homogeneous
    return val


def propagate(gamma_initial, gamma_homomogeneous_solution, omegas_output, scattering_matrix):
    s=scattering_matrix
    # profile=np.zeros(gamhom.shape,dtype='complex')
    hom = gamma_homomogeneous_solution
    U_hom, V_hom, W_hom = omegas_output
    # determines where reflection from the boundaries happen
    boundary1 = int(hom.shape[0] / 2)
    boundary2 = hom.shape[0]
    Mone = np.diag([1.0 + 0.0j] * 2)
    # check if indexes are in order (array length is even)
    if 2 * boundary1 - boundary2 != 0.0: print(boundary1, boundary2)
    for ro in np.arange(hom.shape[0]):
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro == boundary1 - 1 or ro == boundary2:
            gamma_initial = s.dot(gamma_initial).dot(conj(s))
        # profile[i]=gamma_initial
    value=gamma_initial
    return value


def calculate_density_of_states(gamma, gamma_tilde):
    val = np.ones(gamma[:, 0, 0].size)
    # gt=conj(gt) #sometimes required if conjugation is not done for backwards trajectory
    Mone = np.diag([1.0 + 0.0j] * 2)
    for i in np.arange(gamma[:, 0, 0].size):
        gamma_times_gamma_tilde = gamma[i].dot(conj(gamma_tilde[i]))
        if np.linalg.cond(Mone - gamma_times_gamma_tilde) < 1.0 / np.finfo(gamma_times_gamma_tilde.dtype).eps:
            normal_state = inverse2by2(Mone - gamma_times_gamma_tilde)
            val[i] = 0.5 * np.sum(np.diag(1j * normal_state.dot(Mone + gamma_times_gamma_tilde)).imag)
        else:
            val[i] = -1
        # if val[i]<0: flag=True print ("DoS<0!!! not again! ",i);
    return val


def calculate_spectral_density_of_states(gamma, gamma_tilde, sigma, projection):
    val = np.ones((2, gamma[:, 0, 0].size))
    greens_function = np.empty((2, 2), dtype='complex')
    # prevents error of casting complex variables on float array
    Mone = np.diag([1.0 + 0.0j] * 2)
    for i in np.arange(gamma[:, 0, 0].size):
        gamma_times_gamma_tilde = gamma[i].reshape(2, 2).dot(conj(gamma_tilde[i].reshape((2, 2))))
        greens_function = 1j * inverse2by2(Mone - gamma_times_gamma_tilde).dot(Mone + gamma_times_gamma_tilde)
        # G=1j*sp.linalg.inv(Mone-ggt,check_finite=False).dot(Mone+ggt)
        # Dos=0.5*np.sum(np.diag(G).imag)
        if sigma == 'sx':
            val[:, i] = 0.25 * (np.sum(np.diag(greens_function)) + greens_function[0, 1] + greens_function[1, 0]).imag, \
                        0.25 * (np.sum(np.diag(greens_function)) - greens_function[0, 1] - greens_function[1, 0]).imag
        elif sigma == 'sy':
            val[:, i] = 0.25 * (np.sum(np.diag(greens_function)) + 1j * greens_function[0, 1] - 1j * greens_function[1, 0]).imag, \
                        0.25 * (np.sum(np.diag(greens_function)) - 1j * greens_function[0, 1] + 1j * greens_function[1, 0]).imag
        elif sigma == 'sz':
            val[:, i] = 0.5 * np.diag(greens_function).imag
        else:
            print('wrong string')
    return val[:, projection]


def gap_equation(gamma, gamma_tilde, pendulum, m):
    costet = pendulum[:, 2].reshape(m, m)
    if costet[0, 0] < 0:
        av_coefficient = 1.0
    else:
        av_coefficient = -1.0
    # if costet is from -1 to 1 delete minus in coef or change if condition
    dphi = 2 * np.pi / (m - 1)
    px = pendulum[:, 0].reshape(m, m)
    py = pendulum[:, 1].reshape(m, m)
    pz = pendulum[:, 2].reshape(m, m)
    val = np.zeros((3, 3), dtype='complex')
    Fx = np.ones(gamma[:, 0, 0].shape, dtype='complex')
    Fy = np.ones(gamma[:, 0, 0].shape, dtype='complex')
    Fz = np.ones(gamma[:, 0, 0].shape, dtype='complex')
    F0 = np.ones(gamma[:, 0, 0].shape, dtype='complex')
    # Commented lines serve to check if numerical integration done correctly, all values should be 1.0
    # norm=AvCoef*sp.integrate.simps(sp.integrate.simps(np.ones((m,m)),dx=dphi),costet[:,0])/(4*np.pi)
    # normx=3*AvCoef*sp.integrate.simps(sp.integrate.simps(px**2,dx=dphi),costet[:,0])/(4*np.pi)
    # normz=3*AvCoef*sp.integrate.simps(sp.integrate.simps(pz**2,dx=dphi),costet[:,0])/(4*np.pi)
    # print(norm,normx,normz)
    coef = av_coefficient * 3.0 / (2 * np.pi)
    Mone = np.diag([1.0 + 0.0j] * 2)
    for i in np.arange(gamma[:, 0, 0].shape[0]):
        gamma_times_gamma_tilde = gamma[i].reshape(2, 2).dot(conj(gamma_tilde[i].reshape((2, 2))))
        F = np.pi * -2j * inverse2by2(Mone - gamma_times_gamma_tilde).dot(gamma[i].reshape(2, 2))
        # i sigma_Y convention
        Fx[i] = 0.5 * (np.diag(F)[-1] - np.diag(F)[0])
        Fz[i] = 0.5 * (F[0, 1] + F[1, 0])
        Fy[i] = -0.5j * np.sum(np.diag(F))
        F0[i] = 0.5 * (F[0, 1] - F[1, 0])
    # lets reshape everything to 2D!
    Fx = Fx.reshape(m, m)
    Fy = Fy.reshape(m, m)
    Fz = Fz.reshape(m, m)
    F0 = F0.reshape(m, m)
    singlet = -1.0 * sp.integrate.simps(sp.integrate.simps(F0, dx=dphi), costet[:, 0])
    # if np.abs(singlet)>1e-9:
    #    print('singlet!',singlet)

    val[0, 0] = sp.integrate.simps(sp.integrate.simps(px * Fx, dx=dphi), costet[:, 0])
    val[0, 1] = sp.integrate.simps(sp.integrate.simps(py * Fx, dx=dphi), costet[:, 0])
    val[0, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fx, dx=dphi), costet[:, 0])

    val[1, 0] = sp.integrate.simps(sp.integrate.simps(px * Fy, dx=dphi), costet[:, 0])
    val[1, 1] = sp.integrate.simps(sp.integrate.simps(py * Fy, dx=dphi), costet[:, 0])
    val[1, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fy, dx=dphi), costet[:, 0])

    val[2, 0] = sp.integrate.simps(sp.integrate.simps(px * Fz, dx=dphi), costet[:, 0])
    val[2, 1] = sp.integrate.simps(sp.integrate.simps(py * Fz, dx=dphi), costet[:, 0])
    val[2, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fz, dx=dphi), costet[:, 0])
    # print(np.around(val[2,2],15))
    return coef * val, coef * singlet


def pendulum_averaging(offdiagonal_component_of_the_greens_function, pendulum, m):
    costet = pendulum[:, 2].reshape(m, m)
    F = offdiagonal_component_of_the_greens_function
    if costet[0, 0] < 0:
        av_coefficient = 1.0
    else:
        av_coefficient = -1.0
    # if costet is from -1 to 1 delete minus in coef or change if condition
    dphi = 2 * np.pi / (m - 1)
    px = pendulum[:, 0].reshape(m, m)
    py = pendulum[:, 1].reshape(m, m)
    pz = pendulum[:, 2].reshape(m, m)
    val = np.zeros((3, 3), dtype='complex')
    Fx = np.ones(m * m, dtype='complex')
    Fy = np.ones(m * m, dtype='complex')
    Fz = np.ones(m * m, dtype='complex')
    # F0=np.ones(m*m,dtype='complex')
    # Commented lines serve to check if numerical integration done correctly, all values should be 1.0
    # norm=AvCoef*sp.integrate.simps(sp.integrate.simps(np.ones((m,m)),dx=dphi),costet[:,0])/(4*np.pi)
    # normx=3*AvCoef*sp.integrate.simps(sp.integrate.simps(px**2,dx=dphi),costet[:,0])/(4*np.pi)
    # normz=3*AvCoef*sp.integrate.simps(sp.integrate.simps(pz**2,dx=dphi),costet[:,0])/(4*np.pi)
    # print(norm,normx,normz)
    coef = av_coefficient * 3.0 / (2 * np.pi)
    for i in np.arange(m * m):
        # IsigmaYconvention
        Fx[i] = 0.5 * (np.diag(F[i])[-1] - np.diag(F[i])[0])
        Fz[i] = 0.5 * (F[i, 0, 1] + F[i, 1, 0])
        Fy[i] = -0.5j * np.sum(np.diag(F[i]))
    # lets reshape everything to 2D!
    Fx = Fx.reshape(m, m)
    Fy = Fy.reshape(m, m)
    Fz = Fz.reshape(m, m)
    # F0=F0.reshape(m,m)
    # singlet=-1.0*sp.integrate.simps(sp.integrate.simps(F0,dx=dphi),costet[:,0])
    # if np.abs(singlet)>1e-9:
    #    print('singlet!',singlet)

    val[0, 0] = sp.integrate.simps(sp.integrate.simps(px * Fx, dx=dphi), costet[:, 0])
    val[0, 1] = sp.integrate.simps(sp.integrate.simps(py * Fx, dx=dphi), costet[:, 0])
    val[0, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fx, dx=dphi), costet[:, 0])

    val[1, 0] = sp.integrate.simps(sp.integrate.simps(px * Fy, dx=dphi), costet[:, 0])
    val[1, 1] = sp.integrate.simps(sp.integrate.simps(py * Fy, dx=dphi), costet[:, 0])
    val[1, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fy, dx=dphi), costet[:, 0])

    val[2, 0] = sp.integrate.simps(sp.integrate.simps(px * Fz, dx=dphi), costet[:, 0])
    val[2, 1] = sp.integrate.simps(sp.integrate.simps(py * Fz, dx=dphi), costet[:, 0])
    val[2, 2] = sp.integrate.simps(sp.integrate.simps(pz * Fz, dx=dphi), costet[:, 0])
    return coef * val


def calculate_magnetic_boundaries(gamma_initial, gammas_homogeneous, omegas_output, scattering_matrix1,
                                  scattering_matrix2, where, propagation_flag):
    profile = np.ones((gammas_homogeneous.shape[0], 2, 2), dtype='complex')
    hom = gammas_homogeneous
    profile[0] = gamma_initial
    U_hom, V_hom, W_hom = omegas_output
    boundary2 = int(3.0 * hom.shape[0] / 4.0 - 1)
    boundary1 = int(hom.shape[0] / 4.0 - 1)
    Mone = np.diag([1.0 + 0.0j] * 2)
    if propagation_flag > 0:
        s1 = scattering_matrix1
        s2 = scattering_matrix2
    else:
        s1 = scattering_matrix2
        s2 = scattering_matrix1
    for ro in np.arange(hom.shape[0]):
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                 .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro != 0 and ro != int(hom.shape[0] - 1):
            profile[ro] = gamma_initial
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro == boundary1:
            gamma_initial = s1.dot(gamma_initial).dot(conj(s1))
        if ro == boundary2:
            gamma_initial = s2.dot(gamma_initial).dot(conj(s2))
        elif where > int(hom.shape[0] / 2):
            if ro != 0 and np.mod(ro + 1, 10) == 0:  # !=0 and ro!=int(hom.shape[0]-1):
                profile[int((ro + 1) / 10)] = gamma_initial
    profile[-1] = gamma_initial
    return gamma_initial, profile


def calculate_magnetic_boundaries_4scatterings(gamma_initial, gamma_homogeneous, omega_output, scattering_matrix1,
                                               scattering_matrix2,  propagation_flag):
    profile = np.ones((int(gamma_homogeneous.shape[0]), 2, 2), dtype='complex')
    hom = gamma_homogeneous
    profile[0] = gamma_initial
    U_hom, V_hom, W_hom = omega_output
    boundary2 = int(3 * hom.shape[0] / 8 - 1)
    boundary1 = int(hom.shape[0] / 8 - 1)
    boundary3 = int(5 * hom.shape[0] / 8 - 1)
    boundary4 = int(7 * hom.shape[0] / 8 - 1)
    Mone = np.diag([1.0 + 0.0j] * 2)
    if propagation_flag > 0:
        s1 = scattering_matrix1
        s2 = scattering_matrix2
    else:
        s1 = scattering_matrix2
        s2 = scattering_matrix1
    for ro in np.arange(hom.shape[0]):
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro != 0 and ro != int(hom.shape[0] - 1):
            profile[ro] = gamma_initial
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro == boundary1 or ro == boundary3:
            # add if for magnetic scattering
            gamma_initial = s1.dot(gamma_initial).dot(conj(s1))
        if ro == boundary2 or ro == boundary4:
            # add if for magnetic scattering
            gamma_initial = s2.dot(gamma_initial).dot(conj(s2))
    profile[-1] = gamma_initial
    return profile


def calculate_magnetic_boundaries_endpoints(gamma_initial, gamma_homogeneous, omegas_output, scattering_matrix1,
                                               scattering_matrix2, where, propagation_flag):
    hom = gamma_homogeneous
    U_hom, V_hom, W_hom = omegas_output
    boundary2 = int(3.0 * hom.shape[0] / 4.0 - 1)
    boundary1 = int(hom.shape[0] / 4.0 - 1)
    Mone = np.diag([1.0 + 0.0j] * 2)
    where_value = np.copy(Mone)
    if propagation_flag > 0:
        s1 = scattering_matrix1
        s2 = scattering_matrix2
    else:
        s1 = scattering_matrix2
        s2 = scattering_matrix1
    for ro in np.arange(hom.shape[0]):
        if ro == where and where < 0.5 * hom.shape[0]:
            where_value[:] = gamma_initial
        delta = gamma_initial - hom[ro]
        inversed_W_hom = inverse2by2(Mone + W_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta))
        gamma_initial = hom[ro] + U_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)].dot(delta).dot(inversed_W_hom)\
                .dot(V_hom[2 * ro:2 * (ro + 1), 2 * ro:2 * (ro + 1)])
        if ro == boundary1:
            gamma_initial = s1.dot(gamma_initial).dot(conj(s1))
        if ro == boundary2:
            gamma_initial = s2.dot(gamma_initial).dot(conj(s2))
        if ro == where and where > 0.5 * hom.shape[0]:
            where_value[:] = gamma_initial
    return gamma_initial, where_value




