# -*- coding: utf-8 -*-
"""
    File Name : QCcalc.py

    Purpose : Puttting functions that calculate Green's functions via Ricatti amplitudes in one place. 

    Special note: Stepchoice (defines your grid and slices qp trajectory over it) and OrderParameter (slices OP according to grid)
    should be specified within the program as they are crutial imput for functions below, but too special for each given problem.

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

def Inverse2by2(M1):
    val=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)    
    return val

def HomogeneousGammaObtain(Ene,delta,deltatilda):
    e=Ene
    val=np.empty((delta.shape[2],2,2),dtype='complex')
    for k in np.arange(delta[0,0,:].size):
        M11=np.r_[np.c_[-e*np.identity(2),np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex')],[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')]]
        M22=np.r_[[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')],np.c_[np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex'),e*np.identity(2)]]
        M12=np.r_[np.c_[np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex'),deltatilda[:,:,k]],[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')]]
        M21=np.r_[[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')],np.c_[delta[:,:,k],np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex')]]
        M=M11+M12+M21+M22
        if np.any(np.isnan(M)):
            print(M)   
        U=UI=np.identity(4,dtype='complex')
        Maximum=1
        dec=16
        precision=5*1e-14
        cnt=0
        while Maximum>precision:
            i,j=np.unravel_index(np.argmax(np.abs(sp.linalg.tril(M,-1))),(4,4))
            m=np.array([[M[j,j],M[i,j]],[M[j,i],M[i,i]]],dtype='complex')
            u=np.diag([1.0+0.0j]*4)
            ui=np.diag([1.0+0.0j]*4)
            epsi=0.5*(m[1,1]-m[0,0])
            if cnt>1000: 
                print('Im stuck!',Maximum)
                print(i,j)
                print(M)
                M=np.around(M,dec)
            if np.absolute(m[0,1]*m[1,0])<precision**2:
                if np.absolute(m[0,1])<precision:
                    u[i,i]=ui[i,i]=0.0+0.0j
                    u[i,j]=ui[i,j]=1.0+0.0j
                    u[j,i]=ui[j,i]=1.0+0.0j
                    u[j,j]=ui[j,j]=0.0+0.0j
                else:
                    u=np.identity(4,dtype='complex')
                    ui=np.identity(4,dtype='complex')
            else:
                sq=np.around(epsi+1j*np.sqrt(-1.0*m[0,1]*m[1,0]-epsi*epsi),dec)
                if np.abs(sq)==0:
                    print(m)
                    print('sq= ',sq)
                else:
                    t1=-1.0*np.around(m[1,0]/sq,dec)
                    t2=-1.0*np.around(m[0,1]/sq,dec)
                    coef=np.around(1.0/np.sqrt(1.0+t1*t2),dec)
                    u[i,i]=ui[i,i]=u[j,j]=ui[j,j]=coef
                    u[j,i]=np.around(t1*coef,dec)
                    ui[j,i]=-1.0*np.around(t1*coef,dec)
                    u[i,j]=-1.0*np.around(t2*coef,dec)
                    ui[i,j]=np.around(t2*coef,dec)          
            cnt=cnt+1
            M=np.dot(u,np.dot(M,ui))
            Maximum=np.amax(np.abs(sp.linalg.tril(M,-1)))
            U=np.dot(u,U)
            UI=np.dot(UI,ui)
        A=np.array([[U[2,2],0,U[2,3],0],[0,U[2,2],0,U[2,3]],[U[3,2],0,U[3,3],0],[0,U[3,2],0,U[3,3]]],dtype='complex')
        if np.any(np.isnan(A)):
            print(A)
            print(M)            
        b=-U[2:,:2].reshape(4)
        gam=np.linalg.solve(A,b)
        val[k]=gam.reshape(2,2)
    return val
def HomGreen(Ene,delta,deltatilda):
    e=Ene
    val=np.empty((delta.shape[2],2,2),dtype='complex')
    #valt=np.empty((delta.shape[2],2,2),dtype='complex')
    Mone=np.identity(2,dtype='complex')
    for k in np.arange(delta[0,0,:].size):
        M11=np.r_[np.c_[e*Mone,np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex')],[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')]]
        M22=np.r_[[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')],np.c_[np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex'),-e*Mone]]
        M12=np.r_[np.c_[np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex'),-delta[:,:,k]],[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')]]
        M21=np.r_[[np.zeros(4,dtype='complex')],[np.zeros(4,dtype='complex')],np.c_[-deltatilda[:,:,k],np.zeros(2,dtype='complex'),np.zeros(2,dtype='complex')]]
        M=M11+M12+M21+M22
        A=sp.linalg.eig(M)
        GH=-1j*np.pi*A[1].dot(sp.diag(np.sign(A[0].imag))).dot(sp.linalg.inv(A[1]))
        val[k]=sp.linalg.inv(GH[:2,:2]-1j*np.pi*Mone).dot(GH[:2,2:])
        #val[k]=sp.linalg.inv(GH[:2,:2]-1j*np.pi*Mone,check_finite=False).dot(GH[:2,2:])
        #valt[k]=conj(-sp.linalg.inv(1j*np.pi*np.identity(2)+GH[2:,2:]).dot(GH[2:,:2]))
        #conjugate for backwards trajectory
    return val#,valt
def Omegas(Ene,intersections,gammahom,deltatilda):
    gh=sp.linalg.block_diag(*gammahom)
    #times=1j*np.diag(intersections)
    times=0.5*1j*np.diag(intersections)
    #0.5 for double usage in calculate routines
    DT=sp.linalg.block_diag(*deltatilda.T.reshape(gammahom.shape[0],2,2))
    energy=Ene*np.diag(np.ones(intersections.size))
    solution=np.empty((gammahom.shape[0],2,2),dtype='complex')
    Om1=energy-gh.dot(DT)
    Exp1=times.dot(Om1)
    Om2=energy-DT.dot(gh)
    Exp2=times.dot(Om2)
    mu1=0.5*(sp.diag(Exp1)[::2]+sp.diag(Exp1)[1::2])
    mu2=0.5*(sp.diag(Exp2)[::2]+sp.diag(Exp2)[1::2])
    Mone=np.diag([1.0+0.0j]*2)
    for k in np.arange(deltatilda.shape[2]):
        b=DT[2*k:2*k+2,2*k:2*k+2].reshape(4)
        A=sp.linalg.block_diag(Om1[2*k:2*k+2,2*k:2*k+2].T,Om1[2*k:2*k+2,2*k:2*k+2].T)+np.c_[np.array([Om2[2*k,2*k],0,Om2[2*k,2*k+1],0]),np.array([0,Om2[2*k,2*k],0,Om2[2*k,2*k+1]])\
        ,np.array([Om2[2*k+1,2*k],0,Om2[2*k+1,2*k+1],0]),np.array([0,Om2[2*k+1,2*k],0,Om2[2*k+1,2*k+1]])]
        solution[k]=np.linalg.solve(A,b).reshape(2,2)
        O1=Exp1[2*k:2*k+2,2*k:2*k+2]-mu1[k]*Mone
        O2=Exp2[2*k:2*k+2,2*k:2*k+2]-mu2[k]*Mone
        q1=np.sqrt(O1[0,0]*O1[1,1]-O1[1,0]*O1[0,1])
        q2=np.sqrt(O2[0,0]*O2[1,1]-O2[1,0]*O2[0,1])
        if np.isnan(float(np.abs(np.cos(q1)))):
            print(O1,O2)
            print(mu1,mu2)
            print('cos, sin',np.cos(q1),np.sin(q1))
        if np.abs(q1)==0.0:
            Exp1[2*k:2*k+2,2*k:2*k+2]=np.exp(mu1[k])*(Mone+O1)
        else:
            Exp1[2*k:2*k+2,2*k:2*k+2]=np.exp(mu1[k])*(np.cos(q1)*Mone+np.sin(q1)*O1/q1)
        if np.abs(q2)==0.0:
            Exp2[2*k:2*k+2,2*k:2*k+2]=np.exp(mu2[k])*(Mone+O2)
        else:
            Exp2[2*k:2*k+2,2*k:2*k+2]=np.exp(mu2[k])*(np.cos(q2)*Mone+np.sin(q2)*O2/q2)
        #solution[k]=sp.linalg.solve_sylvester(Om2[2*k:2*k+2,2*k:2*k+2],Om1[2*k:2*k+2,2*k:2*k+2],DT[2*k:2*k+2,2*k:2*k+2])
    w=sp.linalg.block_diag(*solution) 
    Uh=sp.linalg.block_diag(Exp1)
    Vh=sp.linalg.block_diag(Exp2)
    Wh=Vh.dot(w).dot(Uh)-w  
    val=Uh,Vh,Wh
    return val
def OmegasR(Ene,intersections,gammahom,deltatilda):
    gh=sp.linalg.block_diag(*gammahom)
    times=1j*np.diag(intersections)
    #times=0.5*1j*np.diag(intersections)
    #0.5 for double usage in calculate routines
    DT=sp.linalg.block_diag(*deltatilda.T.reshape(gammahom.shape[0],2,2))
    energy=Ene*np.diag(np.ones(intersections.size))
    solution=np.empty((gammahom.shape[0],2,2),dtype='complex')
    Om1=energy-gh.dot(DT)
    Exp1=times.dot(Om1)
    Om2=energy-DT.dot(gh)
    Exp2=times.dot(Om2)
    mu1=0.5*(sp.diag(Exp1)[::2]+sp.diag(Exp1)[1::2])
    mu2=0.5*(sp.diag(Exp2)[::2]+sp.diag(Exp2)[1::2])
    Mone=np.diag([1.0+0.0j]*2)
    for k in np.arange(deltatilda.shape[2]):
        b=DT[2*k:2*k+2,2*k:2*k+2].reshape(4)
        A=sp.linalg.block_diag(Om1[2*k:2*k+2,2*k:2*k+2].T,Om1[2*k:2*k+2,2*k:2*k+2].T)+np.c_[np.array([Om2[2*k,2*k],0,Om2[2*k,2*k+1],0]),np.array([0,Om2[2*k,2*k],0,Om2[2*k,2*k+1]])\
        ,np.array([Om2[2*k+1,2*k],0,Om2[2*k+1,2*k+1],0]),np.array([0,Om2[2*k+1,2*k],0,Om2[2*k+1,2*k+1]])]
        solution[k]=np.linalg.solve(A,b).reshape(2,2)
        O1=Exp1[2*k:2*k+2,2*k:2*k+2]-mu1[k]*Mone
        O2=Exp2[2*k:2*k+2,2*k:2*k+2]-mu2[k]*Mone
        q1=np.sqrt(O1[0,0]*O1[1,1]-O1[1,0]*O1[0,1])
        q2=np.sqrt(O2[0,0]*O2[1,1]-O2[1,0]*O2[0,1])
        if np.isnan(float(np.abs(np.cos(q1)))):
            print(O1,O2)
            print(mu1,mu2)
            print('cos, sin',np.cos(q1),np.sin(q1))
        if np.abs(q1)==0.0:
            Exp1[2*k:2*k+2,2*k:2*k+2]=np.exp(mu1[k])*(Mone+O1)
        else:
            Exp1[2*k:2*k+2,2*k:2*k+2]=np.exp(mu1[k])*(np.cos(q1)*Mone+np.sin(q1)*O1/q1)
        if np.abs(q2)==0.0:
            Exp2[2*k:2*k+2,2*k:2*k+2]=np.exp(mu2[k])*(Mone+O2)
        else:
            Exp2[2*k:2*k+2,2*k:2*k+2]=np.exp(mu2[k])*(np.cos(q2)*Mone+np.sin(q2)*O2/q2)
        #solution[k]=sp.linalg.solve_sylvester(Om2[2*k:2*k+2,2*k:2*k+2],Om1[2*k:2*k+2,2*k:2*k+2],DT[2*k:2*k+2,2*k:2*k+2])
    w=sp.linalg.block_diag(*solution) 
    Uh=sp.linalg.block_diag(Exp1)
    Vh=sp.linalg.block_diag(Exp2)
    Wh=Vh.dot(w).dot(Uh)-w  
    val=Uh,Vh,Wh
    return val
def calculate(gamstart,gamhom,Omegas,S):
    gamini=gamstart
    #profile=np.zeros(gamhom.shape,dtype='complex')
    hom=gamhom
    Uh,Vh,Wh  =  Omegas
    b1=int(hom.shape[0]/2)
    b2=hom.shape[0]
    print(b1,b2)
    Mone=np.diag([1.0+0.0j]*2)
    if 2*b1-b2!=0.0: print(b1,b2)
    for ro in np.arange(hom.shape[0]):
        delta=gamini-hom[ro]
        IWh=Inverse2by2(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta))
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro==b1-1 or ro==b2:
            gamini=S.dot(gamini).dot(conj(S))
        #profile[i]=gamini
    return gamini
def calcDoS(g,gt):
    val=np.ones(g[:,0,0].size)
    #gt=conj(gt)
    Mone=np.diag([1.0+0.0j]*2)
    for i in np.arange(g[:,0,0].size):
        ggt=g[i].dot(conj(gt[i])) 
        if np.linalg.cond(Mone-ggt) < 1.0/np.finfo(ggt.dtype).eps:
            InN=Inverse2by2(Mone-ggt)
            #inv=sp.linalg.inv(np.identity(2,dtype='complex')-ggt,check_finite=False)
            val[i]=0.5*np.sum(np.diag(1j*InN.dot(Mone+ggt)).imag)
        else:
            val[i]=-1
        if val[i]<0: flag=True# print ("DoS<0!!! not again! ",i);
        #if val[i]>1e3: val[i]=0
        #elif val[i]>10: val[i]=10
    return val
def calcDoSspec(g,gt,sigma,p):
    val=np.ones((2,g[:,0,0].size))
    G=np.ones((2,2),dtype='complex')
    Mone=np.diag([1.0+0.0j]*2)
    for i in np.arange(g[:,0,0].size):
        ggt=g[i].reshape(2,2).dot(conj(gt[i].reshape((2,2))))
        G=1j*Inverse2by2(Mone-ggt).dot(Mone+ggt)
        #G=1j*sp.linalg.inv(Mone-ggt,check_finite=False).dot(Mone+ggt)
        #Dos=0.5*np.sum(np.diag(G).imag)
        if sigma=='sx':
            #val[:,i]=Dos+0.5*np.sum(np.diag(G.dot([[0.0,1.0],[1.0,0.0]]))).real,Dos-0.5*np.sum(np.diag(G.dot([[0.0,1.0],[1.0,0.0]]))).real
            val[:,i]=0.25*(np.sum(np.diag(G))+G[0,1]+G[1,0]).imag,0.25*(np.sum(np.diag(G))-G[0,1]-G[1,0]).imag   
        elif sigma=='sy':
            #val[:,i]=Dos+0.5*np.sum(np.diag(G.dot([[0.0,1.0j],[-1.0j,0.0]]))).real,Dos-0.5*np.sum(np.diag(G.dot([[0.0,1.0j],[-1.0j,0.0]]))).real
            val[:,i]=0.25*(np.sum(np.diag(G))+1j*G[0,1]-1j*G[1,0]).imag,0.25*(np.sum(np.diag(G))-1j*G[0,1]+1j*G[1,0]).imag
        elif sigma=='sz':
            val[:,i]=0.5*np.diag(G).imag
            #val[:,i]=0.5*np.diag(G).imag
        else: print('wrong string')
        #if val[0,i]>15: val[0,i]=15;
        #if val[1,i]>15: val[1,i]=15;          
    return val[:,p]
def GapEq(g,gt,p,m):
    costet=p[:,2].reshape(m,m)
    if costet[0,0]<0:
        AvCoef=1.0
    else:
        AvCoef=-1.0
    #if costet is from -1 to 1 delete minus in coef or put othervise
    dphi=2*np.pi/(m-1)
    px=p[:,0].reshape(m,m)
    py=p[:,1].reshape(m,m)
    pz=p[:,2].reshape(m,m)
    val=np.zeros((3,3),dtype='complex')
    Fx=np.ones(g[:,0,0].shape,dtype='complex')
    Fy=np.ones(g[:,0,0].shape,dtype='complex')
    Fz=np.ones(g[:,0,0].shape,dtype='complex') 
    F0=np.ones(g[:,0,0].shape,dtype='complex')
    #Commented lines serve to check if numerical integration done correctly, all values should be 1.0
    #norm=AvCoef*sp.integrate.simps(sp.integrate.simps(np.ones((m,m)),dx=dphi),costet[:,0])/(4*np.pi)
    #normx=3*AvCoef*sp.integrate.simps(sp.integrate.simps(px**2,dx=dphi),costet[:,0])/(4*np.pi)
    #normz=3*AvCoef*sp.integrate.simps(sp.integrate.simps(pz**2,dx=dphi),costet[:,0])/(4*np.pi)
    #print(norm,normx,normz)
    coef=AvCoef*3.0/(2*np.pi)
    Mone=np.diag([1.0+0.0j]*2)
    for i in np.arange(g[:,0,0].shape[0]):
        ggt=g[i].reshape(2,2).dot(conj(gt[i].reshape((2,2))))  
        F=np.pi*-2j*Inverse2by2(Mone-ggt).dot(g[i].reshape(2,2))
        #F=np.pi*-2j*sp.linalg.inv(Mone-ggt,check_finite=False).dot(g[i].reshape(2,2))
        #IsigmaYconvention        
        Fx[i]=0.5*(np.diag(F)[-1]-np.diag(F)[0])
        Fz[i]=0.5*(F[0,1]+F[1,0])
        Fy[i]=-0.5j*np.sum(np.diag(F))
        F0[i]=0.5*(F[0,1]-F[1,0])
    #lets reshape everything to 2D!      
    Fx=Fx.reshape(m,m)
    Fy=Fy.reshape(m,m)
    Fz=Fz.reshape(m,m)
    F0=F0.reshape(m,m)
    singlet=-1.0*sp.integrate.simps(sp.integrate.simps(F0,dx=dphi),costet[:,0])
    #if np.abs(singlet)>1e-9: 
    #    print('singlet!',singlet)

    val[0,0]=sp.integrate.simps(sp.integrate.simps(px*Fx,dx=dphi),costet[:,0])
    val[0,1]=sp.integrate.simps(sp.integrate.simps(py*Fx,dx=dphi),costet[:,0])
    val[0,2]=sp.integrate.simps(sp.integrate.simps(pz*Fx,dx=dphi),costet[:,0])

    val[1,0]=sp.integrate.simps(sp.integrate.simps(px*Fy,dx=dphi),costet[:,0])
    val[1,1]=sp.integrate.simps(sp.integrate.simps(py*Fy,dx=dphi),costet[:,0])
    val[1,2]=sp.integrate.simps(sp.integrate.simps(pz*Fy,dx=dphi),costet[:,0])

    val[2,0]=sp.integrate.simps(sp.integrate.simps(px*Fz,dx=dphi),costet[:,0])
    val[2,1]=sp.integrate.simps(sp.integrate.simps(py*Fz,dx=dphi),costet[:,0])
    val[2,2]=sp.integrate.simps(sp.integrate.simps(pz*Fz,dx=dphi),costet[:,0])
    #print(np.around(val[2,2],15))
    return coef*val,coef*singlet
def PAV(F,p,m):
    costet=p[:,2].reshape(m,m)
    if costet[0,0]<0:
        AvCoef=1.0
    else:
        AvCoef=-1.0
    #if costet is from -1 to 1 delete minus in coef or put othervise
    dphi=2*np.pi/(m-1)
    px=p[:,0].reshape(m,m)
    py=p[:,1].reshape(m,m)
    pz=p[:,2].reshape(m,m)
    val=np.zeros((3,3),dtype='complex')
    Fx=np.ones(m*m,dtype='complex')
    Fy=np.ones(m*m,dtype='complex')
    Fz=np.ones(m*m,dtype='complex') 
    #F0=np.ones(m*m,dtype='complex')
    #Commented lines serve to check if numerical integration done correctly, all values should be 1.0
    #norm=AvCoef*sp.integrate.simps(sp.integrate.simps(np.ones((m,m)),dx=dphi),costet[:,0])/(4*np.pi)
    #normx=3*AvCoef*sp.integrate.simps(sp.integrate.simps(px**2,dx=dphi),costet[:,0])/(4*np.pi)
    #normz=3*AvCoef*sp.integrate.simps(sp.integrate.simps(pz**2,dx=dphi),costet[:,0])/(4*np.pi)
    #print(norm,normx,normz)
    coef=AvCoef*3.0/(2*np.pi)
    for i in np.arange(m*m):
        #IsigmaYconvention        
        Fx[i]=0.5*(np.diag(F[i])[-1]-np.diag(F[i])[0])
        Fz[i]=0.5*(F[i,0,1]+F[i,1,0])
        Fy[i]=-0.5j*np.sum(np.diag(F[i]))
    #lets reshape everything to 2D!      
    Fx=Fx.reshape(m,m)
    Fy=Fy.reshape(m,m)
    Fz=Fz.reshape(m,m)
    #F0=F0.reshape(m,m)
    #singlet=-1.0*sp.integrate.simps(sp.integrate.simps(F0,dx=dphi),costet[:,0])
    #if np.abs(singlet)>1e-9: 
    #    print('singlet!',singlet)

    val[0,0]=sp.integrate.simps(sp.integrate.simps(px*Fx,dx=dphi),costet[:,0])
    val[0,1]=sp.integrate.simps(sp.integrate.simps(py*Fx,dx=dphi),costet[:,0])
    val[0,2]=sp.integrate.simps(sp.integrate.simps(pz*Fx,dx=dphi),costet[:,0])

    val[1,0]=sp.integrate.simps(sp.integrate.simps(px*Fy,dx=dphi),costet[:,0])
    val[1,1]=sp.integrate.simps(sp.integrate.simps(py*Fy,dx=dphi),costet[:,0])
    val[1,2]=sp.integrate.simps(sp.integrate.simps(pz*Fy,dx=dphi),costet[:,0])

    val[2,0]=sp.integrate.simps(sp.integrate.simps(px*Fz,dx=dphi),costet[:,0])
    val[2,1]=sp.integrate.simps(sp.integrate.simps(py*Fz,dx=dphi),costet[:,0])
    val[2,2]=sp.integrate.simps(sp.integrate.simps(pz*Fz,dx=dphi),costet[:,0])
    #print(np.around(val[2,2],15))
    return coef*val
def calculateM(gamstart,gamhom,Omegas,S1,S2,where,fl):
    gamini=gamstart
    profile=np.ones((gamhom.shape[0],2,2),dtype='complex')
    hom=gamhom
    profile[0]=gamstart
    Uh,Vh,Wh  =  Omegas
    b2=int(3.0*hom.shape[0]/4.0-1)
    b1=int(hom.shape[0]/4.0-1)
    Mone=np.diag([1.0+0.0j]*2)
    if fl>0:
        s1=S1
        s2=S2
    else:
        s1=S2
        s2=S1
    for ro in np.arange(hom.shape[0]):
        delta=gamini-hom[ro]        
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro!=0 and ro!=int(hom.shape[0]-1):
            profile[ro]=gamini
        delta=gamini-hom[ro]
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro==b1:# or ro==b2:
            #add if for magnetic scattering
            gamini=s1.dot(gamini).dot(conj(s1))
        if ro==b2:# or ro==b2:
            #add if for magnetic scattering
            gamini=s2.dot(gamini).dot(conj(s2))
        elif where>int(gamhom.shape[0]/2):
             if ro!=0 and np.mod(ro+1,10)==0:#!=0 and ro!=int(hom.shape[0]-1):
                profile[int((ro+1)/10)]=gamini       
        delta=gamini-hom[ro]
    profile[-1]=gamini
    return gamini,profile
def calculateM4S(gamstart,gamhom,Omegas,S1,S2,where,fl):
    gamini=gamstart
    profile=np.ones((int(gamhom.shape[0]),2,2),dtype='complex')
    hom=gamhom
    profile[0]=gamstart
    Uh,Vh,Wh  =  Omegas
    b2=int(3*hom.shape[0]/8-1)
    b1=int(hom.shape[0]/8-1)
    b3=int(5*hom.shape[0]/8-1)
    b4=int(7*hom.shape[0]/8-1)
    Mone=np.diag([1.0+0.0j]*2)
    if fl>0:
        s1=S1
        s2=S2
    else:
        s1=S2
        s2=S1
    for ro in np.arange(hom.shape[0]):
        delta=gamini-hom[ro]        
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro!=0 and ro!=int(hom.shape[0]-1):
            profile[ro]=gamini
        delta=gamini-hom[ro]
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro==b1 or ro==b3:
            #add if for magnetic scattering
            gamini=s1.dot(gamini).dot(conj(s1))
        if ro==b2 or ro==b4:
            #add if for magnetic scattering
            gamini=s2.dot(gamini).dot(conj(s2))
    profile[-1]=gamini
    return profile
def calculateM2X(gamstart,gamhom,Omegas,S1,S2,where,fl):
    gamini=gamstart
    hom=gamhom
    Uh,Vh,Wh  =  Omegas
    b2=int(3.0*hom.shape[0]/4.0-1)
    b1=int(hom.shape[0]/4.0-1)
    Mone=np.diag([1.0+0.0j]*2)
    whval=np.copy(Mone)
    if fl>0:
        s1=S1
        s2=S2
    else:
        s1=S2
        s2=S1
    for ro in np.arange(hom.shape[0]):
        if ro==where and where<0.5*hom.shape[0]:
            whval[:]=gamini
        delta=gamini-hom[ro]        
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro==b1:# or ro==b2:
            #add if for magnetic scattering
            gamini=s1.dot(gamini).dot(conj(s1))
        if ro==b2:# or ro==b2:
            #add if for magnetic scattering
            gamini=s2.dot(gamini).dot(conj(s2))
        if ro==where and where>0.5*hom.shape[0]:
            whval[:]=gamini
    return gamini,whval
def calculateMaT(gamstart,gamhom,Omegas,S1,S2,fl):
    gamini=gamstart
    profile=np.ones((int(gamhom.shape[0]),2,2),dtype='complex')
    hom=gamhom
    profile[0]=gamstart
    Uh,Vh,Wh  =  Omegas
    b2=int(3.0*hom.shape[0]/4.0-1)
    b1=int(hom.shape[0]/4.0-1)
    Mone=np.diag([1.0+0.0j]*2)
    if fl>0:
        s1=S1
        s2=S2
    else:
        s1=S2
        s2=S1
    for ro in np.arange(hom.shape[0]):
        delta=gamini-hom[ro]        
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro!=0 and ro!=int(hom.shape[0]-1):
            profile[ro]=gamini
        delta=gamini-hom[ro]
        #IWh=sp.linalg.inv(Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta),check_finite=False)
        M1=Mone+Wh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta)
        IWh=(1/(M1[0,0]*M1[1,1]-M1[1,0]*M1[0,1])*np.c_[M1[1,1],-M1[0,1],-M1[1,0],M1[0,0]]).reshape(2,2)
        gamini=hom[ro]+Uh[2*ro:2*(ro+1),2*ro:2*(ro+1)].dot(delta).dot(IWh).dot(Vh[2*ro:2*(ro+1),2*ro:2*(ro+1)])
        if ro==b1:# or ro==b2:
            #add if for magnetic scattering
            gamini=s1.dot(gamini).dot(conj(s1))
        if ro==b2:# or ro==b2:
            #add if for magnetic scattering
            gamini=s2.dot(gamini).dot(conj(s2))      
    profile[-1]=gamini
    return profile
    """tilda=False
    if tilda:
        for k in np.arange(deltatilda.shape[2]):
            Om1[k]=-e*np.identity(2,dtype='complex')-np.dot(gh[k],deltatilda[:,:k])
            Om2[k]=-e*np.identity(2,dtype='complex')-np.dot(deltatilda[:,:,k],gh[k])
            w[k]=sp.linalg.solve_sylvester(Om2[k],Om1[k],deltatilda[:,:,k])
    else:
        for k in np.arange(deltatilda.shape[2]):
            Om1[k]=e*np.identity(2,dtype='complex')-np.dot(gh[k],deltatilda[:,:,k])
            Om2[k]=e*np.identity(2,dtype='complex')-np.dot(deltatilda[:,:,k],gh[k])
            w[k]=sp.linalg.solve_sylvester(Om2[k],Om1[k],deltatilda[:,:,k])
    Om1=sp.linalg.block_diag(*Om1)
    Om2=sp.linalg.block_diag(*Om2)
    w=sp.linalg.block_diag(*w)"""
