#Micro. Meso. Mat. 152, 246-252 (2012)

import sys
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import pickle

dT = 1
Tcenter = 263.0

#param for CH4
ln_beta0m = -6.39
beta1m = 1931
b0m = 0.24
b1m = -59.3
c0m = -0.068
c1m = 17.2
ita0m = -4.02
ita1m = 4635

#param for CO2
ln_beta0c = -7.48
beta1c = 2774
b0c = 0.45
b1c = -157.5
c0c = -0.069
c1c = 21.8
ita0c = -8.29
ita1c = 7643

#heat cap
ha0 = 0
hb0 = -3.105e-10
hc0 = 2.427e-7
hd0 = -7.2796e-5
he0 = 1.239e-2
hf0 = -4.26e-1
ha1 = -1.018616e-10
hb1 = 1.731346e-7
hc1 = -1.172852e-4
hd1 = 3.95955e-2
he1 = -6.663567
hf1 = 4.480867e2


N01 = 0
N02 = 0

R = 8.314 * 0.001 #kJ/mol-CO2

def calc_param_CH4(T):
    iT = 1.0 / T
    beta = ln_beta0m + beta1m * iT
    beta = np.exp(beta)
    b = b0m + b1m * iT
    c = c0m + c1m * iT
    Nmax = ita0m + ita1m * iT
    return beta, b, c, Nmax

def calc_param_CO2(T):
    iT = 1.0 / T
    beta = ln_beta0c + beta1c * iT
    beta = np.exp(beta)
    b = b0c + b1c * iT
    c = c0c + c1c * iT
    Nmax = ita0c + ita1c * iT
    return beta, b, c, Nmax

def comp_P_CH4(N):
    global betam, bm, cm, Nmaxm
    P = Nmaxm * N / (betam * (Nmaxm - N)) * np.exp(bm * N + cm * N * N )
    return P

def dPdN_CH4(N):
    global betam, bm, cm, Nmaxm
    dPdN = Nmaxm / betam * np.exp(bm * N + cm * N * N ) / (Nmaxm - N) / (Nmaxm - N) * (Nmaxm + N * (Nmaxm - N ) * (bm + 2 * cm * N))
    return dPdN
    
def dPdN_CO2(N):
    global betac, bc, cc, Nmaxc
    dPdN = Nmaxc / betac * np.exp(bc * N + cc * N * N ) / (Nmaxc - N) / (Nmaxc - N) * (Nmaxc + N * (Nmaxc - N ) * (bc + 2 * cc * N))
    return dPdN
    

def comp_P_CO2(N):
    global betac, bc, cc, Nmaxc
    P = Nmaxc * N / (betac * (Nmaxc - N)) * np.exp(bc * N + cc * N * N )
    return P

def sum_f_CO2(N):
    p = comp_P_CO2(N)
    dpdn = dPdN_CO2(N)
    return N * dpdn / p

def sum_f_CH4(N):
    p = comp_P_CH4(N)
    dpdn = dPdN_CH4(N)
    return N * dpdn / p

def f_CO2(N):
    global p0_1
    a = p0_1 - comp_P_CO2(N)
    return a*a

def f_CH4(N):
    global p0_2
    a = p0_2 - comp_P_CH4(N)
    return a*a

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def f(x):
    global ptotal, p0_1, p0_2, N01, N02
    #x = sigmoid(x/100.0)
    x = x/100.0
    p0_1 = frac_y1 * ptotal / x
    p0_2 = frac_y2 * ptotal / (1.0 - x)
    N0_1 = optimize.fmin_bfgs(f_CO2,N01,gtol=1e-12,disp=False)
    N0_2 = optimize.fmin_bfgs(f_CH4,0,gtol=1e-12,disp=False)
    N01 = N0_1[0]
    N02 = N0_2[0]
    ansm, err = integrate.quad( sum_f_CO2, 0, N0_1)
    ansc, err = integrate.quad( sum_f_CH4, 0, N0_2)
    #ansm = p0_1 * N0_1 - ansm
    #ansc = p0_2 * N0_2 - ansc
    #print x, p0_1, p0_2, N0_1, N0_2, ansm, ansc
    return (ansm - ansc)*(ansm - ansc)

def print_f(x,T):
    global ptotal, p0_1, p0_2, N01, N02
    global betam, bm, cm, Nmaxm
    global betac, bc, cc, Nmaxc
    #x = sigmoid(x/100.0)
    x = x / 100.0
    p0_1 = frac_y1 * ptotal / x
    p0_2 = frac_y2 * ptotal / (1.0 - x)
    x0_1 = x
    x0_2 = 1.0 - x0_1
    N0_1 = optimize.fmin_bfgs(f_CO2,N01,gtol=1e-12,disp=False)
    N0_2 = optimize.fmin_bfgs(f_CH4,0,gtol=1e-12,disp=False)
    N01 = N0_1[0]
    N02 = N0_2[0]
    Nt = N0_1 * N0_2 / ( N0_2 * x0_1 + N0_1 * x0_2 )
    S = p0_2 / p0_1

    Nc = Nt[0] * x0_1[0]
    Nm = Nt[0] * x0_2[0]

    Hc = - beta1c + b1c * Nc + c1c * Nc * Nc + ita1c / Nmaxc - ita1c / (Nmaxc - Nc)
    Hc *= -R
    Hm = - beta1m + b1m * Nm + c1m * Nm * Nm + ita1m / Nmaxm - ita1m / (Nmaxm - Nm)
    Hm *= -R

    return Nc, Nm, Hc, Hm, S[0]

def singleCO2(p,Tall):
    global betac, bc, cc, Nmaxc,p0_1
    p0_1 = p
    Ncx = np.empty(0)
    Qcx = np.empty(0)
    Nc = 0
    for T in Tall[::-1]:
        betac, bc, cc, Nmaxc = calc_param_CO2(T)
        Nc = optimize.fmin_bfgs(f_CO2,Nc,gtol=1e-9,disp=False)
        Qc = - beta1c + b1c * Nc + c1c * Nc * Nc + ita1c / Nmaxc - ita1c / (Nmaxc - Nc)
        Qc *= -R
        Ncx = np.append(Ncx,Nc)
        Qcx = np.append(Qcx,Qc)

    return Ncx, Qcx

def singleCH4(p,Tall):
    global betam, bm, cm, Nmaxm, p0_2
    p0_2 = p
    Nmx = np.empty(0)
    Qmx = np.empty(0)
    Nm = 0
    for T in Tall[::-1]:
        betam, bm, cm, Nmaxm = calc_param_CH4(T)
        Nm = optimize.fmin_bfgs(f_CH4,Nm,gtol=1e-9,disp=False)
        Qm = - beta1m + b1m * Nm + c1m * Nm * Nm + ita1m / Nmaxm - ita1m / (Nmaxm - Nm)
        Qm *= -R
        Nmx = np.append(Nmx,Nm)
        Qmx = np.append(Qmx,Qm)
    return Nmx, Qmx

def calc_dNdT(dT,N):
    #top backward diff
    #end forward diff
    #other center diff

    dNdT = np.zeros_like(N)
    dNdT[1:-1] = (N[2:] - N[:-2]) / (2.0 * dT)
    dNdT[0] = (N[1] - N[0]) / dT
    dNdT[-1] = (N[-1] - N[-2]) / dT

    return dNdT

def specific_heat(T):
    T2 = T * T
    T4 = T2 * T2 
    if T < 300:
        return ha0 * T4 * T + hb0 * T4 + hc0 * T2 * T + hd0 * T2 + he0 * T + hf0
    if T > 400:
        return 1
    else:
        return ha1 * T4 * T + hb1 * T4 + hc1 * T2 * T + hd1 * T2 + he1 * T + hf1

def centerize(x,*a):
    Cp = a[0]
    HI = a[1]
    HIV = a[2]
    Tall = a[3]
    dT = a[4]

    shiftCp = Cp + x
    Ti, Hi = crosspoint(shiftCp,HI,Tall,dT)
    Tiv, Hiv = crosspoint(shiftCp,HIV,Tall,dT)
    ans = Ti + Tiv - 2*Tcenter
    return ans * ans

def crosspoint(A,B,Tall,dT):
    idx = np.where(A>B)[0][0]
    slopeA = (A[idx] - A[idx-1]) / dT
    slopeB = (B[idx] - B[idx-1]) / dT
    T = (B[idx-1]-A[idx-1])/(slopeA-slopeB) + Tall[idx-1]
    H = slopeA*(T-Tall[idx-1]) + A[idx-1]
    return T, H

def interpol(A,T,Tall,dT):
    idx = np.where(Tall>T)[0][0]
    slopeA = (A[idx] - A[idx-1]) / dT
    return slopeA * (T - Tall[idx-1]) + A[idx-1]

if __name__ == "__main__":
    argv = sys.argv
    Tall = np.arange(200,800,dT)

    try:
        with open('H.pickle','rb') as f:
            NcI,NcII,NcIII,NcIV,NmI,NmIV,SI,HI,HII,HIII,HIV,intCp = pickle.load(f)
    except:
        #step I: mixture 5bar (CO2 2.5bar CH4 2.5bar)
        ptotal = 5.0
        frac_y1 = 0.5
        frac_y2 = 1.0 - frac_y1
        x = 80.0
        NcI = np.empty(0)
        NmI = np.empty(0)
        QcI = np.empty(0)
        QmI = np.empty(0)
        SI = np.empty(0)
    	
        for T in Tall[::-1]:
            betam, bm, cm, Nmaxm = calc_param_CH4(T)
            betac, bc, cc, Nmaxc = calc_param_CO2(T)
            ix = x
            x = optimize.fmin_bfgs(f,ix,gtol=1e-12,disp=False)
            Nc, Nm, Qc, Qm, S = print_f(x,T)
            NcI = np.append(NcI,Nc)
            NmI = np.append(NmI,Nm)
            QcI = np.append(QcI,Qc)
            QmI = np.append(QmI,Qm)
            SI = np.append(SI,S)
    
        #reverse
        NcI = NcI[::-1]
        NmI = NmI[::-1]
        QcI = QcI[::-1]
        QmI = QmI[::-1]
        SI = SI[::-1]
       
    
        #step II: rinse with CO2 2.5bar (CO2 2.5bar CH4 0bar)
        NcII, QcII = singleCO2(2.5, Tall)
        NcII = NcII[::-1]
        QcII = QcII[::-1]
        
    
        #step III: depress to 0.05bar (CO2 5kPa, CH4 0kPa)
        NcIII, QcIII = singleCO2(0.15, Tall)
        NcIII = NcIII[::-1]
        QcIII = QcIII[::-1]
    
        #step IV: purge with CH4 0.05bar (CO2 0kPa, CH4 5kPa)
        NmIV, QmIV = singleCH4(0.15, Tall)
        NcIV = np.zeros_like(NmIV)
        NmIV = NmIV[::-1]
        QmIV = QmIV[::-1]
    
        #dN/dT
        dNdT_cI = calc_dNdT(dT,NcI)
        dNdT_cII = calc_dNdT(dT,NcII)
        dNdT_cIII = calc_dNdT(dT,NcIII)
        dNdT_mI = calc_dNdT(dT,NmI)
        dNdT_mIV = calc_dNdT(dT,NmIV)
    
        dHdT_cI = QcI * dNdT_cI
        dHdT_cII = QcII * dNdT_cII
        dHdT_cIII = QcIII * dNdT_cIII
        dHdT_mI = QmI * dNdT_mI
        dHdT_mIV = QmIV * dNdT_mIV
    
        dHdT_I = dHdT_cI + dHdT_mI
    
        HI = integrate.cumtrapz(dHdT_I[::-1],Tall[::-1],initial=0)
        HI = HI[::-1] + QcI[-1]*NcI[-1] + QmI[-1]*NmI[-1]
    
        HII = integrate.cumtrapz(dHdT_cII[::-1],Tall[::-1],initial=0)
        HII = HII[::-1] + QcII[-1]*NcII[-1]
    
        HIII = integrate.cumtrapz(dHdT_cIII[::-1],Tall[::-1],initial=0)
        HIII = HIII[::-1] + QcIII[-1]*NcIII[-1]
    
        HIV = integrate.cumtrapz(dHdT_mIV[::-1],Tall[::-1],initial=0)
        HIV = HIV[::-1] + QmIV[-1]*NmIV[-1]

        Cp = [specific_heat(T) for T in Tall]
        intCp = integrate.cumtrapz(Cp,Tall,initial=0)

        with open('H.pickle','wb') as f:
            h = [NcI,NcII,NcIII,NcIV,NmI,NmIV,SI,HI,HII,HIII,HIV, intCp]
            pickle.dump(h,f)

    #step 4 to step 1
    hiv = interpol(HIV,Tcenter,Tall,dT)
    cp298 = interpol(intCp,Tcenter,Tall,dT)

    shift = hiv - cp298

    ti,hi = crosspoint(intCp+shift,HI,Tall,dT)
    tii,hii = crosspoint(intCp+shift,HII,Tall,dT)
    tiii, hiii = crosspoint(intCp+shift,HIII,Tall,dT)
    tiv, hiv = crosspoint(intCp+shift,HIV,Tall,dT)

    nci  = interpol(NcI,ti,Tall,dT)
    nmi  = interpol(NmI,ti,Tall,dT)
    ncii = interpol(NcII,tii,Tall,dT)
    nciii= interpol(NcIII,tiii,Tall,dT)
    nmiv = interpol(NmIV,tiv,Tall,dT)

    print("step T[K] H[J/g] Nco2[mmol/g] Nch4[mmol/g]")
    print("stepI",ti, hi,nci,nmi)
    print("stepII",tii, hii,ncii,0)
    print("stepIII",tiii, hiii, nciii, 0)
    print("stepIV",tiv, hiv, 0, nmiv)
    print()
    print("effective amount adsorbed of CO2 [mmol/g]:", nci)
    print("working capacity [mmol/g]:", nci - nciii)
    print("selectivity [-]:", nci/nmi)
    print("regenerability [%]:", (nci - nciii) / nci * 100)
    
    # for i in [HI,HII,HIII,HIV,intCp+shift]:
    #     plt.plot(Tall,i,'-')
    # plt.show()


    # plt.plot(Tall,NcI,'-',label="NcI")
    # plt.plot(Tall,NcII,'-',label="NcII")
    # plt.plot(Tall,NcIII,'-',label="NcIII")
    # plt.plot(Tall,NcIV,'-',label="NcIV")
    # plt.legend()
    # plt.show()
    
    print()
    print("T[K] NcI[mmol/g] NmI[mmol/g] NtI[mmol/g] NcII[mmol/g] NcIII[mmol/g] NcIV[mmol/g] NmIV[mmol/g] HI[J/g] HII[J/g] HIII[J/g] HIV[J/g] intCp[J/g]")
    for t,qci,qmi,qcii,qciii,qciv,qmiv,hi,hii,hiii,hiv,cp in zip(Tall,NcI,NmI,NcII,NcIII,NcIV,NmIV,HI,HII,HIII,HIV,intCp+shift):
        if t < 401:
            print(t,qci,qmi,qci+qmi,qcii,qciii,qciv,qmiv,hi,hii,hiii,hiv,cp)
