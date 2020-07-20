import numpy as np
import matplotlib.pyplot as plt
import os, sys

#equipment setting
theta = 60.0
eps = 0.5
A = 1 #area
alpha = 1.0

#influent landfill gas
Fi = 100.0 #mol/s
yiCO2 = 0.5
yiCH4 = 0.5
Pt = 500.0
PgateCO2 = 20.84786
ymCO2 = PgateCO2 / Pt
ymCH4 = 1 - ymCO2

rho_HKUST = 1127.74393
rho_ELM = 1290.500743

def systemA():
    NadsCO2 = 2.661462922098774 * rho_HKUST # mol/m3
    NdesCO2 = 1.1618529072248476 * rho_HKUST # mol/m3
    NadsCH4 = 0.6467749161906667 * rho_HKUST # mol/m3
    R = (NadsCO2 - NdesCO2) / NadsCO2
    S = (NadsCO2/NadsCH4) / (yiCO2/yiCH4)

    L = theta / (A * (1 - eps)) * yiCO2 * Fi / NadsCO2
    V = L * A
    ZadsCO2 = V * (1 - eps) * NadsCO2 # mol
    ZadsCH4 = V * (1 - eps) * NadsCH4 # mol
    ZinCO2 = theta * Fi * yiCO2
    ZinCH4 = theta * Fi * yiCH4
    ZpurgeCH4 = alpha * V * (1 - eps) * NdesCO2 # mol
    ZproductCH4 = ZinCH4 - ZpurgeCH4 - ZadsCH4
    FproductCH4 = ZproductCH4 / theta
    Firequired = Fi / FproductCH4 * 100
    L *= Firequired / Fi
    print("####### system A #########")
    print("selectivity:", S)
    print("Regenerability:", R)
    print("ZinCO2:", ZinCO2)
    print("ZadsCO2:", ZadsCO2)
    print("ZinCH4:", ZinCH4)
    print("ZadsCH4:", ZadsCH4)
    print("ZpurgeCH4:", ZpurgeCH4)
    print("FinCO2:", ZinCO2 / theta / FproductCH4 * 100)
    print("FadsCO2:", ZadsCO2 / theta / FproductCH4 * 100)
    print("FinCH4:", ZinCH4 / theta / FproductCH4 * 100)
    print("FadsCH4:", ZadsCH4 / theta / FproductCH4 * 100)
    print("FpurgeCH4:", ZpurgeCH4 / theta / FproductCH4 * 100)
    print("ZproductCH4:", ZproductCH4)
    print("FproductCH4:", FproductCH4)
    print("***")
    print("Column length:", L)
    print("Fi required to obtain 100F CH4:", Firequired)
    print()
    return L, Firequired

def systemB_caseI():
    NadsCO2B1 = 3.133773 * rho_ELM # mol/m3
    NadsCH4B1 = 0.078479 * rho_ELM # mol/m3
    NdesCO2B1 = 0.0 # mol/m3
    R_B1 = (NadsCO2B1 - NdesCO2B1) / NadsCO2B1
    S_B1 = (NadsCO2B1/NadsCH4B1) / (yiCO2/yiCH4)

    Fm = (NadsCO2B1 * yiCH4 - NadsCH4B1 * yiCO2) / (NadsCO2B1 * ymCH4 - NadsCH4B1 * ymCO2) * Fi
    L_B1 = theta / (A * (1 - eps)) * (yiCO2 * Fi - ymCO2 * Fm) / NadsCO2B1

    V = L_B1 * A
    ZadsCO2B1 = V * (1 - eps) * NadsCO2B1 # mol
    ZadsCH4B1 = V * (1 - eps) * NadsCH4B1 # mol
    ZinCO2 = theta * Fi * yiCO2
    ZinCH4 = theta * Fi * yiCH4
    ZpurgeCH4B1 = alpha * V * (1 - eps) * NdesCO2B1 # mol
    print("####### system B case I #########")
    print(" *** column B1 *** ")
    print("selectivity:", S_B1)
    print("Regenerability:", R_B1)
    print("ZinCO2:", ZinCO2)
    print("ZadsCO2:", ZadsCO2B1)
    print("ZslippedCO2:", Fm * ymCO2 * theta)
    print("ZinCH4:", ZinCH4)
    print("ZadsCH4:", ZadsCH4B1)
    print("ZpurgeCH4:", ZpurgeCH4B1)

    NadsCO2B2 = 0.531504531142 * rho_HKUST # mol/m3
    NadsCH4B2 = 2.28378733015 * rho_HKUST # mol/m3
    NdesCO2B2 = 1.16185290722 * rho_HKUST # mol/m3
    R_B2 = (NadsCO2B2 - NdesCO2B2) / NadsCO2B2
    S_B2 = (NadsCO2B2/NadsCH4B2) / (yiCO2/yiCH4)

    L_B2 = theta / (A * (1-eps)) * ymCO2 * Fm / NadsCO2B2

    V = L_B2 * A
    ZadsCO2B2 = V * (1 - eps) * NadsCO2B2 # mol
    ZadsCH4B2 = V * (1 - eps) * NadsCH4B2 # mol
    ZmidCO2 = theta * Fm * ymCO2
    ZmidCH4 = theta * Fm * ymCH4
    ZpurgeCH4B2 = alpha * V * (1 - eps) * NdesCO2B2 # mol
    print(" *** column B2 *** ")
    print("selectivity:", S_B2)
    print("Regenerability:", R_B2)
    print("ZmidCO2:", ZmidCO2)
    print("ZadsCO2:", ZadsCO2B2)
    print("ZmidCH4:", ZmidCH4)
    print("ZadsCH4:", ZadsCH4B2)
    print("ZpurgeCH4:", ZpurgeCH4B2)

    ZproductCH4 = ZinCH4 - ZpurgeCH4B1 - ZpurgeCH4B2 - ZadsCH4B1 - ZadsCH4B2
    FproductCH4 = ZproductCH4 / theta
    Firequired = Fi / FproductCH4 * 100
    L_B1 *= Firequired / Fi
    L_B2 *= Firequired / Fi

    print("FinCO2:", ZinCO2 / theta / FproductCH4)
    print("FinCH4:", ZinCH4 / theta / FproductCH4)
    print("FadsCO2 1:", ZadsCO2B1 / theta / FproductCH4)   
    print("FadsCH4 1:", ZadsCH4B1 / theta / FproductCH4)   
    print("FmidCO2:", ZmidCO2 / theta / FproductCH4)
    print("FmidCH4:", ZmidCH4 / theta / FproductCH4)    
    print("FadsCO2 2:", ZadsCO2B2 / theta / FproductCH4)   
    print("FadsCH4 2:", ZadsCH4B2 / theta / FproductCH4)
    print("FpurgeCH4:", (ZpurgeCH4B1 + ZpurgeCH4B2) / theta / FproductCH4)
    print("ZproductCH4:", ZproductCH4)
    print("FproductCH4:", FproductCH4)
    print("***")
    print("Column length B1:", L_B1)
    print("Column length B2:", L_B2)
    print("L_B2 / L_B1:", L_B2 / L_B1)
    print("Fi required to obtain 100F CH4:", Firequired)
    print()
    return L_B1, L_B2, Firequired

def systemB_caseII():
    NadsCO2B1 = 3.133773 * rho_ELM # mol/m3
    NadsCH4B1 = 0.078479 * rho_ELM # mol/m3
    NdesCO2B1 = 0.0 # mol/m3
    R_B1 = (NadsCO2B1 - NdesCO2B1) / NadsCO2B1
    S_B1 = (NadsCO2B1/NadsCH4B1) / (yiCO2/yiCH4)

    # full spec
    NadsCO2B2 = 2.661462922098774 * rho_HKUST # mol/m3
    NdesCO2B2 = 1.1618529072248476 * rho_HKUST # mol/m3
    NadsCH4B2 = 0.6467749161906667 * rho_HKUST # mol/m3
    # for slipping off
    NadsCO2B2slip = 0.531504531142 * rho_HKUST # mol/m3
    R_B2 = (NadsCO2B2 - NdesCO2B2) / NadsCO2B2
    S_B2 = (NadsCO2B2/NadsCH4B2) / (yiCO2/yiCH4)

    Fm = (NadsCO2B1 * yiCH4 - NadsCH4B1 * yiCO2) / (NadsCO2B1 * ymCH4 - NadsCH4B1 * ymCO2) * Fi
    MTZmoveL = Fm * ymCO2 / (NadsCO2B2slip * A * (1 - eps))
    MTZmoveH = (Fi * yiCO2 - Fm * ymCO2) / ((NadsCO2B2 - NadsCO2B2slip) * A * (1-eps))
    B1_saturated_time = (MTZmoveH - MTZmoveL) / MTZmoveH * theta
    L_B1 = B1_saturated_time / (A * (1 - eps)) * (yiCO2 * Fi - ymCO2 * Fm) / NadsCO2B1

    V = L_B1 * A
    ZadsCO2B1 = V * (1 - eps) * NadsCO2B1 # mol
    ZadsCH4B1 = V * (1 - eps) * NadsCH4B1 # mol
    ZinCO2 = theta * Fi * yiCO2
    ZinCH4 = theta * Fi * yiCH4
    ZpurgeCH4B1 = alpha * V * (1 - eps) * NdesCO2B1 # mol
    print("####### system B case II #########")
    print(" *** column B1 *** ")
    print("selectivity:", S_B1)
    print("Regenerability:", R_B1)
    print("ZinCO2:", ZinCO2)
    print("ZadsCO2:", ZadsCO2B1)
    print("ZslippedCO2:", Fm * ymCO2 * theta)
    print("ZinCH4:", ZinCH4)
    print("ZadsCH4:", ZadsCH4B1)
    print("ZpurgeCH4:", ZpurgeCH4B1)

    L_B2 = theta / (A * (1-eps)) * ymCO2 * Fm / NadsCO2B2slip

    V = L_B2 * A
    ZadsCO2B2 = V * (1 - eps) * NadsCO2B2 # mol
    ZadsCH4B2 = V * (1 - eps) * NadsCH4B2 # mol
    ZmidCO2 = B1_saturated_time * Fm * ymCO2 + (theta - B1_saturated_time) * Fi * yiCO2
    ZmidCH4 = B1_saturated_time * Fm * ymCH4 + (theta - B1_saturated_time) * Fi * yiCH4
    ZpurgeCH4B2 = alpha * V * (1 - eps) * NdesCO2B2 # mol
    print(" *** column B2 *** ")
    print("selectivity:", S_B2)
    print("Regenerability:", R_B2)
    print("ZmidCO2:", ZmidCO2)
    print("ZadsCO2:", ZadsCO2B2)
    print("ZmidCH4:", ZmidCH4)
    print("ZadsCH4:", ZadsCH4B2)
    print("ZpurgeCH4:", ZpurgeCH4B2)

    ZproductCH4 = ZinCH4 - ZpurgeCH4B1 - ZpurgeCH4B2 - ZadsCH4B1 - ZadsCH4B2
    FproductCH4 = ZproductCH4 / theta
    Firequired = Fi / FproductCH4 * 100
    L_B1 *= Firequired / Fi
    L_B2 *= Firequired / Fi

    print("FinCO2:", ZinCO2 / theta / FproductCH4)
    print("FinCH4:", ZinCH4 / theta / FproductCH4)
    print("FadsCO2 1:", ZadsCO2B1 / theta / FproductCH4)   
    print("FadsCH4 1:", ZadsCH4B1 / theta / FproductCH4)   
    print("FmidCO2:", ZmidCO2 / theta / FproductCH4)
    print("FmidCH4:", ZmidCH4 / theta / FproductCH4)    
    print("FadsCO2 2:", ZadsCO2B2 / theta / FproductCH4)   
    print("FadsCH4 2:", ZadsCH4B2 / theta / FproductCH4)
    print("FpurgeCH4:", (ZpurgeCH4B1 + ZpurgeCH4B2) / theta / FproductCH4)
    
    print("ZproductCH4:", ZproductCH4)
    print("FproductCH4:", FproductCH4)
    print("***")
    print("Column length B1:", L_B1)
    print("Column length B2:", L_B2)
    print("L_B2 / L_B1:", L_B2 / L_B1)
    print("Fi required to obtain 100F CH4:", Firequired)
    print()
    return L_B1, L_B2, Firequired


if __name__ == "__main__":
    L_A, Fi_A = systemA()
    L_B1I, L_B2I, Fi_BI = systemB_caseI()
    L_B1II, L_B2II, Fi_BII = systemB_caseII()
    print(" *** relative to system A *** ")
    print(" case I ")
    print("L_B1:", L_B1I / L_A)
    print("L_B2:", L_B2I / L_A)
    print("total:", (L_B1I + L_B2I)/ L_A)
    print("down percentage of length:", 100 * (1 - (L_B1I + L_B2I)/ L_A))
    print("down percentage of feed:", 100 * (1 - Fi_BI / Fi_A))
    print()
    print(" case II ")
    print("L_B1:", L_B1II / L_A)
    print("L_B2:", L_B2II / L_A)
    print("total:", (L_B1II + L_B2II)/ L_A)
    print("down percentage of length:", 100 * (1 - (L_B1II + L_B2II)/ L_A))
    print("down percentage of feed:", 100 * (1 - Fi_BII / Fi_A))
