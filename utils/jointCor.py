import numpy as np
def jointCor(RT):
    alpha6 = 0.15
    RTc6 = 350
    alpha5 = 0.2
    RTc5 = 200
    alpha4 = 0.3
    RTc4 = 120
    alpha3 = 0.4
    RTc3 = 60
    alpha2 = 0.5
    RTc2 = 30
    r61 = 1 / (1 + np.exp(-alpha6 * (RT - RTc6)))
    r62 = 1 - r61
    r51 = 1 / (1 + np.exp(-alpha5 * (RT - RTc5)))
    r52 = 1 - r51
    r41 = 1 / (1 + np.exp(-alpha4 * (RT - RTc4)))
    r42 = 1 - r41
    r31 = 1 / (1 + np.exp(-alpha3 * (RT - RTc3)))
    r32 = 1 - r31
    r21 = 1 / (1 + np.exp(-alpha2 * (RT - RTc2)))
    r22 = 1 - r21

    R6 = r61
    R5 = r62 * r51
    R4 = r62 * r52 * r41
    R3 = r62 * r52 * r42 * r31
    R2 = r62 * r52 * r42 * r32 * r21
    R1 = r62 * r52 * r42 * r32 * r22
    return [R1, R2, R3, R4, R5, R6]