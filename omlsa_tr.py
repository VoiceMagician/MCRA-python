""" 20240711
MCRA+OMLSA的去突发噪声版本流式处理, 更加适配C/C++迁移对比数据

参考论文: 
[1] Hirszhorn, Ariel , et al. "Transient Interference Suppression in Speech Signals Based on the OM-LSA Algorithm." VDE (2012).
[2] Cohen, Israel , and B. Berdugo . "Speech enhancement for non-stationary noise environments." Signal Processing 81.11(2001):2403-2418.
"""
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf
#from utils import *
from scipy import interpolate


############### Initialize the data ################
f_win_length = 1
win_freq = np.array([0.25, 0.5, 0.25])
alpha_eta = 0.92
alpha_s = 0.9
alpha_d = 0.85
beta = 2
eta_min = 0.0158
GH0 = np.power(eta_min,0.5)
gamma0 = 4.6
gamma1 = 3
zeta0 = 1.67
Bmin = 1.66
Vwin = 15
Nwin = 8
loop_i = 0
lambda_d = 0
eta_2term = 0
S = 0
St = 0
lambda_dav = 0
Smin = 0
Smint_sw = 0
Smint = 0
Smin_sw = 0
SW = 0
G = 0
conv_Y = 0

l_mod_lswitch = 0

fs = 16000
frame_length = 256
frame_move = 128
N_eff = int(frame_length / 2 + 1)

# 瞬态噪声
transient_frame_len = 64
transient_frame_hop = 32
transient_N_eff = int(transient_frame_len / 2) + 1

def check_init():
    '''窗初始化和dct初始化 RNNoise'''
    half_window = np.zeros(frame_move)
    #dct_table = np.zeros(int(NB_BANDS * NB_BANDS))
    for frame_i in range(frame_move):
        half_window[frame_i] = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (frame_i + 0.5) / frame_move)**2)
    
    #for nb_i in range(NB_BANDS):
    #    for nb_j in range(NB_BANDS):
    #        dct_table[nb_i*NB_BANDS + nb_j] = np.cos((nb_i + 0.5) * nb_j * np.pi / NB_BANDS)
    #        if (0 == nb_j):
    #            dct_table[nb_i*NB_BANDS + nb_j] *= np.sqrt(0.5)
    
    return half_window #, dct_table

def hann_window(fft_size, hop_size):
    block_num = fft_size // hop_size
    ana_win = np.zeros(fft_size)
    syn_win = np.zeros(fft_size)
    norm_win = np.zeros(fft_size)

    ana_win = np.hanning(fft_size)
    norm_win = ana_win * ana_win

    #for i in range(fft_size):
    #    ana_win[i] = (0.54 - 0.46*np.cos((2*i)*np.pi/(fft_size-1)))
    
    #for i in range(fft_size):
    #    norm_win[i] = ana_win[i] * ana_win[i]

    for i in range(hop_size):
        temp = 0
        for j in range(int(block_num)):
            temp += norm_win[i + j * hop_size]
        norm_win[i] = 1 / temp
    
    for i in range(hop_size):
        # 因为j=0的时候只是自身替换，没有意义，所以跳过这个处理过程
        for j in range(1, int(block_num)):
            norm_win[i + j * hop_size] = norm_win[i]

    for i in range(fft_size):
        syn_win[i] = norm_win[i] * ana_win[i]

    return ana_win, syn_win

# Return real part of exponential integral, same as matlab expint()
def expint(v):
    return np.real(-scipy.special.expi(-v)-np.pi*1j)

def omlsa_streamer(cplx_data, plot=None):
    global loop_i,frame_buffer,frame_out,frame_in,frame_result,y_out_time,l_mod_lswitch,lambda_d,eta_2term,S,St,lambda_dav,Smin,Smin_sw,Smint_sw,Smint,zi,G,conv_Y
    if loop_i < 1:
        loop_i = loop_i + 1 # 0->1
        return cplx_data # 不处理
    else:
        Ya2 = np.power(abs(cplx_data), 2) # 功率谱
        Sf = np.convolve(win_freq.flatten(), Ya2.flatten()) # 频率平滑
        #print(Sf.shape)
        Sf = Sf[f_win_length:(f_win_length+len(cplx_data))] # 长度裁剪

        if (loop_i==1):
            lambda_dav = lambda_d = Ya2 # 初始化
            gamma = 1
            GH1 = 1
            eta_2term = np.power(GH1, 2) * gamma # eta_2term=1, G^2_{H_1}*\gamma
            S = Smin = St = Smint = Smin_sw = Smint_sw = Sf
        
        gamma = np.divide(Ya2, np.maximum(lambda_d, 1e-10)) # posteriori SNR # eq.3

        # eta_2term = G_H^2 * \gamma
        eta = alpha_eta * eta_2term + (1-alpha_eta) * np.maximum((gamma-1), 0) # eq.32 # est priori SNR

        eta = np.maximum(eta, eta_min)
        v = np.divide(gamma * eta, (1 + eta)) # 这个怎么来的？

        GH1 = np.divide(eta, (1+eta)) * np.exp(0.5 * expint(v)) # eq.33 # eta->Xi

        S = alpha_s * S + (1-alpha_s) * Sf # eq.15

        if (loop_i < 30):
            Smin = S
            Smin_sw = S
        else:
            Smin = np.minimum(Smin, S)
            Smin_sw = np.minimum(Smin_sw, S)

        gamma_min = np.divide((Ya2 / Bmin), Smin) # eq.18
        zeta = np.divide(S / Bmin, Smin) # eq.18

        I_f = np.zeros((N_eff, ))
        I_f[gamma_min < gamma0] = 1
        I_f[zeta < zeta0] = 1
        #I_f = np.where((gamma_min < gamma0) and (zeta < zeta0), 1, I_f) # eq.21

        conv_I = np.convolve(win_freq, I_f)

        conv_I = conv_I[f_win_length:N_eff+f_win_length] # 平滑

        Sft = St

        conv_Y = np.convolve(win_freq.flatten(), (I_f*Ya2).flatten())

        conv_Y = conv_Y[f_win_length:N_eff+f_win_length]

        Sft = St
        Sft = np.divide(conv_Y, conv_I+1e-7) # eq.26
        #Sft[(conv_I) == 0] = St[(conv_I) == 0] # 频率里面的块进行重新组合
        Sft = np.where((conv_I) == 0, St, Sft)

        St = alpha_s * St + (1 - alpha_s) * Sft # eq.27

        if (loop_i < 30):
            Smint = St
            Smint_sw = St
        else:
            Smint = np.minimum(Smint, St)
            Smint_sw = np.minimum(Smint_sw, St)
        
        gamma_mint = np.divide(Ya2 / Bmin, Smint) # eq.28
        zetat = np.divide(S / Bmin, Smint) # eq.28

        temp = [0] * N_eff

        qhat = (gamma1 - gamma_mint) / (gamma1 - 1) # eq.29
        qhat[gamma_mint < 1] = 1
        qhat[gamma_mint < gamma1] = 1
        qhat[zetat < zeta0] = 1
        qhat[gamma_mint >= gamma1] = 0
        qhat[zetat >= zeta0] = 0

        phat = np.divide(1, (1 + np.divide(qhat, (1 - qhat) + 1e-7) * (1 + eta) * np.exp(-v)) + 1e-7) # eq.7
        phat[gamma_mint >= gamma1] = 1
        phat[zetat >= zeta0] = 1

        alpha_dt = alpha_d + (1-alpha_d) * phat # eq.11
        lambda_dav = alpha_dt * lambda_dav + (1 - alpha_dt) * Ya2 # eq.10
        lambda_d = lambda_dav * beta # eq.12

        if l_mod_lswitch == 2 * Vwin:
            l_mod_lswitch = 0
            try:
                SW = np.concatenate((SW[1:Nwin], Smin_sw))
                Smin = np.amin(SW)
                Smin_sw = S
                SWt = np.concatenate((SWt[1:Nwin], Smint_sw))
                Smint = np.amin(SWt)
                Smint_sw = St
            except:
                SW = np.tile(S, (Nwin))
                SWt = np.tile(St, (Nwin))
            
        l_mod_lswitch = l_mod_lswitch + 1

        gamma = np.divide(Ya2, np.maximum(lambda_d, 1e-10))

        eta = alpha_eta * eta_2term + (1 - alpha_eta) * np.maximum(gamma - 1, 0)

        eta[eta < eta_min] = eta_min

        v = np.divide(gamma * eta, (1 + eta))

        GH1 = np.divide(eta, (1 + eta)) * np.exp(0.5 * expint(v))

        G = np.divide(GH1, phat) * np.power(GH0, (1 - phat))

        eta_2term = np.power(GH1, 2) * gamma

        X = np.concatenate((np.zeros((3,)), (G[3:(N_eff-1)]) * (cplx_data[3:(N_eff-1)]), [0]))

        loop_i = loop_i + 1

    return X

'''mcra data of 
'''
fcwL=60
min_buff = np.zeros([fcwL, N_eff])
amp = np.zeros(N_eff)
amp_min = np.zeros(N_eff)
amp_tmp = np.zeros(N_eff)
noise_est = np.zeros(N_eff)
init_p = np.zeros(N_eff)
amp_sparm = 0.8
init_p_sparm = 0.2
noise_est_parm = 0.95
omega = 1
b = np.hanning(int(2 * omega + 1)) # [0.25, 0.5, 0.25]
def mcra_streamer(cplx_data):
    '''mcra 噪声估计
    '''
    global loop_i, amp, amp_min, amp_tmp, noise_est, init_p

    amp_pr = np.sqrt((cplx_data.real ** 2) + (cplx_data.imag ** 2))

    L = 1
    if (loop_i == 0):
        amp = np.copy(amp_pr)
        amp_min = np.copy(amp_pr)
        amp_tmp = np.copy(amp_pr)
        noise_est = np.copy(amp_pr)
        init_p = init_p * 0
        pass
    
    # python
    amp_sm = np.convolve(amp_pr, b, 'full')[omega:-omega]
    #amp_min = np.min(np.array([amp_tmp, amp]), axis=0)
    #amp_tmp = amp
    # github
    #amp_sm = np.convolve(win_freq.flatten(), amp_pr.flatten()) # 频率平滑
    #amp_sm = amp_sm[f_win_length:(f_win_length+len(amp_pr))] # 长度裁剪

    amp = amp_sparm * amp + (1-amp_sparm) * amp_sm # S(k,l) \alpha_s
    #if (loop_i < fcwL):
    #    min_buff[loop_i, :] = amp
    #else:
    #    min_buff[:(fcwL-1)] = min_buff[1:] # 清除上一帧老数据
    #    min_buff[-1] = amp # 放置新数据
    #amp_min = np.min(min_buff, axis=0)
    # 该方法是原版论文方法
    if ((0 == loop_i % fcwL) and (loop_i != 0)):
        amp_min = np.min([amp_tmp, amp], axis=0)
        amp_tmp = amp
    else:
        amp_min = np.min([amp_min, amp], axis=0)
        amp_tmp = np.min([amp_tmp, amp], axis=0)


    for k in range(N_eff):
        #amp[k] = amp_sparm * amp[k] + (1-amp_sparm) * amp_pr[k]
        #amp_min[k] = np.min([amp_min[k], amp[k]])
        #amp_tmp[k] = np.min([amp_tmp[k], amp[k]])

        if (amp[k] > (3*amp_min[k])): # -3dB
            p = 1 # I(k,l)
        else:
            p = 0
        init_p[k] = (init_p_sparm * init_p[k] + (1-init_p_sparm) * p) # 影响p更新速度 # p'
        noise_est[k] = ((noise_est_parm * noise_est[k] + (1-noise_est_parm) \
                   * amp_pr[k]) * (1 - init_p[k]) + noise_est[k] * init_p[k])
    
    # 需要想个方法降低低频的语音噪声
    #tmp = noise_est[:10]
    #noise_est[:10] = np.where(tmp > 1, tmp * 0.5, tmp)
    #noise_est = np.where(noise_est > 0.2, 0.2, noise_est) # 可以通过限制最大值来达到不强压强噪

    return noise_est, amp_pr

int_value = [123380,87676,71942,62610,56275,51622,48025,45140,42763,40762,39050,37565,36261,35106,34074,33145,32304,31538,30837,30193,29599,29049,28538,28062,27617,27200,26809,26442,26095,25768,25458,25165,24887,24622,24371,24131,23903,23685,23477,23277,23086,22903,22728,22559,22397,22242,22092,21948,21809,21675,21546,21421,21301,21185,21072,20964,20858,20757,20658,20563,20470,20380,20293,20209,20127,20047,19970,19894,19821,19750,19681,19614,19548,19485,19423,19362,19304,19246,19191,19136,19083,19031,18981,18932,18883,18837,18791,18746,18702,18660,18618,18577,18538,18499,18461,18424,18387,18352,18317,18283,18250,18217,18185,18154,18124,18094,18064,18036,18008,17980,17953,17927,17901,17876,17851,17827,17803,17779,17756,17734,17712,17690,17669,17648,17628,17608,17588,17569,17550,17532,17513,17495,17478,17461,17444,17427,17411,17395,17379,17364,17349,17334,17319,17305,17291,17277,17263,17250,17237,17224,17211,17199,17186,17174,17162,17151,17139,17128,17117,17106,17095,17085,17074,17064,17054,17044,17035,17025,17016,17007,16998,16989,16980,16971,16963,16954,16946,16938,16930,16922,16915,16907,16900,16892,16885,16878,16871,16864,16857,16850,16844,16837,16831,16825,16819,16812,16806,16801,16795,16789,16783,16778,16772,16767,16762,16756,16751,16746,16741,16736,16731,16727,16722,16717,16713,16708,16704,16700,16695,16691,16687,16683,16679,16675,16671,16667,16663,16659,16656,16652,16648,16645,16641,16638,16634,16631,16628,16625,16621,16618,16615,16612,16609,16606,16603,16600,16597,16594,16592,16589,16586,16584,16581,16578,16576,16573,16571,16568,16566,16563,16561,16559,16557,16554,16552,16550,16548,16546,16543,16541,16539,16537,16535,16533,16531,16530,16528,16526,16524,16522,16520,16519,16517,16515,16513,16512,16510,16509,16507,16505,16504,16502,16501,16499,16498,16496,16495,16494,16492,16491,16489,16488,16487,16485,16484,16483,16482,16480,16479,16478,16477,16476,16474,16473,16472,16471,16470,16469,16468,16467,16466,16465,16464,16463,16462,16461,16460,16459,16458,16457,16456,16455,16454,16453,16452,16452,16451,16450,16449,16448,16447,16447,16446,16445,16444,16444,16443,16442,16441,16441,16440,16439,16439,16438,16437,16437,16436,16435,16435,16434,16433,16433,16432,16432,16431,16430,16430,16429,16429,16428,16428,16427,16427,16426,16426,16425,16425,16424,16424,16423,16423,16422,16422,16421,16421,16420,16420,16419,16419,16418,16418,16418,16417,16417,16416,16416,16416,16415,16415,16414,16414,16414,16413,16413,16413,16412,16412,16412,16411,16411,16411,16410,16410,16410,16409,16409,16409,16408,16408,16408,16408,16407,16407,16407,16406,16406,16406,16406,16405,16405,16405,16405,16404,16404,16404,16404,16403,16403,16403,16403,16402,16402,16402,16402,16402,16401,16401,16401,16401,16400,16400,16400,16400,16400,16400,16399,16399,16399,16399,16399,16398,16398,16398,16398,16398,16398,16397,16397,16397,16397,16397,16397,16396,16396,16396,16396,16396,16396,16396,16395,16395,16395,16395,16395,16395,16395,16394,16394,16394,16394,16394,16394,16394,16394,16393,16393,16393,16393,16393,16393,16393,16393,16393,16392,16392,16392,16392,16392,16392,16392,16392,16392,16392,16391,16391,16391,16391,16391,16391,16391,16391,16391,16391,16391,16391,16390,16390,16390,16390,16390,16390,16390,16390,16390,16390,16390,16390,16390,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16389,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16388,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16387,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16386,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16385,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384]

def expintpow_solution(v_subscript):
    v_subscript = v_subscript * 100
    vec = int(v_subscript)
    if (vec < 1):
        vec = 1
    elif (vec > 1500):
        vec = 1500
    g = int_value[vec-1]
    return g

pr_snr = np.zeros(N_eff) # priori SNR of previous frame
P_MIN = 0.0001
pr_snr_parm = 0.95
post_snr_tmp = np.zeros(N_eff) # 上一帧后验信噪比
def get_gh1(noise_est, amp_pr, gh1):
    global pr_snr, post_snr_tmp
    post_snr = (amp_pr / (noise_est+1e-7)) ** 2 # 这个SNR没有转化到dB级别
    post_snr = np.where(post_snr > 10000, 10000, post_snr)

    post_temp = np.where(post_snr-1 < 0, 0, post_snr-1)

    pr_snr = np.power(gh1, 2) * post_snr_tmp # est_priori
    post_snr_tmp = post_snr
    # estimate priori SNR
    current_snr = pr_snr_parm * pr_snr + (1-pr_snr_parm) * post_temp # pr_snr_parm=\alpha # current_snr=\xi
    current_snr = np.where(current_snr < P_MIN, P_MIN, current_snr)
    current_snr = np.where(current_snr > 10000, 10000, current_snr)

    v_int = (current_snr * post_snr) / (1 + current_snr)
    v_int = np.where(v_int > 10000, 10000, v_int)
    v_int = np.where(v_int < 0.01, 0.01, v_int)
    integra = np.zeros_like(v_int)
    for i in range(len(v_int)):
        integra[i] = expintpow_solution(v_int[i]) # 查表获取积分
    m_int = integra / 16384 # Q15->float

    gh1 = current_snr * m_int / (1 + current_snr)
    gh1 = np.where(gh1 > 8, 8, gh1)

    return gh1, current_snr, v_int

hann_win_f_len = 9
hann_win_f_hal = 4
omega_local = 1
omega_global = 15
h_local = [0, 1, 0]#np.hanning(int(2*omega_local)+1) / (int(2*omega_local)+1) #np.zeros(int(2*omega_local)+1)
h_global = np.hanning(int(2*omega_global)+1) #/ (int(2*omega_global)+1) #np.zeros(int(2*omega_global)+1)
#for i in range(len(h_local)):
#    h_local[i] = 0.5 - 0.5 * np.cos(2 * np.pi * i / (len(h_local) + 1))
#for i in range(len(h_global)):
#    h_global[i] = 0.5 - 0.5 * np.cos(2 * np.pi * i / (len(h_global) + 1))
print("h_global[]")
for i in range(len(h_global)):
    print(h_global[i], ',', sep='')
SNR_MIN = 0.1 # 10lg(0.1)=-10dB
SNR_MAX = 0.4 # 10lg(0.4)=-4dB
SNR_P_MIN = 1
SNR_P_MAX = 2
beta = 0.7
q_max = 0.95
def mcra_omlsa_speech_absent_est(old_snr, cur_snr):
    '''语音不存在概率
    '''
    old_snr_frame = 0
    snr_frame = 0
    p_frame = 0
    hann_win_f = np.zeros(hann_win_f_len)
    p_local = np.zeros(N_eff) + 0.5
    p_global = np.zeros(N_eff) + 0.5
    #for i in range(1, (len(hann_win_f)+1)):
    #    hann_win_f[i-1] = 0.5 - 0.5 * np.cos(2 * np.pi * i / (hann_win_f_len + 1))

    snr = beta * old_snr + (1-beta) * cur_snr
    snr_lframe = np.mean(old_snr)
    snr_frame = np.mean(snr)
    snr_global = np.zeros_like(snr)
    snr_local = np.zeros_like(snr)

    # python
    snr_global = np.convolve(snr, h_global, mode='full')[omega_global:-omega_global]
    snr_local = np.convolve(snr, h_local, mode='full')[omega_local:-omega_local]

    # github
    # snr_global[:hann_win_f_hal] = snr[:hann_win_f_hal]
    # snr_local[0] = snr[0]
    # tmp = np.convolve(win_freq.flatten(), snr[1:hann_win_f_hal].flatten()) # 频率平滑
    # snr_local[1:hann_win_f_hal] = tmp[f_win_length:(f_win_length+len(snr[1:hann_win_f_hal]))] # 长度裁剪

    # snr_global[-5:] = snr[-5:]
    # tmp = np.convolve(win_freq.flatten(), snr[-5:].flatten()) # 频率平滑
    # snr_local[-5:] = tmp[f_win_length:(f_win_length+len(snr[-5:]))] # 长度裁剪

    # for k in range(hann_win_f_hal, (len(snr)-5)):
    #     tmp = snr[k+hann_win_f_hal-8:k+hann_win_f_hal+1]
    #     tmp = tmp[::-1] # 反转
    #     a = np.sum(hann_win_f * tmp)
    #     snr_global[k] = (a / (hann_win_f_hal + 1))
    # tmp = np.convolve(win_freq.flatten(), snr[hann_win_f_hal:-5].flatten()) # 频率平滑
    # snr_local[hann_win_f_hal:-5] = tmp[f_win_length:(f_win_length+ \
    #                                len(snr[hann_win_f_hal:-5]))] # 长度裁剪

    p_local = np.where(snr_local <= SNR_MIN, 0, p_local)
    p_local = np.where((snr_local >= SNR_MAX), 1, p_local)
    #snr_min_log = np.log(SNR_MIN)
    #snr_max_log = np.log(SNR_MAX)
    p_local = np.where(0.5 == p_local, \
              (np.log(snr_local-SNR_MIN)) / (np.log(SNR_MAX-SNR_MIN)),\
              p_local)

    p_global = np.where(snr_global <= SNR_MIN, 0, p_global)
    p_global = np.where(snr_global >= SNR_MAX, 1, p_global)
    p_global = np.where(0.5 == p_global, \
              (np.log(snr_global-SNR_MIN)) / (np.log(SNR_MAX-SNR_MIN)),\
              p_global)
    
    # python
    snr_peak = np.max([snr_frame, SNR_P_MIN])
    snr_peak = np.min([snr_peak, SNR_P_MAX])
    if (snr_frame > SNR_MIN):
        if (snr_frame > snr_lframe):
            p_frame = 1
        else:
            if (snr_frame <= (snr_peak * SNR_MIN)):
                mu = 0 # \mu
            elif (snr_frame >= (snr_peak * SNR_MAX)):
                mu = 1
            else:
                mu = 1 * np.log(snr_frame * 1 / snr_peak / SNR_MIN) / np.log(SNR_MAX / SNR_MIN)
            p_frame = mu
    else:
        p_frame = 0
        
    # github
    # sum = np.sum(snr)
    # snr_frame = sum / (N_eff)
    # sum = 0
    # snr_peak = np.max([snr_frame, SNR_P_MIN])
    # snr_peak = np.min([snr_peak, SNR_P_MAX])

    # if (snr_frame <= (snr_peak * SNR_MAX)):
    #     mu = 0
    # elif (snr_frame >= (snr_peak * SNR_MAX)):
    #     mu = 1
    # else:
    #     mu = 1 * np.log(snr_frame * 1 / snr_peak / SNR_MIN) / np.log(SNR_MAX / SNR_MIN)
    
    # if (snr_frame > SNR_MIN):
    #     if (snr_frame > old_snr_frame):
    #         p_frame = 1
    #     else:
    #         p_frame = mu
    # else:
    #     p_frame = 0
    
    q_est = 1 - (p_local * p_global * p_frame)
    #q_est[:10] = 0.3 * q_est[:10] # 减少低频失真 # test
    q_est = np.where(q_est > q_max, q_max, q_est)
    old_snr = snr

    return old_snr, q_est

expsub_value = [16220,16059,15899,15741,15584,15429,15276,15124,14973,14824,14677,14531,14386,14243,14101,13961,13822,13685,13548,13414,13280,13148,13017,12888,12759,12632,12507,12382,12259,12137,12016,11897,11778,11661,11545,11430,11316,11204,11092,10982,10873,10765,10657,10551,10446,10342,10240,10138,10037,9937,9838,9740,9643,9547,9452,9358,9265,9173,9082,8991,8902,8813,8725,8639,8553,8468,8383,8300,8217,8136,8055,7974,7895,7817,7739,7662,7586,7510,7435,7361,7288,7216,7144,7073,7002,6933,6864,6795,6728,6661,6594,6529,6464,6400,6336,6273,6210,6149,6087,6027,5967,5907,5849,5791,5733,5676,5619,5563,5508,5453,5399,5345,5292,5239,5187,5136,5085,5034,4984,4934,4885,4837,4788,4741,4694,4647,4601,4555,4510,4465,4420,4376,4333,4290,4247,4205,4163,4121,4080,4040,4000,3960,3920,3881,3843,3804,3767,3729,3692,3655,3619,3583,3547,3512,3477,3442,3408,3374,3341,3307,3274,3242,3210,3178,3146,3115,3084,3053,3023,2993,2963,2933,2904,2875,2847,2818,2790,2762,2735,2708,2681,2654,2628,2602,2576,2550,2525,2500,2475,2450,2426,2402,2378,2354,2331,2307,2284,2262,2239,2217,2195,2173,2151,2130,2109,2088,2067,2046,2026,2006,1986,1966,1947,1927,1908,1889,1870,1852,1833,1815,1797,1779,1761,1744,1726,1709,1692,1675,1659,1642,1626,1610,1594,1578,1562,1546,1531,1516,1501,1486,1471,1456,1442,1428,1413,1399,1385,1372,1358,1344,1331,1318,1305,1292,1279,1266,1253,1241,1229,1216,1204,1192,1180,1169,1157,1146,1134,1123,1112,1101,1090,1079,1068,1057,1047,1036,1026,1016,1006,996,986,976,966,957,947,938,928,919,910,901,892,883,874,866,857,849,840,832,823,815,807,799,791,783,775,768,760,752,745,738,730,723,716,709,702,695,688,681,674,667,661,654,648,641,635,628,622,616,610,604,598,592,586,580,574,569,563,557,552,546,541,535,530,525,520,514,509,504,499,494,489,484,480,475,470,465,461,456,452,447,443,438,434,430,425,421,417,413,409,405,401,397,393,389,385,381,377,373,370,366,362,359,355,352,348,345,341,338,334,331,328,325,321,318,315,312,309,306,303,300,297,294,291,288,285,282,279,277,274,271,268,266,263,260,258,255,253,250,248,245,243,240,238,236,233,231,229,226,224,222,220,217,215,213,211,209,207,205,203,201,199,197,195,193,191,189,187,185,183,182,180,178,176,174,173,171,169,168,166,164,163,161,159,158,156,155,153,152,150,149,147,146,144,143,141,140,138,137,136,134,133,132,130,129,128,126,125,124,123,122,120,119,118,117,116,114,113,112,111,110,109,108,107,106,105,103,102,101,100,99,98,97,96,95,95,94,93,92,91,90,89,88,87,86,85,85,84,83,82,81,80,80,79,78,77,77,76,75,74,73,73,72,71,71,70,69,68,68,67,66,66,65,64,64,63,63,62,61,61,60,59,59,58,58,57,57,56,55,55,54,54,53,53,52,52,51,51,50,50,49,49,48,48,47,47,46,46,45,45,44,44,43,43,43,42,42,41,41,41,40,40,39,39,39,38,38,37,37,37,36,36,36,35,35,34,34,34,33,33,33,32,32,32,31,31,31,31,30,30,30,29,29,29,28,28,28,28,27,27,27,26,26,26,26,25,25,25,25,24,24,24,24,23,23,23,23,22,22,22,22,22,21,21,21,21,20,20,20,20,20,19,19,19,19,19,18,18,18,18,18,18,17,17,17,17,17,17,16,16,16,16,16,16,15,15,15,15,15,15,14,14,14,14,14,14,14,13,13,13,13,13,13,13,12,12,12,12,12,12,12,12,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def subexp_solution(v_subscript):
    v_subscript = v_subscript * 100
    vec = int(v_subscript)
    if (vec < 1):
        vec = 1
    elif (vec > 1500):
        vec = 1500
    g = expsub_value[vec-1]
    return g

NOISE_FACTOR = 0.001
def mcra_omlsa_filtering(v_int, q_est, cur_snr, gh1, amp_pr):
    #global pr_snr # 使用global进行全局修改
    integra = np.zeros(N_eff)
    for i in range(N_eff):
        integra[i] = subexp_solution(v_int[i])
    m_int = integra / 16384 # Q15->float
    arr_temp = 1 + ((1 + cur_snr) * integra * q_est) / (1 - q_est + 1e-7)

    p_est = 1 / (arr_temp + 1e-7)
    p_est = np.where(p_est < 0.0001, 0.0001, p_est)
    p_est = np.where(p_est > 1, 1, p_est)
    noise_fa = np.full(N_eff, NOISE_FACTOR) # G_min
    g = np.power(gh1, p_est) * np.power(noise_fa, (1 - p_est)) # G
    g = np.where(g > 1, 1, g)
    g = np.where(g < 0, 0, g)
    amp_last = g * amp_pr
    #pr_snr = np.power(g, 2) + 1e-7 #np.power((amp_last / noise_est + 1e-30), 2) # G^2
    #pr_snr = np.where(pr_snr > 10000, 10000, pr_snr)

    return g

tr_alpha_s = 0.7
tr_alpha_p = 0.2
tr_alpha_d = 0
tr_loop_i = 0
fcwin_L = 5
tr_buf = np.zeros(transient_frame_len) # 64
tr_buf_out = np.zeros(transient_frame_len) # 64
tr_min_bff = np.zeros([fcwin_L, transient_N_eff])
tr_amp = np.zeros(transient_N_eff)
tr_amp_min = np.zeros(transient_N_eff)
tr_amp_tmp = np.zeros(transient_N_eff)
tr_init_p = np.zeros(transient_N_eff)
tr_noise_est_tmp = np.zeros(transient_N_eff)
tr_ana_win, tr_syn_win = hann_window(transient_frame_len, transient_frame_hop)
print("tr_ana_win[]")
for i in range(len(tr_ana_win)):
    print(tr_ana_win[i], ',', sep='')
def mcra_transient(in_buf):
    global tr_buf_out, tr_buf, tr_loop_i, tr_amp, tr_min_bff, tr_noise_est_tmp, \
            tr_amp_min, tr_amp_tmp
    tr_noise_est = np.zeros([4, transient_N_eff])
    alpha_wideh = np.zeros(transient_N_eff)
    out_buf = np.zeros_like(in_buf)
    for i in range(4):
        tmp = in_buf[i * transient_frame_hop : (i+1) * transient_frame_hop] # 32
        tr_buf[(transient_frame_len-transient_frame_hop):] = tmp
        tr_buf_win = tr_buf * tr_ana_win
        tr_buf_fft = np.fft.rfft(tr_buf_win) # t->f

        tr_amp_pr = np.sqrt((tr_buf_fft.real ** 2) + (tr_buf_fft.imag ** 2))
        tr_amp = tr_alpha_s * tr_amp + (1-tr_alpha_s) * tr_amp_pr # S(k,l) \alpha_s
        #if (tr_loop_i < fcwin_L):
        #    tr_min_bff[tr_loop_i, :] = tr_amp
        #    tr_loop_i = tr_loop_i + 1
        #else:
        #    tr_min_bff[:(fcwin_L-1)] = tr_min_bff[1:] # 清除上一帧老数据
        #    tr_min_bff[-1] = tr_amp                   # 放置新数据
        #tr_amp_min = np.min(tr_min_bff, axis=0)       # 获取最小值
        if ((0 == tr_loop_i % fcwin_L) and (tr_loop_i != 0)):
            tr_amp_min = np.min([tr_amp_tmp, tr_amp], axis=0)
            tr_amp_tmp = tr_amp
        else:
            tr_amp_min = np.min([tr_amp_min, tr_amp], axis=0)
            tr_amp_tmp = np.min([tr_amp_tmp, tr_amp], axis=0)
        tr_loop_i = tr_loop_i + 1

        for k in range(transient_N_eff):
            if (tr_amp[k] > (5 * tr_amp_min[k])): # -5dB
                p = 1
            else:
                p = 0
            tr_init_p[k] = tr_alpha_p * tr_init_p[k] + (1 - tr_alpha_p) * p
            alpha_wideh[k] = tr_alpha_d + (1 - tr_alpha_d) * tr_init_p[k]
            tr_noise_est[i, k] = tr_init_p[k] * tr_amp_pr[k]
            #tr_noise_est[i, k] = alpha_wideh[k] * tr_noise_est_tmp[k] + \
            #                     (1 - alpha_wideh[k]) * tr_amp_pr[k]
            #tr_noise_est[i, k] = (alpha_wideh[k]) * tr_amp_pr[k]
            #tr_noise_est[i, k] = ((tr_alpha_d * tr_noise_est_tmp[k] + (1-tr_alpha_d)\
            #                       * tr_amp_pr[k]) * (1 - tr_init_p[k]) + \
            #                      tr_noise_est_tmp[k] * tr_init_p[k])
        #alpha_wideh = tr_alpha_d + (1 - tr_alpha_d) * tr_init_p
        tr_noise_est_tmp = tr_noise_est[i, :] # 预估完噪声上一帧数据

        tr_buf_ifft = np.fft.irfft(tr_buf_fft) # t->f
        tr_buf_syn = tr_buf_ifft * tr_syn_win
        tr_buf_out = tr_buf_out + tr_buf_syn
        out_buf[i * transient_frame_hop : (i+1) * transient_frame_hop] = \
            tr_buf_out[:transient_frame_hop]
        tr_buf_out[:(transient_frame_len-transient_frame_hop)] = \
            tr_buf_out[transient_frame_hop:]
        tr_buf_out[(transient_frame_len-transient_frame_hop):] = 0
        tr_buf[:(transient_frame_len-transient_frame_hop)] = tr_buf[transient_frame_hop:]

        # test
        #tr_noise_est[i] = tr_init_p
    
    # 测试帧移数据是否正确
    #return out_buf
    # 返回预估噪声
    return np.sum(tr_noise_est, axis=0)

if __name__ == "__main__":
    print("omlsa TRANSIENT noise V0.3")
    IMCRA_open = False

    wav_d, _ = sf.read(r'wav\path')

    out = np.zeros(len(wav_d))
    out_tr_test = np.zeros(len(wav_d))
    in_buff = np.zeros(frame_length)
    out_buff = np.zeros(frame_length)
    old_snr = np.zeros(N_eff)
    gh1 = np.zeros(N_eff)

    # 标准hann
    #ana_win, syn_win = hann_window(frame_length, frame_move)
    # RNNoise window
    half_window = check_init()
    window = np.concatenate([half_window, half_window[::-1]])

    frame_num = len(wav_d) // frame_move - 3

    noisy = []
    noise_plot = []
    gh1_plot = []
    q_est_plot = []
    g_plot = []
    tr_noise_plot = []
    tr_noise_plot_ex = []

    for n in range(frame_num):
        frame_data = wav_d[n*frame_move:(n+1)*frame_move]
        in_buff[(frame_length-frame_move):] = frame_data
        # 测试帧移数据是否正确
        #out_tr_test[n*frame_move:(n+1)*frame_move] = mcra_transient(frame_data)
        # 瞬态噪声估计代码
        tr_noise_estimate = mcra_transient(frame_data)
        tr_noise_plot.append(tr_noise_estimate)
        x = np.linspace(0, 1, transient_N_eff)
        y = tr_noise_estimate
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, 1, N_eff)
        tr_noise_estff = f(x_new) # 瞬态噪声插值到稳态噪声相同长度
        tr_noise_plot_ex.append(tr_noise_estff)

        in_buff_win = in_buff * window
        in_fft = np.fft.rfft(in_buff_win) # T->F

        if IMCRA_open: # imcra
            in_fft = omlsa_streamer(in_fft)
        else: # mcra
            noise_plot_part, amp_pr = np.copy(mcra_streamer(in_fft)) # 防止全局变量共享内存
            noise_plot_part = noise_plot_part + tr_noise_estff # 稳态噪声与瞬态噪声相加 # 只要稳态噪声可以不要
            noise_plot.append(noise_plot_part)
            noisy.append(abs(in_fft))
            gh1, cur_snr, v_int = get_gh1(noise_plot_part, amp_pr, gh1) # gh1 语音的Mask
            gh1_plot.append(gh1)
            old_snr, q_est = mcra_omlsa_speech_absent_est(old_snr, cur_snr) # q_est_语音不存在概率
            q_est_plot.append(q_est)
            g = mcra_omlsa_filtering(v_int, q_est, cur_snr, gh1, amp_pr)
            g_plot.append(g)
            g = np.where(g > 1, 1, g)
            g = np.where(g < 0, 0, g)
            in_fft = in_fft * g
            loop_i = loop_i + 1

        out_buff_win = np.fft.irfft(in_fft) # F->T
        out_buff_syn = out_buff_win * window
        out_buff = out_buff + out_buff_syn
        out[n*frame_move:(n+1)*frame_move] = out_buff[:frame_move]
        out_buff[:(frame_length-frame_move)] = out_buff[frame_move:]
        out_buff[(frame_length-frame_move):] = 0
        in_buff[:(frame_length-frame_move)] = in_buff[frame_move:]

    if not IMCRA_open:
        noise_plot = np.array(noise_plot).T
        noisy = np.array(noisy).T
        gh1_plot = np.array(gh1_plot).T
        q_est_plot = np.array(q_est_plot).T
        g_plot = np.array(g_plot).T
        tr_noise_plot = np.array(tr_noise_plot)
        tr_noise_plot = tr_noise_plot.reshape(-1, transient_N_eff).T
        tr_noise_plot_ex = np.array(tr_noise_plot_ex).T
        # 该操作放到上面
        # tr_noise_plot_ex = np.zeros_like(noise_plot)
        # for i in range(len(tr_noise_plot[0, :])):
        #     x = np.linspace(0, 1, transient_N_eff)
        #     y = tr_noise_plot[:, i]
        #     f = interpolate.interp1d(x, y)
        #     x_new = np.linspace(0, 1, N_eff)
        #     y_new = f(x_new)
        #     tr_noise_plot_ex[:, i] = y_new

        fig=plt.figure(num=1)
        ax = fig.add_subplot(511)
        plt.imshow(noisy, origin='lower', aspect='auto')
        plt.clim(0, 0.2)

        ax = fig.add_subplot(512)
        plt.imshow(noise_plot, origin='lower', aspect='auto')
        plt.clim(0, 0.2)

        ax = fig.add_subplot(513)
        plt.imshow(gh1_plot, origin='lower', aspect='auto')
        plt.clim(0, 1)

        ax = fig.add_subplot(514)
        plt.imshow(q_est_plot, origin='lower', aspect='auto')
        plt.clim(0, 1)

        ax = fig.add_subplot(515)
        plt.imshow(g_plot, origin='lower', aspect='auto')
        plt.clim(0, 1)

        plt.tight_layout()

        fig=plt.figure(num=2)
        ax = fig.add_subplot(311)
        plt.imshow(noisy, origin='lower', aspect='auto')
        plt.clim(0, 0.2)

        ax = fig.add_subplot(312)
        plt.imshow(tr_noise_plot[:, :], origin='lower', aspect='auto')
        plt.clim(0, 0.2)

        ax = fig.add_subplot(313)
        plt.imshow(tr_noise_plot_ex[:, :], origin='lower', aspect='auto')
        plt.clim(0, 0.2)

        plt.tight_layout()
        plt.show()

    sf.write(r'wav\out\path', out, fs)
    # 测试帧移数据是否正确
    #sf.write('./wav/out_tr.wav', out_tr_test, fs)


