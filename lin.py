import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

# Values
# Original list sweeping number of units
# lst = np.array([[1,793.5], [2,787.1], [3,780.7], [4,774.5], [5,768.4], [6,762.4], [7,756.4], [8,750.6], [9,744.8], [10,739.2], [16,706.9], [20,686.9], [30,641.5 ], [32,633.2], [40,601.9], [50,567], [60,536], [64,524.6]])

# List of values sweeping number of RWL pulses...20 units
# lst = np.array([[0 , 600], [1 , 561.4], [2 , 523.7], [3 , 486.8], [4 , 450.8], [5 , 415.6], [6 , 381.2], [7 , 347.6], [8 , 314.9], [9 , 283.2], [10, 252.6], [11, 222.7], [12, 194.3], [13, 167.5], [14, 142.4], [15, 119.4]])


# List of values sweeping number of cells
# lst = np.array([[1 , 598], [2 , 596.1], [3 , 594.1], [4 , 592.2], [5 , 590.3], [8 , 584.4], [10, 580.6], [12, 576.7], [16, 569.1], [20, 561.4], [25, 552], [32, 538.8], [40, 523.9], [50, 505.6], [64, 480.4]])

# 0.6V, 100ps, 400fF...FOR uvtnfet I think
# 0.8V, 100ps, 50fF...200ps, 100fF FOR uvtnfet

# 0.8V, 100ps, 120fF...FOR nfet VARIATION GETS WORSE

# Sweeping number of RWL Pulses
lst_rwl = np.array([[1, 800], [2, 750], [3, 700.9], [4, 652.9], [5, 605.9], [6, 559.9], [7, 515], [8, 471.2], [9, 428.6], [10, 387.2], [11, 347.3], [12, 309.1], [13, 272.8], [14, 238.8], [15, 207.3], [16, 178.5]]) - np.array([1, 0])

# Sweeping number of cells
lst_cells = np.array([[0, 800], [1, 750.1], [2, 701.1], [3, 653.3], [4, 606.5], [5, 560.8], [6, 516.1], [7, 472.6], [8, 430.2], [9, 388.9], [10, 349], [11, 310.5], [12, 273.6], [13, 238.6], [14, 205.9], [15, 175.7], [16, 148.3]])


lst_rwl = lst_rwl.T
lst_cells = lst_cells.T

def f(x, a, b):
    return a * x + b

result = scipy.optimize.curve_fit(f, lst_rwl[0], lst_rwl[1], p0=(-5, 800))
a, b = result[0]

change = []
for i in range(1, len(lst_rwl.T)):
    change += [(lst_rwl[1][i] - lst_rwl[1][i - 1]) / (lst_rwl[0][i - 1] - lst_rwl[0][i])]


lst1 = [f(x, a, b) for x in lst_rwl[0]]

num = 8
vdd = lst_rwl[1][0]
plt.plot(lst_rwl[0][:num], lst_rwl[1][:num], label="simulated result")
avg = (lst_rwl[1][0] - lst_rwl[1][num - 1]) / (num - 1)
adc_decision = np.ones(len(lst_rwl[1])) * vdd - np.array(list(range(len(lst_rwl[1])))) * avg
plt.plot(lst_rwl[0][:num], adc_decision[:num], label="ADC Decision Levels")
plt.legend()
plt.show()

plt.plot(lst_rwl[0][:num], np.array(adc_decision[:num]) - np.array(lst_rwl[1][:num]), label="Difference between ADC Decision Level and Measured Results")
plt.xlabel("RWL Pulses")
plt.ylabel("Output Voltage Difference (mV)")
plt.grid()
plt.show()

plt.plot(lst_rwl[0], np.array(adc_decision) - np.array(lst_rwl[1]), label="Difference between ADC Decision Level and Measured Results")
plt.show()

# plt.figure()
# plt.plot(lst_rwl[0], lst_rwl[1], label="Simulated Data")
# plt.plot(lst_rwl[0], lst1, label="Best Fit Curve")
# # plt.xlabel("RWL Pulses")
# plt.xlabel("SRAM Units")
# plt.ylabel("Output Voltage (mV)")
# plt.title("Output Voltage versus RWL Pulses\nFor VDD=600mV")
# plt.legend()
# plt.show()
#
#
# plt.figure()
# plt.xlabel("RWL Pulses")
# # plt.xlabel("SRAM Units")
# # plt.xlabel("RWL Pulses\nSRAM Units")
# plt.ylabel("Output Voltage (mV)")
# plt.title("Output Voltage versus RWL Pulses")
# # plt.plot(lst[0], lst[1], label="Simulated Data", marker=".")
# plt.plot(lst_rwl[0], lst_rwl[1], label="Sweeping RWL Pulses", marker=".")
# # plt.plot(lst_cells[0], lst_cells[1], label="Sweeping number of SRAM Units", marker=".")
# # plt.legend()
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.plot(lst_rwl[0][1:], change, marker=".")
# plt.xlabel("RWL Pulses")
# # plt.xlabel("SRAM Units")
# plt.ylabel("Difference from previous (mV)")
# plt.title("Looking at change in voltage per RWL Pulse applied")
# plt.grid()
# plt.show()
#
#
#
# plt.figure()
# plt.xlabel("Intended Digital Multiplication Result")
# plt.ylabel("Output Voltage Difference (mV)")
# plt.plot(lst_rwl[0], -(lst_rwl[1] - lst_cells[1][:16]), marker=".")
# plt.grid()
# plt.show()
