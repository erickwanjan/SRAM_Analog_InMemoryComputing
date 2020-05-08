import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# folder = "ee290_data"
folder = "ee290_dat_new"

def parse_single(pulse_num, num_units, num_bits):
    fname = folder + "/Vend" + str(pulse_num) + "_" + str(num_units) + "unit_" + str(num_bits) + "bit.vcsv"
    f = open(fname, "r")
    dat = f.read()
    lst = dat.split("\n")
    lst.pop()
    lst = lst[6:]
    lst = [float(x.split(",")[1]) for x in lst]
    return lst

def parse_noise(num_units, num_bits, vdd=0.8, parts=None):
    if parts == None:
        all_lst = parse_noise_helper(num_units, num_bits, vdd=vdd)
    else:
        all_lst = []
        for i in range(parts):
            all_lst += parse_noise_helper("part" + str(i + 1) + "_" + str(num_units), num_bits, vdd=vdd)
    return all_lst

def parse_noise_helper(num_units, num_bits, vdd=0.8):
    fname = folder + "/noise_" + str(num_units) + "unit_" + str(num_bits) + "bit.vcsv"
    f = open(fname, "r")
    dat = f.read()
    lst = dat.split("\n")
    lst.pop()
    lst = lst[6:]
    lst = np.array([[float(y) for y in x.split(",")][1::2] for x in lst])
    lstT = np.array(lst).T
    offset = vdd - lstT[0]
    lstT += offset
    return lstT.T.tolist()

def parse_num_bits(num_units, num_bits, vdd=0.8):
    max_pulse_num = int(2**num_bits / num_units)
    parse_lst = []
    for i in range(1, max_pulse_num + 1):
        parse_lst += [parse_single(i, num_units, num_bits)]
    parse_lst = [[vdd for x in range(len(parse_lst[0]))]] + parse_lst
    parse_lst = np.array(parse_lst).T.tolist()
    return parse_lst



def get_differences(lst, ind_a, ind_b):
    return [x[ind_a - 1] - x[ind_b - 1] for x in lst]

# Same as get_differences, but run on each entry
def make_point_diff(lst):
    temp_lst = np.array(lst).T
    new_lst = []
    for i in range(1, len(temp_lst)):
        new_lst += [temp_lst[i - 1] - temp_lst[i]]
    return np.array(new_lst).T.tolist()

def get_std(lst, over_avg=False):
    avg = sum(lst) / len(lst)
    if over_avg:
        return sqrt(sum([x**2 for x in (np.array(lst) - avg).tolist()]) / len(lst)) / avg
        # return sqrt(sum([x**2 for x in (np.array(lst) - avg).tolist()]) / len(lst) / avg)
    else:
        return sqrt(sum([x**2 for x in (np.array(lst) - avg).tolist()]) / len(lst))


def get_diff_std(lst):
    std_lst = []
    for i in range(1, len(lst[0])):
        cur_diff = get_differences(lst, i, i + 1)
        std_lst += [get_std(cur_diff)]
    return std_lst

def get_incr_std(lst):
    return [get_std(cur_lst) for cur_lst in np.array(lst).T.tolist()]


def transpose(lst):
    return np.array(lst).T.tolist()

def parse_dat_file(str_name, num_bits, vdd=0.8):
    fname = folder + "/dat" + str(str_name)
    f = open(fname, "r")
    dat = f.read()
    lst = dat.split()
    cur_val = -1
    all_dat = []
    for i in range(2**num_bits + 1):
        cur_val += 7
        cur_lst = lst[cur_val:cur_val + 300]
        temp_dat = []
        for j in range(100):
            temp_dat += [float(cur_lst[3 * j + 2])]
        all_dat += [temp_dat]
        cur_val += 300

    offset_lst = vdd - np.array(all_dat[0])
    all_dat = (np.array(all_dat) + offset_lst).tolist()
    return transpose(all_dat)

def parse_rsnm(vdd):
    fname = folder + "/" + str(vdd) + "_snm.vcsv"
    f = open(fname, "r")
    dat = f.read()
    lst = dat.split("\n")
    lst.pop()
    lst = lst[6:]
    lst = [[float(y) for y in x.split(",")] for x in lst]
    return transpose(lst)

def cross(lst, cross_val):
    count = 0
    while lst[count] > cross_val:
        count += 1
    return count

def get_area(lst, vdd=0.8):
    max_side = -float("inf")
    for i in range(len(lst[0])):
        x1 = lst[0][i]
        y1 = lst[1][i]

        x2 = vdd - lst[1][i]
        y2 = vdd - lst[0][i]

        # print("(", x1, ",", y1, ") , (",x2, ",", y2, ")")
        side = x1 - x2
        if side > max_side:
            max_side = side
        else:
            return max_side

def get_max(lst, true_max=True):
    if true_max:
        return max(lst)
    else:
        return sqrt(sum([x**2 for x in lst]) / len(lst))



rsnm_0p8 = parse_rsnm("0p8")
rsnmv_0p8 = get_area(rsnm_0p8, 0.8)
# plt.figure()
# plt.plot(rsnm_0p8[0], rsnm_0p8[1])
# plt.plot(rsnm_0p8[1], rsnm_0p8[0])
# plt.xlabel("V (V)")
# plt.ylabel("VB (V)")
# plt.title("SRAM Read Static Noise Margin\nFor VDD: 0.8V")
# plt.show()

rsnm_1p0 = parse_rsnm("1p0")
rsnmv_1p0 = get_area(rsnm_1p0, 1)
# plt.figure()
# plt.plot(rsnm_1p0[0], rsnm_1p0[1])
# plt.plot(rsnm_1p0[1], rsnm_1p0[0])
# plt.xlabel("V (V)")
# plt.ylabel("VB (V)")
# plt.title("SRAM Read Static Noise Margin\nFor VDD: 1.0V")
# plt.show()


lst1 = parse_num_bits(1, 4)
lst2 = parse_num_bits(2, 4)
lst4 = parse_num_bits(4, 4)
lst8 = parse_num_bits(8, 4)
lst16 = parse_num_bits(16, 4)

# Setting DNL Requirement
diff1 = get_diff_std(lst1)
diff1_comp2 = get_diff_std(transpose(transpose(lst1)[::2]))
diff1_comp4 = get_diff_std(transpose(transpose(lst1)[::4]))
diff1_comp8 = get_diff_std(transpose(transpose(lst1)[::8]))
diff1_comp16 = get_diff_std(transpose(transpose(lst1)[::16]))
diff2 = get_diff_std(lst2)
diff2_comp4 = get_diff_std(transpose(transpose(lst2)[::2]))
diff2_comp8 = get_diff_std(transpose(transpose(lst2)[::4]))
diff2_comp16 = get_diff_std(transpose(transpose(lst2)[::8]))
diff4 = get_diff_std(lst4)
diff4_comp8 = get_diff_std(transpose(transpose(lst4)[::2]))
diff4_comp16 = get_diff_std(transpose(transpose(lst4)[::4]))
diff8 = get_diff_std(lst8)
diff8_comp16 = get_diff_std(transpose(transpose(lst8)[::2]))
diff16 = get_diff_std(lst16)

plt.figure()
plt.title("Interval Width Standard Deviation\nPulsing one SRAM Unit RWL Repeatedly")
plt.plot(np.array(diff1) * 1e3)
plt.xlabel("Increment number (VDD/16-valued steps)")
plt.ylabel("Interval Width Standard Deviation (mV)")
plt.show()

plt.figure()
plt.title("Sweeping number of SRAM Units versus Interal Width Standard Deviation")
plt.plot(np.array(diff1_comp4) * 1e3, label="1 SRAM Unit")
plt.plot(np.array(diff2_comp4) * 1e3, label="2 SRAM Units")
plt.plot(np.array(diff4) * 1e3, label="4 SRAM Units")
plt.xlabel("Increment number (VDD/4-valued steps)")
plt.ylabel("Interval Width Standard Deviation (mV)")
plt.xticks([0, 1, 2, 3])
plt.legend()
plt.show()

max_diff1 = get_max(diff1)
max_diff2 = get_max(diff2)
max_diff4 = get_max(diff4)
max_diff8 = get_max(diff8)
max_diff16 = get_max(diff16)

# Setting INL Requirement
incr1 = get_incr_std(lst1)
incr2 = get_incr_std(lst2)
incr4 = get_incr_std(lst4)
incr8 = get_incr_std(lst8)
incr16 = get_incr_std(lst16)

max_incr1 = get_max(incr1)
max_incr2 = get_max(incr2)
max_incr4 = get_max(incr4)
max_incr8 = get_max(incr8)
max_incr16 = get_max(incr16)

# diff = get_differences(lst, 1, 4)

plt.figure()
plt.title("Standard Deviation of Analog Voltage Representations\nPulsing one SRAM Unit RWL repeatedly")
plt.plot(np.array(incr1) * 1e3)
plt.xlabel("Digital Value Equivalent")
plt.ylabel("Analog Voltage Standard Deviation (mV)")
plt.show()

plt.figure()
plt.title("Sweeping number of SRAM Units looking at\nStandard Deviation of Analog Voltage Representations")
plt.plot(np.array(incr1[::4]) * 1e3, label="1 SRAM Unit")
plt.plot(np.array(incr2[::2]) * 1e3, label="2 SRAM Units")
plt.plot(np.array(incr4) * 1e3, label="4 SRAM Units")
plt.xlabel("Digital Value Equivalent")
plt.ylabel("Analog Voltage Standard Deviation (mV)")
plt.xticks([0, 1, 2, 3])
plt.legend()
plt.show()




lst1_5 = parse_num_bits(1, 5)
diff1_5 = get_diff_std(lst1_5)
max_diff1_5 = get_max(diff1_5)
incr1_5 = get_incr_std(lst1_5)
max_incr1_5 = get_max(incr1_5)
diff1_5_comp2 = get_diff_std(transpose(transpose(lst1_5)[::2]))
diff1_5_comp4 = get_diff_std(transpose(transpose(lst1_5)[::4]))
diff1_5_comp8 = get_diff_std(transpose(transpose(lst1_5)[::8]))
diff1_5_comp16 = get_diff_std(transpose(transpose(lst1_5)[::16]))


lst1_6 = parse_dat_file("64", 6)
diff1_6 = get_diff_std(lst1_6)
max_diff1_6 = get_max(diff1_6)
incr1_6 = get_incr_std(lst1_6)
max_incr1_6 = get_max(incr1_6)
diff1_6_comp2 = get_diff_std(transpose(transpose(lst1_6)[::2]))
diff1_6_comp4 = get_diff_std(transpose(transpose(lst1_6)[::4]))
diff1_6_comp8 = get_diff_std(transpose(transpose(lst1_6)[::8]))
diff1_6_comp16 = get_diff_std(transpose(transpose(lst1_6)[::16]))

lst1_3 = parse_dat_file("8", 3)
diff1_3 = get_diff_std(lst1_3)
max_diff1_3 = get_max(diff1_3)
incr1_3 = get_incr_std(lst1_3)
max_incr1_3 = get_max(incr1_3)
diff1_3_comp2 = get_diff_std(transpose(transpose(lst1_3)[::2]))
diff1_3_comp4 = get_diff_std(transpose(transpose(lst1_3)[::4]))
diff1_3_comp8 = get_diff_std(transpose(transpose(lst1_3)[::8]))
diff1_3_comp16 = get_diff_std(transpose(transpose(lst1_3)[::16]))


plt.figure()
plt.title("Bits of Precision versus Interval Width Standard Deviation")
plt.plot(np.array(diff1_3) * 1e3, label="3-bit precision")
plt.plot(np.array(diff1_comp2) * 1e3, label="4-bit precision")
plt.plot(np.array(diff1_5_comp4) * 1e3, label="5-bit precision")
plt.plot(np.array(diff1_6_comp8) * 1e3, label="6-bit precision")
plt.legend()
plt.xlabel("Increment number (VDD/8-valued steps)")
plt.ylabel("Interval Width Standard Deviation (mV)")
plt.show()

lst1p0_1_4 = parse_dat_file("16_1p0", 4, 1.0)
diff1p0_1_4 = get_diff_std(lst1p0_1_4)
max_diff1p0_1_4 = get_max(diff1p0_1_4)
incr1p0_1_4 = get_incr_std(lst1p0_1_4)
max_incr1p0_1_4 = get_max(incr1p0_1_4)

lst1p1_1_4 = parse_dat_file("16_1p1", 4, 1.1)
diff1p1_1_4 = get_diff_std(lst1p1_1_4)
max_diff1p1_1_4 = get_max(diff1p1_1_4)
incr1p1_1_4 = get_incr_std(lst1p1_1_4)
max_incr1p1_1_4 = get_max(incr1p1_1_4)

lst0p6_1_4 = parse_dat_file("16_0p6", 4, 0.6)
diff0p6_1_4 = get_diff_std(lst0p6_1_4)
max_diff0p6_1_4 = get_max(diff0p6_1_4)
incr0p6_1_4 = get_incr_std(lst0p6_1_4)
max_incr0p6_1_4 = get_max(incr0p6_1_4)


vdd_lst = [0.6, 0.8, 1, 1.1]
plt.figure()
plt.plot(vdd_lst, [max_diff0p6_1_4, max_diff1, max_diff1p0_1_4, max_diff1p1_1_4])
plt.show()


# 1 unit, 4 bits, vdd=0.8, using 1.7pF cap and 400ps time
lstlarge_1_4 = parse_dat_file("16_large", 4)
difflarge_1_4 = get_diff_std(lstlarge_1_4)
max_difflarge_1_4 = get_max(difflarge_1_4)
incrlarge_1_4 = get_incr_std(lstlarge_1_4)
max_incrlarge_1_4 = get_max(incrlarge_1_4)

lst5bitsout_1_4 = parse_dat_file("16_2unit_5bits", 4)
diff5bitsout_1_4 = get_diff_std(lst5bitsout_1_4)
max_diff5bitsout_1_4 = get_max(diff5bitsout_1_4)
incr5bitsout_1_4 = get_incr_std(lst5bitsout_1_4)
max_incr5bitsout_1_4 = get_max(incr5bitsout_1_4)

lst6bitsout_1_4 = parse_dat_file("16_4unit_6bits", 4)
diff6bitsout_1_4 = get_diff_std(lst6bitsout_1_4)
max_diff6bitsout_1_4 = get_max(diff6bitsout_1_4)
incr6bitsout_1_4 = get_incr_std(lst6bitsout_1_4)
max_incr6bitsout_1_4 = get_max(incr6bitsout_1_4)


noise1_4 = parse_noise(1, 4)
noise1_4_std = get_incr_std(noise1_4)
noise2_4 = parse_noise(2, 4)
noise2_4_std = get_incr_std(noise2_4)
noise4_4 = parse_noise(4, 4)
noise4_4_std = get_incr_std(noise4_4)
noise8_4 = parse_noise(8, 4, parts=2)
noise8_4_std = get_incr_std(noise8_4)
noise16_4 = parse_noise(16, 4)
noise16_4_std = get_incr_std(noise16_4)


noise1_3 = parse_noise(1, 3)
noise1_3_std = get_incr_std(noise1_3)
noise1_5 = parse_noise(1, 5)
noise1_5_std = get_incr_std(noise1_5)
noise1_6 = parse_noise(1, 6)
noise1_6_std = get_incr_std(noise1_6)



noiseoff1_3 = parse_noise("off_1", 3)
noiseoff1_3_std = get_incr_std(noiseoff1_3)
noiseoff1_4 = parse_noise("off_1", 4)
noiseoff1_4_std = get_incr_std(noiseoff1_4)

noiseshortint1_3 = parse_noise("shortint_1", 3)
noiseshortint1_3_std = get_incr_std(noiseshortint1_3)
noiseshortint1_5 = parse_noise("shortint_1", 5)
noiseshortint1_5_std = get_incr_std(noiseshortint1_5)
noiseshortint1_6 = parse_noise("shortint_1", 6)
noiseshortint1_6_std = get_incr_std(noiseshortint1_6)

noisebck16_4 = parse_noise("bck_" + str(16), 4)
noisebck16_4_std = get_incr_std(noisebck16_4)



# diff_lst = [max_diff1, max_diff2, max_diff4, max_diff8, max_diff16]
# plt.figure()
# plt.plot([1, 2, 4, 8, 16], diff_lst)
# plt.xlabel("Number of SRAM cells with weight=1")
# plt.ylabel("Standard deviation of voltage level (V)")
# plt.title("Plotting change in")
# plt.show()
