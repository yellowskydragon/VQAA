import os

from run_VQAA import *
from pathlib import Path
from utils import *
from numpy import polyfit, poly1d


## 从上到下依次是1qb、2qb、spam噪声；从左到右分别是，噪声为0->噪声最大的时候迭代次数,这一版本是没有去除无法得到正确结果的
# iter_nums = [
# [8.6333, 15.5667, 12.6667, 15.6333, 19.7, 21.7667, 25.5, 29.2667, 34.8, 35.5667, 29.1333, 36.4333, 33.6333, 38.2, 37.0667, 43.3, 36.3, 40.3333, 43.9],
# [7.8571, 8.9143, 10.2, 15.0857, 15.7143, 11.6571, 14.1429, 14.1714, 19.0286, 19.6286, 22.9143, 27.0857, 26.1143, 23.4857, 28.5429, 33.2, 27.6, 25.1714, 29.0857],
# [8.16, 9.04, 13.64, 14.6, 16.84, 16.4, 23.36, 27.52, 27.04, 27.08, 31.8, 30.36, 35.08, 35.16, 33.72, 34.32, 28.96, 34.96, 39.6]
# ]

## 从上到下依次是1qb、2qb、spam噪声；从左到右分别是，噪声为0->噪声最大的时候迭代次数,这一版去除无法得到正确结果的
iter_nums = [
[8.6333, 13.1071, 11.3793, 11.8148, 12.125, 19.75, 16.5909, 21.7273, 21.5, 22.9375, 18.7, 20.9286, 19.3125, 22.7692, 22.2857, 27.6667, 18.3846, 27.6923, 29.6667],
[7.8571, 8.9143, 10.2, 11.8125, 14.7059, 10.5294, 13.0882, 13.1176, 13.8667, 15.7097, 17.3103, 19.1538, 19.037, 15.6296, 19.96, 19.0526, 14.3636, 15.24, 21.8462],
[8.16, 9.04, 12.125, 11.5217, 13.9565, 13.4783, 10.8235, 18.7778, 16.2353, 14.1875, 21.5625, 22.7222, 21.3077, 26.8125, 22.8667, 10.8, 19.0588, 23.1429, 24.0]
]

# less_than_max_iter_num = [
# [30, 28, 29, 27, 24, 28, 22, 22, 16, 16, 20, 14, 16, 13, 14, 9, 13, 13, 9],
# [35, 35, 35, 32, 34, 34, 34, 34, 30, 31, 29, 26, 27, 27, 25, 19, 22, 25, 26],
# [25, 25, 24, 23, 23, 23, 17, 18, 17, 16, 16, 18, 13, 16, 15, 10, 17, 14, 10]
# ]

less_than_max_iter_num_percent = [
[0.75, 0.7, 0.72, 0.68, 0.6, 0.7, 0.55, 0.55, 0.4, 0.4, 0.5, 0.35, 0.4, 0.33, 0.35, 0.23, 0.33, 0.33, 0.23],
[0.88, 0.88, 0.88, 0.8, 0.85, 0.85, 0.85, 0.85, 0.75, 0.78, 0.72, 0.65, 0.68, 0.68, 0.62, 0.47, 0.55, 0.62, 0.65],
[0.62, 0.62, 0.6, 0.57, 0.57, 0.57, 0.42, 0.45, 0.42, 0.4, 0.4, 0.45, 0.33, 0.4, 0.38, 0.25, 0.42, 0.35, 0.25]
]

noise_level = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009000000000000001, 0.010000000000000002, 0.011000000000000003, 0.012000000000000004, 0.013000000000000005, 0.014000000000000005, 0.015000000000000006, 0.016000000000000007, 0.017000000000000008, 0.01800000000000001, 0.01900000000000001]

begin = 0.001
step = 0.001
max_ = 0.02

gpu = 0
ansatz_type = "rx"
opti_method = "gd"
if_back = 0
key = "11001010"
plaintext = "10110010"
ciphertext = "10000111"
max_iter = 50
repeat = "0, 5"
end_prob = 0.5


def get_correspond_repeat(error_type):
    index = [4, 9, 14, 18]
    for idx in index:
        level = noise_level[idx]
        print(f"current noise type is {error_type}, level is {level}")
        repeated_list_path = Path(f"./results_{error_type}")
        repeat_list = [(child_file.name.split('_')[-1]) for child_file in repeated_list_path.iterdir()]
        repeat_times = len(repeat_list)
        for i in repeat_list:
            iter_file_path = Path(f"./results_{error_type}/repeat_{i}/type_{error_type}/"
                                  f"level_{level}/ansatz_{ansatz_type}/"
                                  f"optimize_{opti_method}/if_back_{str(if_back)}/"
                                  f"p_{plaintext}_k_{key}_c_{ciphertext}")
            iter_num = len(os.listdir(iter_file_path))
            print(f"repeat is : {i} ; iter_nums is {iter_num - 1}")


def draw(iter_nums):
    word_font = 30
    if_fit = [1]
    markers = ["o", "*", "^", "s", "p", "v", "D", "<", ">"]
    lines = ["-", "--", "dotted"]
    colors = ["gold", "royalblue", "olive", "black", "grey", "lightcoral", "red", "peachpuff", "tan", "teal", "violet"]

    labels = [r"$p_1$", r"$p_2$", r"$p_{spam}$"]
    plt.figure(figsize=(12, 8))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylabel(r"$S_i$", labelpad=30, fontdict={'size': word_font,
                                 'color': "black",
                                 'rotation': 0,
                                       # "font" : "Heiti TC"
                                    })


    plt.xlabel("Noise Level", fontdict={'size': word_font,
                                'color': "black",
                                # "font" : "Heiti TC"
                                })
    plt.tick_params(axis="x", colors="black", size=5, labelsize=20)
    plt.tick_params(axis="y", colors="black", size=5, labelsize=word_font)

    for i in range(len(iter_nums)):
        plt.scatter(noise_level, iter_nums[i], marker=markers[i], s=35, c="black",
            label=f"Noise type : {labels[i]}")
        if if_fit:
            line = lines[i]
        coeff = polyfit(noise_level, iter_nums[i], 1)
        poly_y = []
        new_x_list = [min(noise_level), max(noise_level)]

        for one_y in new_x_list:
            poly_y.append(coeff[0] * one_y + coeff[1])
        plt.plot(new_x_list, poly_y, c=colors[i], linewidth=3, label=f"Noise type : {labels[i]}",
             linestyle=line)

    plt.legend(loc="upper left", prop={'size': 15})
    pic_save_path = Path("graphs")
    plt.savefig(pic_save_path / f"iter_num_with_diff_noise.eps", format='eps', bbox_inches="tight")
    plt.savefig(pic_save_path / f"iter_num_with_diff_noise.png", format='png', bbox_inches="tight")
    plt.show()
    plt.close()

def get_near_repeat():
    level = [4, 9, 14, 19]
    for one_level in level:
        print(noise_level[one_level])



def simulation_result_process(error_type):

    if "1" in error_type:
        error_list["2qb"] = -1
        error_list["spam"] = -1
        error_list["1qb"] = begin
    elif "2" in error_type:
        error_list["1qb"] = -1
        error_list["spam"] = -1
        error_list["2qb"] = begin
    elif "spam" in error_type:
        error_list["2qb"] = -1
        error_list["1qb"] = -1
        error_list["spam"] = begin
    else:
        error_lists = get_T1_T2()
        for one_t1_t2 in error_lists:
            print()
    avg_iter_file_nums_list = []
    num_of_less_than_max_iter_list = []
    while error_list[error_type] <= max_:


        repeated_list_path = Path(f"./results_{error_type}")
        repeat_list = [(child_file.name.split('_')[-1]) for child_file in repeated_list_path.iterdir()]
        repeat_times = len(repeat_list)
        iter_file_nums = 0
        for i in repeat_list:
            iter_file_path = Path(f"./results_{error_type}/repeat_{i}/type_{error_type}/"
                                  f"level_{error_list[error_type]}/ansatz_{ansatz_type}/"
                                  f"optimize_{opti_method}/if_back_{str(if_back)}/"
                                  f"p_{plaintext}_k_{key}_c_{ciphertext}")
            iter_num = len(os.listdir(iter_file_path))
            if iter_num == 51:
                repeat_times -= 1
                continue
            iter_file_nums += iter_num - 1

        avg_iter_file_nums = round(iter_file_nums / repeat_times, 4)
        avg_iter_file_nums_list.append(avg_iter_file_nums)
        num_of_less_than_max_iter_list.append(repeat_times)
        print(f"Current noise level is {error_list[error_type]} ; average iter nums is : {avg_iter_file_nums}")
        # print(repeat_list)
        error_list[error_type] += step

    # print(avg_iter_file_nums_list)
    print(num_of_less_than_max_iter_list)


if __name__ == '__main__':
    get_correspond_repeat("1qb")
    # for i in iter_nums:
    #     print(len(i))
