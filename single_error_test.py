import os
from run_VQAA import *
from pathlib import Path
from time import time

if __name__ == '__main__':
    debug_mode = 1
    if debug_mode:
        error_type = "2qb"
        begin = 0.00
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
        repeat = "0, 10"
        end_prob = 0.5
    else:

        error_type = input("Input Error type : ")
        begin = float(input("Input begin : "))
        step = float(input("Input step : "))
        max_ = float(input("Input max : "))
        gpu = int(input("If use GPU: "))
        ansatz_type = input("Input ansatz type (rx,ry or rz) : ")
        opti_method = input("Input optimization method (nm or gd) : ")
        if_back = int(input("If Backward : "))
        key = input("Input key (0 as default) : ")
        plaintext = input("Input plaintext (0 as default) : ")
        ciphertext = input("Input ciphertext (0 as default) : ")
        max_iter = input("Max interation : ")
        repeat = input("repeated : ")


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

    STEP = step
    MAX_ERROR_RATE = max_
    # 对于每一类噪声都要实验
    while error_list[error_type] <= MAX_ERROR_RATE:

        ## 对于每一个噪声大小，重复repeated次
        for i in range(int(repeat.split(',')[0]), int(repeat.split(',')[1])):

            error_level = str(error_list[error_type])
            # 这里需要重新写thermal error 的情况

            new_path = Path(f"results/repeat_{i}/type_{error_type}/level_{error_level}/ansatz_{ansatz_type}/optimize_{opti_method}/if_back_{str(if_back)}/p_{plaintext}_k_{key}_c_{ciphertext}/")
            if not new_path.exists():
                new_path.mkdir(exist_ok=True, parents=True)

                os.chdir(new_path)
                print(f"Error type : {error_type} . Error level : {error_list[error_type]} . Current repeat : {i}")
                VQAA(ansatz=ansatz_type, if_back=if_back, optimization=opti_method, max_iter=max_iter, plain=plaintext,
                                key=key, cipher=ciphertext, if_GPU=gpu, error_type=error_type, threshold=0.1, end_prob=end_prob)
                os.chdir(Path("./../../../../../../../.."))
            else:
                continue
        error_list[error_type] += STEP


