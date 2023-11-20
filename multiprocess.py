from single_error_test import *
import math
import datetime
# import multiprocessing as mp
from multiprocessing import Pool

if __name__ == '__main__':
    # num_cores = int(mp.cpu_count())
    # print("本地计算机有: " + str(num_cores) + " 核心")
    num_core = 5
    total_times = 500
    num_each_core_word = total_times // num_core

    para_list = [(a * num_each_core_word, (a + 1) * num_each_core_word) for a in range(0, num_core)]
    print(para_list)

    with Pool(num_core) as p:
        p.map(one_thread_vqaa, para_list)

    # pool.apply_async(one_thread_vqaa, args=(a, b)) for [a, b] in para_list
