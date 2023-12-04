from single_error_test import *
import math
import datetime
# import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import cpu_count
if __name__ == '__main__':
    # num_cores = int(cpu_count())
    # print("本地计算机有: " + str(num_cores) + " 核心")
    error_type = "1qb"
    num_core = 10



    repeat_list = list(range(0, 500, 1))
    total_times = len(repeat_list)

    step = int(total_times / num_core)

    para_list = [(repeat_list[i], repeat_list[i + step - 1], error_type) for i in range(0, total_times, step)]

    print(para_list)

    #
    # with Pool(num_core) as p:
    #     p.map(one_thread_vqaa, para_list)


