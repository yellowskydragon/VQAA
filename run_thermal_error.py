import os
import pandas

from run_VQAA import *
from pathlib import Path
from time import time
from qiskit.providers.aer.noise import NoiseModel


def get_T1_T2():
    colnames = ["Qubit", "Frequency", "T1", "T2", "ReadoutError", "SQError", "TQError", "Date"]
    path = Path("ibmq_16_melbourne_calibrations.csv")
    data = pandas.read_csv(path, names=colnames)
    '''Returns the thermal relaxation time T1 and the qubit dephasing time T2, as given by IBMQ.'''
    t1er = data.T1.tolist()
    del t1er[0]
    t2er = data.T2.tolist()
    del t2er[0]

    for i in range(0, len(t1er)):
        t1er[i] = float(t1er[i] + "e3")
        t2er[i] = float(t2er[i] + "e3")
    T1s = np.array(t1er)
    T2s = np.array(t2er)
    return [[t1, t2] for t1, t2 in zip(T1s, T2s)]



def run_thermal_error():
    gpu = 0
    ansatz_type = "rx"
    opti_method = "gd"
    if_back = 0
    key = "11001010"
    plaintext = "10110010"
    ciphertext = "10000111"
    max_iter = 50
    repeat = "0, 1"
    end_prob = 0.5
    for t1_t2 in get_T1_T2():
        error_list["thermal"] = [t1_t2[0], t1_t2[1]]
        print(error_list['thermal'])
        for i in range(int(repeat.split(',')[0]), int(repeat.split(',')[1])):
            new_path = Path(f"results/repeat_{i}/type_thermal/level_{error_list['thermal']}/ansatz_{ansatz_type}/optimize_{opti_method}/if_back_{str(if_back)}/p_{plaintext}_k_{key}_c_{ciphertext}/")
            if not new_path.exists():
                new_path.mkdir(exist_ok=True, parents=True)
                os.chdir(new_path)
                print(f"Error type : Error . Error level : {error_list['thermal']} . Current repeat : {i}")
                VQAA(ansatz=ansatz_type, if_back=if_back, optimization=opti_method, max_iter=max_iter, plain=plaintext,
                 key=key, cipher=ciphertext, if_GPU=gpu, error_type="thermal", threshold=0.1, end_prob=end_prob)
                os.chdir(Path("./../../../../../../../.."))
            else:
                continue
if __name__ == '__main__':
    run_thermal_error()