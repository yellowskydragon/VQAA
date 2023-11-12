import matplotlib.pyplot as plt
import numpy as np

from run_VQAA import *
from matplotlib import *

if __name__ == '__main__':
    # n = 2
    # key_space = QuantumRegister(n)
    # data_space  = QuantumRegister(n)
    #
    # qc = QuantumCircuit(key_space, data_space)
    # qc.x(data_space[1])
    # qc.h(key_space)
    # qc.save_statevector(label="final")
    # backend = Aer.get_backend("statevector_simulator")
    # tqc = transpile(qc, backend)
    # job = backend.run(tqc)
    # state_vector = job.result().data()["final"]
    # qc.draw(output="mpl")
    # matplotlib.pyplot.savefig("1.png")
    #
    # omega_i, omega_j = 0, 1
    # print(state_vector)
    # prob = [0] * 4
    # normalized_v = state_vector / np.linalg.norm(state_vector)
    # print(normalized_v)
    # for i in range(2 ** (2 * n)):
    #     bin_index = bin(i)[2:].rjust(2 * n, '0')
    #     # print(i)
    #     # print(bin_index)
    #     if bin_index[omega_i] == "0" and bin_index[omega_j] == "0":
    #         prob[0] += abs(normalized_v[i]) ** 2
    #     elif bin_index[omega_i] == "0" and bin_index[omega_j] == "1":
    #         prob[1] += abs(normalized_v[i]) ** 2
    #     elif bin_index[omega_i] == "1" and bin_index[omega_j] == "0":
    #         prob[2] += abs(normalized_v[i]) ** 2
    #     elif bin_index[omega_i] == "1" and bin_index[omega_j] == "1":
    #         prob[3] += abs(normalized_v[i]) ** 2
    #     print("----")
    # print(prob)
    # qc = generate_circuit_with_state_vector("rx",0,"001",[0]*3)
    # print(run_one_sim(qc,0,"1qb"))
    # def f(x, y):
    #     return np.sin(np.sqrt(x ** 2 + y ** 2))
    #
    #
    # x = np.linspace(-6, 0, 4)
    # y = np.linspace(0, 6, 3
    #                 )
    #
    # X, Y = np.meshgrid(x, y)
    # print(X)
    # print(Y)

    def draw_prob_and_cost_with_iter():
        word_font = 30
        iter = [0, 1, 2, 3, 4, 5, 6, 7]
        prob = [0.00829, 0.050439, 0.09862933, 0.160, 0.23472797, 0.340, 0.42567341, 0.63920]
        cost = [-1.1940, -4.4827852, -6.69554, -8.667392, -10.0, -11.07671, -11.73282, -13.319778181]

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        plt.grid(True, linestyle='--', alpha=0.5)
        ax1.set_ylabel('Probability', fontdict={'size': word_font,
                                                   'color': "black",
                                                   'rotation': 90,
                                                   # "font" : "Heiti TC"
                                                   })
        # ax1.set_title("Prob and cost with iter")
        ax1.plot(iter, prob, c="blue", linewidth=3, label=f"Probability", linestyle="-")
        ax1.set_xlabel(f"Iteration", fontdict={'size': word_font,
                                             'color': "black",
                                             })
        ax1.tick_params(axis="x", colors="black", size=5, labelsize=20)
        ax1.tick_params(axis="y", colors="black", size=5, labelsize=20)

        plt.legend(loc="upper right", fontsize=word_font)
        ax2 = ax1.twinx()
        ax2.plot(iter, cost, c="red", linewidth=3, label=f"Cost", linestyle="-")
        ax2.set_ylabel(r'Cost', fontdict={'size': word_font,
                                          'color': "black",
                                          'rotation': 90,
                                         })
        ax2.tick_params(axis="y", colors="black", size=5, labelsize=20)
        plt.legend(loc="upper left", fontsize=word_font)

        plt.savefig("./1.png", format="png", bbox_inches="tight")
        plt.show()

    # from matplotlib.font_manager import fontManager
    # print([f.name for f in fontManager.ttflist])
    # print(plt.rcParams['font.family'])
    plt.rcParams['font.family'] = 'Times New Roman'
    draw_prob_and_cost_with_iter()