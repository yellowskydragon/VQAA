import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

pauli_Z = np.array([[1, 0], [0, -1]])


def create_graph():
    G = nx.random_graphs.random_regular_graph(DEGREE, AES_KEY_LENGTH)

    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")

    plt.savefig("g.png", format="png")
    return G


def get_corresponding_theta_ij(measure_result, i, j):

    return np.array([0, 0, 0, 0])


def get_corresponding_theta_i(measure_result, i):

    return np.array([0, 0])


def calculate_expected(g: nx.Graph, measure_result):
    ham = 0

    edge_list = g.edges
    # print(g.edges)
    for one_edge in edge_list:
        V_i, V_j = one_edge[0], one_edge[1]
        if Ciphertext[V_i] == Ciphertext[V_j]:
            omega_ij = -1
        else:
            omega_ij = 1
        beta_ij = get_corresponding_theta_ij(measure_result=measure_result, i=V_i, j=V_j)
        ham += omega_ij * (beta_ij) * np.kron(pauli_Z, pauli_Z) * beta_ij

    for i in range(AES_KEY_LENGTH):
        if Ciphertext[i] == 0:
            t_i = -0.5
        else:
            t_i = 0.5
        beta_i = get_corresponding_theta_i(measure_result, i=i)
        ham += t_i * beta_i * pauli_Z * beta_i.transpose()

    return ham


def optimization(meathod, value_of_cost_func):
    """
    Does not return anything, change THETA_LIST instead
    :param ham:
    :return:
    """


def calculate_data_space_state_vector(ansatz, if_back_control ):
    qc = generate_circuit_with_state_vector(ansatz_type=ansatz, if_back_control=0, plain_text=Plaintext)
    backend = Aer.get_backend('statevector_simulator')

    print("Corresponding Theta List : ", THETA_LIST)
    start_time = time.perf_counter()
    tqc = transpile(qc, backend)
    job = backend.run(tqc)
    result = job.result()
    end_time = time.perf_counter()
    print("Calculate state vector time is : ", round(end_time - start_time))
    state_vector = result.data()["final"]

    # print(state_vector)

    def calculate_data_space_vec(vector):

        return []

    return calculate_data_space_vec(state_vector)

def arr2str(vec: np.array):
    return ','.join(str(i) for i in vec)




if __name__ == '__main__':

    draw_prob_and_cost_with_iter()