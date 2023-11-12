def gd(graph, ansatz, if_back, plain, cipher, if_GPU, error_type, expectation, thetas, threshold, lr=0.72, xerr=-9):
    times = 0
    x0 = thetas
    prev_cost = 114514
    length = len(x0)
    cost = 0
    for ii in range(1024):
        print("***"*20)
        print(f"This is iteration {ii} from one gd!")
        print(f"current theta list is {x0}")
        if ii == 0:
            cost = expectation
        else:
            qc = generate_circuit_with_state_vector(ansatz_type=ansatz, if_back_control=if_back,
                                                    plain_text=plain, theta_list=x0)
            state_vector = run_one_sim(qc=qc, if_gpu=if_GPU, error_type=error_type)
            cost = cal_expect_of_ham(g=graph, state_vector=state_vector, cipher=cipher)
        times += 1
        if cost < xerr * 100:
            break
        Gd = [0] * length
        for i in range(length):
            x = [t for t in x0]
            x[i] += 0.01
            qc = generate_circuit_with_state_vector(ansatz_type=ansatz, if_back_control=if_back,
                                                    plain_text=plain, theta_list=x)
            state_vector = run_one_sim(qc=qc, if_gpu=if_GPU, error_type=error_type)
            cost_ = cal_expect_of_ham(g=graph, state_vector=state_vector, cipher=cipher)
            print(f"changing the {i}th of parameter of theta list and new cost is {cost_}")
            times += 1
            Gd[i] = (cost_ - cost) / 0.01
        if (prev_cost - cost_) <= threshold:
            break
        else:
            prev_cost = cost_
        r0 = random.uniform(0, 1)
        print(f"current gradient is : {Gd}")

        for j in range(length):
            x0[j] = x0[j] - (lr / abs(cost) + math.log10(times) / times * r0) * Gd[j]

        def abs_sum(some_list):
            return sum([abs(tmp) for tmp in some_list])

        t_sum = abs_sum(Gd)
        if t_sum < 0.8:
            x0 = [random.uniform(0, math.pi / 2) for _ in range(length)]
        print(f"After one iter of gd, |Gd| is {t_sum}")
    return cost, x0
