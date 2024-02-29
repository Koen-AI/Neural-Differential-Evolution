import numpy as np
import matplotlib.pyplot as plt
import csv


def scientific2int(sci):
    if 'e' in sci:
        sci_list = sci.split('e')
        sci_float = float(sci_list[0])
        sci_exp = int(sci_list[1][1:])
        return int(sci_float * 10**sci_exp)

    else:
        return int(sci)


def parse_data_vdn(infile, keep_best=False):
    res = []
    best_fitness = -100
    with open(infile, 'r') as file:
        for line in csv.reader(file, delimiter=':'):
            fitness_entry = line[2]
            fitness = float(fitness_entry[1:-10])
            best_fitness = max(fitness, best_fitness)
            episodes_entry = line[0].split(" ")
            episodes_entry = episodes_entry[0]
            episode = int(episodes_entry[1:])
            if keep_best:
                res.append((episode, best_fitness))
            else:
                res.append((episode, fitness))
    return res


def parse_data_ioh(infile, lim=535712):
    res = []
    initilialised = False
    with open(infile, 'r') as file:
        #temp = []
        counter = 0
        last = (-10, 1000)
        score = 0
        last_score = 0
        for line in csv.reader(file, delimiter=' '):
            if counter != 0:
                # print(line[1])
                if last[0] > 0:# and last[0] < lim:
                    #res.append(int(line[0]), -1*float(line[1]))
                    #'''
                    res.append(last)

                last_score = score
                score = -1*float(line[1])
                last = (scientific2int(line[0]), score)
                #'''
            counter += 1
        # for
        res.append((last[0], last_score))
    return res


def parse_data_rng(infile):
    res = []
    initilialised = False
    with open(infile, 'r') as file:
        #temp = []
        counter = 0
        last = (-10, 1000)
        score = 0
        last_score = 0
        for line in csv.reader(file, delimiter=' '):
            # print(line[0], " = ", counter)
            if counter > 2:
                # print(line[1])
                if last[0] >= 0:
                    #res.append(int(line[0]), -1*float(line[1]))
                    #'''
                    res.append(last)

                last_score = score
                score = float(line[1])
                last = (scientific2int(line[0]), score)
                #'''
            counter += 1
        # for
        res.append((last[0], last_score))
    return res


def plot_data(res, indices, plot_color='red', label="ModDE", std_mode=True, boundaries=True, boundry_lines=True):
    lines = len(res)
    #build list of indices:
    x_temp = []
    for i in range(lines):
        for r in res[i]:
            x_temp.append(r[0])

    x = sorted(set(x_temp))

    length = len(x)

    lab_base = "run with seed "
    avg = np.zeros(length)
    std = np.zeros(length)
    min_lst = np.zeros(length)
    max_lst = np.zeros(length)
    temp_list = np.zeros(len(indices))

    prevs = np.zeros(lines)
    temp_steps = np.zeros(lines, dtype=int)
    for seed_index in range(lines):
        prevs[seed_index] = res[seed_index][0][1]

    for step_index in range(length):
        for seed_index in range(lines):
            temp_x = x[step_index]
            temp_res = res[seed_index][temp_steps[seed_index]][0]
            if temp_x == temp_res:
                prevs[seed_index] = res[seed_index][temp_steps[seed_index]][1]
                if temp_steps[seed_index] < len(res[seed_index]) - 1:    # move to the next index if it isn't already the last
                    temp_steps[seed_index] += 1

            temp_list[seed_index] = prevs[seed_index]

        avg[step_index] = np.average(temp_list)
        std[step_index] = np.std(temp_list)
        min_lst[step_index] = min(temp_list)
        max_lst[step_index] = max(temp_list)

        std_min_lst = avg - std
        std_max_lst = avg + std
    # for step_index
    # print("highest used x-value is ", x[-1])
    # plt.plot(x, min_lst, color='red', linestyle='--', label="bounds")
    # plt.plot(x, max_lst, color='red', linestyle='--')
    # plt.plot(x, avg, color='blue', label="average")

    # avg_label = label  # + " average return"
    # plt.step(x, avg, color=plot_color, label=avg_label, where='post')
    plt.step(x, avg, color=plot_color, label=label, where='post')

    if boundaries:
        if std_mode:
            if boundry_lines:
                # bounds_label = label + " avg +/- 1 std"
                plt.step(x, std_min_lst, color=plot_color, linestyle='--', where='post')  # , label=bounds_label)
                plt.step(x, std_max_lst, color=plot_color, linestyle='--', where='post')

            plt.fill_between(x, std_min_lst, std_max_lst, color=plot_color, alpha=0.1, step='post')
        else:
            if boundry_lines:
                # bounds_label = label + " upper and lower bounds"
                plt.step(x, min_lst, color=plot_color, linestyle='--', where='post')  # , label=bounds_label)
                plt.step(x, max_lst, color=plot_color, linestyle='--', where='post')

            plt.fill_between(x, min_lst, max_lst, color=plot_color, alpha=0.1, step='post')

    for i, j in zip(indices, range(len(indices))):
        lab = lab_base + str(i)
        # print(i, ", ", j)
        # print(res[j])
        tot = len(res[j])
        t_x = np.zeros(tot)
        t_y = np.zeros(tot)
        for q in range(tot):
            t_x[q] = res[j][q][0]
            t_y[q] = res[j][q][1]
    return x[-1]


if __name__=="__main__":
    # define the optimal values:
    optimal_y = [86.14, 86.14]  # 86.14000000000001, let's round that to 86.14 for the plot...
    single_best_y = [59.64, 59.64]
    switch_optimal_y = [8.3, 8.3]  # 86.14000000000001, let's round that to 86.14 for the plot...
    switch_single_best_y = [-0.6, -0.6]

    vdn_y = [79.84, 79.84]  # 86.14000000000001, let's round that to 86.14 for the plot...
    optimal_x = [0, 750013000000]
    smode = False
    lab_ref = "30"
    indices = [1, 2, 3, 4, 5]

    modDE_purple = "#ffe623"  # "#440054"  # now it is gold
    RS_teal = "#440054"  # "#208c8c"  # now it is purple
    VDN_gold = "#208c8c"  # "#ffe623"  # now it is teal
    alt1_blue = "#5ac864"  # now it is lime
    alt2_lime = "#3c4887"  # "#5ac864"  # now it is blue

    # /home/koen/Thesis_logs/RNG/checkers_rng

    # Plot the VDN results:
    '''
    lim = []

    vdn_res = []
    vdn_strat = "/VDN/scores/scores"
    vdn_fin = ".vnd"  # lol, this was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    for i in vdn_indices:
        # print(i)                                                                                 750013
        vdn_res.append(
            parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin))  # 750013))
    lim.append(plot_data(vdn_res, vdn_indices, std_mode=True, plot_color="gold", label="VDN reward +/- 1 std", boundaries=True, boundry_lines=False))

    # print(max(lim))
    optimal_y = [86.14, 86.14]  # 86.14000000000001, let's round that to 86.14 for the plot...
    vdn_y = [79.84, 79.84]  # 86.14000000000001, let's round that to 86.14 for the plot...
    optimal_x = [0, 750013000000]

    # plot the base_lines:
    # plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color="gray", label="VDN score reported on checkers")
    
    plt.legend()
    plt.xlim(0, max(lim))  # 1154680)
    # plt.xlim(0, 750000)
    plt.ylim(-5, 90)
    plt.xlabel("episodes")
    plt.ylabel("test reward")
    # plt.title("Average of 5 replications of VDN on magym checkers-v0")
    plt.show()
    # '''

    # Plot the boundary optimization:
    #/home/koen/Thesis_logs/Hyper_params/modDE_no_bias_bounds7/modD1/data_f60_checkers
    # '''
    lim = []
    indices = [1, 2, 3, 4, 5]

    # parse the results with b = 1:
    ioh_res1 = []
    ioh_strat1 = "/Hyper_params/smallbounds/smallbounds"
    ioh_fin1 = "/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res1.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat1 + str(i) + ioh_fin1))
    lim.append(plot_data(ioh_res1, indices, std_mode=smode, plot_color=alt1_blue, label="b=1"))

    # parse the results with b = 7:
    ioh_res7 = []
    ioh_strat7 = "/Hyper_params/modDE_no_bias_bounds7/modD"
    ioh_fin7 = "/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res7.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat7 + str(i) + ioh_fin7))
    lim.append(plot_data(ioh_res7, indices, std_mode=smode, plot_color=modDE_purple, label="b=7"))

    # parse the results with b = 10:
    ioh_res10 = []
    ioh_strat10 = "/Hyper_params/largebounds/largebounds"
    ioh_fin10 = "/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res10.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat10 + str(i) + ioh_fin10))
    lim.append(plot_data(ioh_res10, indices, std_mode=smode, plot_color=alt2_lime, label="b=10"))

    # plot the horizontal lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    # plt.title("Evolution of return plotted for different boundary sizes on Compressed Checkers")
    plt.xlim(0, 1.025 * max(lim))
    plt.ylim(-5, 90)
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()

    # '''

    # Plot the large network against the smallest network
    # '''
    lim = []
    indices = [1, 2, 3, 4, 5]

    ioh_reslong = []
    ioh_stratlong = "/long_14h1_dim406/14h1_checkersF07CR09_"
    ioh_finlong = "/data_f60_checkers/IOHprofiler_f60_DIM812.dat"

    for i in indices:
        ioh_reslong.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_stratlong + str(i) + ioh_finlong))
    lim.append(plot_data(ioh_reslong, indices, std_mode=smode, plot_color=alt2_lime, label="Large Network (406 weights)"))

    # parse the compressed modDE on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDEF07CR09/ModD"
    ioh_fin3 = "F07CR09/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=modDE_purple, label="Small Network (120 weights)"))

    # plot the horizontal lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    # plt.title("Evolution of return on Compressed Checkers plotted for different network architectures")
    plt.xlim(0, 1.025 * max(lim))
    plt.ylim(-5, 90)
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()

    # '''

    # Plot the bias network against the biasless network
    #'''
    lim = []
    indices = [1, 2, 3, 4, 5]

    ioh_reslong = []
    ioh_stratlong = "/Hyper_params/Checkers_bias/Checkers_bias_F07CR09_"
    ioh_finlong = "/data_f60_checkers/IOHprofiler_f60_DIM250.dat"

    for i in indices:
        ioh_reslong.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_stratlong + str(i) + ioh_finlong))
    lim.append(plot_data(ioh_reslong, indices, std_mode=smode, plot_color=alt2_lime, label="Bias Network (125 weights)"))

    # parse the compressed modDE on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDEF07CR09/ModD"
    ioh_fin3 = "F07CR09/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=modDE_purple, label="Bias-less Network (120 weights)"))

    # plot the horizontal lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    # plt.title("Bias or no bias?")
    plt.xlim(0, 1.025 * max(lim))
    plt.ylim(-5, 90)
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()

    # '''

    # Plot RNG vs Best Network vs VDN? for compressed Checkers
    # '''
    lim = []

    vdn_res = []
    vdn_strat = "/VDN/scores/scores"
    vdn_fin = ".vnd"  # This was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    for i in vdn_indices:
        vdn_res.append(parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin))
    # Plotting VDN in this case would not be a fair comparisson...
    # lim.append(plot_data(vdn_res, vdn_indices, std_mode=False, plot_color="gold", label="VDN", boundaries=True, boundry_lines=False))

    # parse the rng run
    rng_res = []
    rng_strat = "/RNG/checkers_rng/checkers_rng_"
    rng_fin = ".bs"
    rng_indices = [1, 2, 3, 4, 5]

    for i in rng_indices:
        rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + rng_strat + str(i) + rng_fin))
    lim.append(plot_data(rng_res, rng_indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the compressed modDE on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDEF07CR09/ModD"
    ioh_fin3 = "F07CR09/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        # print(i)                                                                                 750013
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 1.025*max(lim))
    plt.ylim(-5, 90)
    # plt.title("ModDE and RS on Compressed Checkers")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()

    # '''

    # Plot RNG vs Best Network vs VDN? for vanilla Checkers -> long version
    # '''
    lim = []

    vdn_res = []
    # vdn_strat = "/VDN/scores/scores"
    vdn_strat = "/VDN/scores_VDN2/scores2"
    vdn_fin = ".vnd"  # This was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    for i in vdn_indices:
        vdn_res.append(parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin))
    lim.append(plot_data(vdn_res, vdn_indices, std_mode=smode, plot_color=VDN_gold, label="VDN", boundaries=True, boundry_lines=False))

    # parse the long rng run
    long_rng_res = []
    long_rng_strat = "/rng_no_compression/no_compression_checkers_rng_"
    long_rng_fin = ".bs"
    long_rng_indices = [1, 2, 3, 4, 5]

    for i in long_rng_indices:
        long_rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + long_rng_strat + str(i) + long_rng_fin))
    lim.append(plot_data(long_rng_res, long_rng_indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the decompressed modDE on checkers:
    DC_res = []
    DC_strat = "/no_compression/no_compression_checkersF07CR09_"
    DC_fin = "/data_f60_checkers/IOHprofiler_f60_DIM470.dat"
    DC_indices = [1, 2, 3, 4, 5]

    for i in DC_indices:
        DC_res.append(parse_data_ioh("/home/koen/Thesis_logs" + DC_strat + str(i) + DC_fin))
    lim.append(plot_data(DC_res, DC_indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 1.025*max(lim))
    plt.ylim(-5, 90)
    # plt.title("ModDE, RS, and VDN on Checkers")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot RNG vs Best Network vs VDN? for vanilla Checkers -> short version
    # '''
    lim = []

    vdn_res = []
    # vdn_strat = "/VDN/scores/scores"
    vdn_strat = "/VDN/scores_VDN2/scores2"
    vdn_fin = ".vnd"  # This was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    for i in vdn_indices:
        # print(i)                                                                                 750013
        vdn_res.append(
            parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin))  # 750013))
    lim.append(plot_data(vdn_res, vdn_indices, std_mode=smode, plot_color=VDN_gold, label="VDN", boundaries=True, boundry_lines=False))

    # parse the long rng run
    long_rng_res = []
    long_rng_strat = "/rng_no_compression/no_compression_checkers_rng_"
    long_rng_fin = ".bs"
    long_rng_indices = [1, 2, 3, 4, 5]

    for i in long_rng_indices:
        long_rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + long_rng_strat + str(i) + long_rng_fin))
    lim.append(plot_data(long_rng_res, long_rng_indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the decompressed modDE on checkers:
    DC_res = []
    DC_strat = "/no_compression/no_compression_checkersF07CR09_"
    DC_fin = "/data_f60_checkers/IOHprofiler_f60_DIM470.dat"
    DC_indices = [1, 2, 3, 4, 5]

    for i in DC_indices:
        # print(i)                                                                                 750013
        DC_res.append(parse_data_ioh("/home/koen/Thesis_logs" + DC_strat + str(i) + DC_fin))
    lim.append(plot_data(DC_res, DC_indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 250000)
    plt.ylim(-5, 90)
    # plt.title("ModDE, RS, and VDN on Checkers (close-up)")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot RNG vs Best Network vs VDN? for vanilla Checkers -> short version -> keep best
    # '''
    lim = []

    vdn_res = []
    # vdn_strat = "/VDN/scores/scores"
    vdn_strat = "/VDN/scores_VDN2/scores2"
    vdn_fin = ".vnd"  # This was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    for i in vdn_indices:
        vdn_res.append(parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin, keep_best=True))
    lim.append(plot_data(vdn_res, vdn_indices, std_mode=smode, plot_color=VDN_gold, label="VDN", boundaries=True,
                         boundry_lines=False))

    # parse the long rng run
    long_rng_res = []
    long_rng_strat = "/rng_no_compression/no_compression_checkers_rng_"
    long_rng_fin = ".bs"
    long_rng_indices = [1, 2, 3, 4, 5]

    for i in long_rng_indices:
        long_rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + long_rng_strat + str(i) + long_rng_fin))
    lim.append(plot_data(long_rng_res, long_rng_indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the decompressed modDE on checkers:
    DC_res = []
    DC_strat = "/no_compression/no_compression_checkersF07CR09_"
    DC_fin = "/data_f60_checkers/IOHprofiler_f60_DIM470.dat"
    DC_indices = [1, 2, 3, 4, 5]

    for i in DC_indices:
        # print(i)                                                                                 750013
        DC_res.append(parse_data_ioh("/home/koen/Thesis_logs" + DC_strat + str(i) + DC_fin))
    lim.append(plot_data(DC_res, DC_indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 250000)
    plt.ylim(-5, 90)
    # plt.title("ModDE, RS, and elitist VDN on Checkers (close-up)")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot RNG vs Best Network vs VDN? for Switch2
    # '''
    lim = []

    vdn_res = []
    vdn_strat = "/VDN/switch_scores_vdn/switch_scores"
    vdn_fin = ".vnd"  # This was a typo when I ran the experiment, let's just roll with it...

    # VDN on switch:
    for i in indices:
        vdn_res.append(parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin))
    lim.append(plot_data(vdn_res, indices, std_mode=smode, plot_color=VDN_gold, label="VDN"))

    # parse the long rng run
    switch_rng_res = []
    switch_rng_strat = "/RNG/switch_rng/switch_rng_"
    switch_rng_fin = ".bs"

    for i in indices:
        # print(i)                                                                                 750013
        switch_rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + switch_rng_strat + str(i) + switch_rng_fin))
    lim.append(plot_data(switch_rng_res, indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the decompressed modDE on checkers:
    modDE_res = []
    modDE_strat = "/Hyper_params/switchF07CR09/switchF07CR09_"
    modDE_fin = "/data_f60_switch/IOHprofiler_f60_DIM40.dat"

    for i in indices:
        # print(i)                                                                                 750013
        modDE_res.append(parse_data_ioh("/home/koen/Thesis_logs" + modDE_strat + str(i) + modDE_fin))
    lim.append(plot_data(modDE_res, indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, switch_optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, switch_single_best_y, linestyle=':', color="cyan", label="Single-agent high score")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 20000)
    # plt.title("ModDE, VDN, and RS on Switch2")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot RNG vs Best Network vs VDN? for Switch2 -> keep_best
    # '''
    lim = []

    vdn_res = []
    vdn_strat = "/VDN/switch_scores_vdn/switch_scores"
    vdn_fin = ".vnd"  # lol, this was a typo when I ran the experiment, let's just roll with it...

    # VDN on switch:
    for i in indices:
        vdn_res.append(parse_data_vdn("/home/koen/Thesis_logs" + vdn_strat + str(i) + vdn_fin, keep_best=True))
    lim.append(plot_data(vdn_res, indices, std_mode=smode, plot_color=VDN_gold, label="VDN"))

    # parse the long rng run
    # /home/koen/Thesis_logs/rng_no_compression/no_compression_checkers_rng_1.bs
    switch_rng_res = []
    switch_rng_strat = "/RNG/switch_rng/switch_rng_"
    switch_rng_fin = ".bs"

    for i in indices:
        switch_rng_res.append(parse_data_rng("/home/koen/Thesis_logs" + switch_rng_strat + str(i) + switch_rng_fin))
    lim.append(plot_data(switch_rng_res, indices, std_mode=smode, plot_color=RS_teal, label="RS"))

    # parse the decompressed modDE on checkers:
    modDE_res = []
    modDE_strat = "/Hyper_params/switchF07CR09/switchF07CR09_"
    modDE_fin = "/data_f60_switch/IOHprofiler_f60_DIM40.dat"

    for i in indices:
        # print(i)                                                                                 750013
        modDE_res.append(parse_data_ioh("/home/koen/Thesis_logs" + modDE_strat + str(i) + modDE_fin))
    lim.append(plot_data(modDE_res, indices, std_mode=smode, plot_color=modDE_purple, label="modDE"))

    # plot the base_lines:
    plt.plot(optimal_x, switch_optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, switch_single_best_y, linestyle=':', color="cyan", label="Single-agent high score")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 20000)
    # plt.title("modDE, RS, and Elitist VDN on Switch2")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot modDE civil war on (compressed) checkers:
    # '''
    lim = []

    vdn_res = []
    vdn_strat = "/VDN/scores/scores"
    vdn_fin = ".vnd"  # lol, this was a typo when I ran the experiment, let's just roll with it...
    vdn_indices = [1, 2, 3, 4, 5]

    # parse the compressed modDE on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDEF07CR09/ModD"
    ioh_fin3 = "F07CR09/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=alt1_blue, label="Compressed Checkers"))
    # parse the decompressed modDE on checkers:
    DC_res = []
    DC_strat = "/no_compression/no_compression_checkersF07CR09_"
    DC_fin = "/data_f60_checkers/IOHprofiler_f60_DIM470.dat"
    DC_indices = [1, 2, 3, 4, 5]

    for i in DC_indices:
        # print(i)                                                                                 750013
        DC_res.append(parse_data_ioh("/home/koen/Thesis_logs" + DC_strat + str(i) + DC_fin))
    lim.append(plot_data(DC_res, DC_indices, std_mode=smode, plot_color=alt2_lime, label="Checkers"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    # plt.xlim(0, 100000)
    plt.xlim(0, 1.025 * max(lim))  # +100000 )# 1154680)
    plt.ylim(-5, 90)
    # plt.title("Checkers vs Compressed Checkers")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

    # Plot hyperparams on compressed checkers:
    # '''
    lim = []

    # parse the modDE-7-9 on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDEF07CR09/ModD"
    ioh_fin3 = "F07CR09/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=modDE_purple, label="F=0.7, CR=0.9"))

    # parse ModDE-9-10 on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/ModDEF09CR10/ModD"
    ioh_fin3 = "F09CR10/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=alt1_blue, label="F=0.9, CR=1.0"))

    # parse modDE-5-5 on checkers:
    ioh_res3 = []
    ioh_strat3 = "/Hyper_params/modDE_no_bias_bounds7/modD"
    ioh_fin3 = "/data_f60_checkers/IOHprofiler_f60_DIM240.dat"

    for i in indices:
        # print(i)                                                                                 750013
        ioh_res3.append(parse_data_ioh("/home/koen/Thesis_logs" + ioh_strat3 + str(i) + ioh_fin3))
    lim.append(plot_data(ioh_res3, indices, std_mode=smode, plot_color=alt2_lime, label="F=0.5, CR=0.5"))

    # plot the base_lines:
    plt.plot(optimal_x, optimal_y, linestyle=':', color="black", label="Theoretically optimal score")
    plt.plot(optimal_x, single_best_y, linestyle=':', color="cyan", label="Single-agent high score")
    plt.plot(optimal_x, vdn_y, linestyle=':', color=VDN_gold, label="VDN score reported by [" + lab_ref + "]")

    # prepare and show the plots:
    plt.legend(loc="lower right")
    plt.xlim(0, 1.025 * max(lim))
    plt.ylim(-5, 90)
    # plt.title("Hyper parameter sweep")
    plt.xlabel("fitness evaluations")
    plt.ylabel("return")
    plt.show()
    # '''

