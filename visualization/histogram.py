import pandas as pd

def barChart(path,pathToSave):
    final_csv = pd.read_csv(path)
    ks_x = []
    i = 0
    while i < len(final_csv["crypto_name"]):
        ks_x.append(str(final_csv["crypto_name"][i]))
        i += 1

    ind = np.arange(len(ks_x))  # the x locations for the groups
    ind = ind * 1.7
    width = 0.2  # the width of the bars

    fig = plt.figure(figsize=(20, 10), dpi=70)
    ax = fig.add_subplot(111)
    # ax.set_title('Cluster '+ str(clusterid)+": Average RMSE configuration oriented")

    yvals = []
    for avgrmse in final_csv["average_rmse_norm"].values:
        yvals.append(avgrmse)
    rects1 = ax.bar(ind, yvals, width, color='green')
    # zvals = [0.0600, 0.0522]
    # rects2 = ax.bar(ind + width, zvals, width, color='orange')

    ax.set_ylabel('Average (RMSE)')
    ax.set_xlabel('Crypto names')
    ax.set_xticks(ind)
    ax.set_xticklabels(ks_x, rotation = 90, ha="right")

    # ax.legend((rects1[0],rects2[0]), (exp_name[0],exp_name[1]))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.grid(linewidth=0.2, color='black')
    plt.savefig(pathToSave + "/cluster.png")
    # plt.show()"""