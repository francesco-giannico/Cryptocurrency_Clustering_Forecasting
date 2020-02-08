import matplotlib.pyplot as plt
import numpy as np

x = ["Single Target", "Single Target with Indicators", "Multi Target", "Multi Target with Indicators", "Simple", "VAR"]

exp_name = ["Top 8", "Top 5"]
ind = np.arange(len(x))  # the x locations for the groups
ind = ind * 2.3
width = 0.45  # the width of the bars

fig = plt.figure(figsize=(12, 7), dpi=70)
ax = fig.add_subplot(111)

yvals = [0.0504, 0.0512, 0.0482, 0.0465, 0.0764, 0.0645]
rects1 = ax.bar(ind, yvals, width, color='b')
zvals = [0.0600, 0.0522, 0.0368, 0.0596, 0.0912, 0.0768]
rects2 = ax.bar(ind + width, zvals, width, color='orange')


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1 * height,
                "%.6f" % float(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Average (RMSE)')
ax.set_xticks(ind + width)
ax.set_xticklabels(x)
ax.legend((rects1[0], rects2[0]), (exp_name[0], exp_name[1]))
ax.set_axisbelow(True)
plt.tight_layout()
plt.grid(linewidth=0.2, color='black')
plt.savefig("../total.png")
# plt.show()
