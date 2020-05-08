from matplotlib import pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import pickle as pkl

def load(tol):
    seed = 1
    tol = float(tol)
    methods = ["lm", "aalm", "raalm", "raalm0"]
    loaded_dicts = []
    for m in methods:
        with open("{}_{}_tol{}.pkl".format(m, seed, tol), "rb") as f:
            loaded_dicts.append(pkl.load(f))
    return loaded_dicts

method_names = ["ALM", "A-ALM", "RA-ALM, $\\eta = 1 - \sqrt{k}/2$", "RA-ALM, $\\eta = 1 - \sqrt{k}$"]
for tol in [0.0, 0.5, 1.0]:
    dicts = load(tol)
    plt.figure(dpi=300, figsize=[4,3])
    for i in range(len(method_names)):
        plt.semilogy(dicts[i]["ds"], label=method_names[i])
    plt.legend()
    #plt.title("noise level {}".format(tol))
    plt.xlabel("iteration")
    plt.ylabel("$\|x - x^*\|_2$")
    plt.ylim(1e-4, 1e4)
    #plt.show()
    plt.savefig("dual_gradient_tol{}.pdf".format(tol), bbox_inches='tight')