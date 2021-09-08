from matplotlib import pyplot as plt
from config import style, parameters

def plot_effect_on(of_dataset, of_tag, on_tag):
    for dataset_name, dataset_tag, evaluation in of_dataset:
        x = parameters['list'][:-3]
        y = [test_dict[on_tag] for test_dict in evaluation][:-3]
        plt.plot(x, y, label=dataset_name, color=style[on_tag]['color'], linestyle=style[dataset_tag]['linestyle'])
        plt.scatter(x[0], y[0], s=100, c=style[on_tag]['color'], marker=style[dataset_tag]['marker'])
    plt.xlabel(r'$\mu$ for $l_{fair}=$' + of_tag, fontsize=16)
    plt.ylabel(on_tag, fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig('images/' + of_tag + '--' + on_tag + '.png', bbox_inches="tight", dpi = 400)
    plt.show()