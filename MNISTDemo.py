from SumProductNets import *
from NetGenerator import NetGenerator, read_from_file, save2file
from Preprocessing import get_data_loader, generate_data
import numpy as np
import os
import sys


def main(load=False):
    rv_list = [RV((0, 1), 'x' + str(i)) for i in range(788)]
    test_gen = NetGenerator(788, rv_list, sum_replicate=2, prod_replicate=4)

    # Train the model from scratch
    if not load:
        S = SPN(test_gen.generate(), rv_list)
        mnist_dl = get_data_loader()
        for f, l in mnist_dl:
            inter = generate_data(f, l)
            S.train(inter, iterations=100, step_size=5)

        # S.print_weight()
        save2file(os.getcwd() + "/spn_with_weight.obj", S)

    else:
        S = read_from_file(os.getcwd().join("/spn_with_weight.obj"))

    test_dl = get_data_loader()
    for f, l in test_dl:
        # TODO: Predict the output & compute corresponding metric
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1] == 1)
    else:
        main()