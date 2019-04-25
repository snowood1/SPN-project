from SumProductNets import *
from NetGenerator import NetGenerator, read_from_file, save2file
from Preprocessing import get_data_loader, generate_data, stupid_tfs
import numpy as np
import os
import sys


def main(load=False):
    rv_list = [RV((0, 1), 'x' + str(i)) for i in range(788)]
    test_gen = NetGenerator(788, rv_list, sum_replicate=2, prod_replicate=4)

    # Train the model from scratch
    if not load:
        S = SPN(test_gen.generate(), rv_list)
        S.init_weight()
        mnist_dl = get_data_loader()
        for iters in range(100):
            print("iter " + str(iters+1))
            for f, l in mnist_dl:
                inter = generate_data(f, l)
                S.update_weight(inter, step_size=1e-2)

        # S.print_weight()
        # save2file(os.getcwd() + "/spn_with_weight.obj", S)

    else:
        S = read_from_file(os.getcwd()+ "/spn_with_weight.obj")

    test_dl = get_data_loader(batch_size=1)
    acc_count = 0
    for f, l in test_dl:
        # TODO: Predict the output & compute corresponding metric
        f = generate_data(f, l, feature_only=True)
        l = stupid_tfs(l).numpy()

        tmp = S.map(rv_list[:-4], f)
        res = tmp[-4:]
        if (res == l).all():
            acc_count += 1

    acc = acc_count / test_dl.__len__()
    print(acc)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]) == 1)
    else:
        main()