from SumProductNets import *
from NetGenerator import NetGenerator, read_from_file
from Demo.Preprocessing import get_data_loader, generate_data, stupid_tfs
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(images):
    images = np.reshape(images, [-1, 28 * 28])
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        img = img.reshape(28, 28)
        # img = np.transpose(img, [1, 2, 0])
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.savefig('./test.png')


def main(load=False):
    rv_list = [RV((0, 1), 'x' + str(i)) for i in range(28 * 28)]
    test_gen = NetGenerator(28 * 28, 0, rv_list, sum_replicate=2, prod_replicate=5, cut_limit=(5, 8))

    # Train the model from scratch
    if not load:
        S = SPN(test_gen.generate('forest'), rv_list)
        S.init_weight()
        mnist_dl = get_data_loader(batch_size=1000)
        for f, _ in mnist_dl:
            inter = generate_data(f, None, feature_only=True)
            S.train(inter, iterations=100, step_size=.1)
        # for iters in range(100):
        #     print("iter " + str(iters+1))
        #     for f, _ in mnist_dl:
        #         inter = generate_data(f, None, feature_only=True)
        #         S.update_weight(inter, step_size=1e-3)

        # S.print_weight()
        # save2file(os.getcwd() + "/spn_with_weight.obj", S)

    else:
        S = read_from_file(os.getcwd()+ "/spn_with_weight.obj")

    test_dl = get_data_loader(batch_size=1, data_size=16)
    tmp_out = []
    for f, _ in test_dl:
        f = generate_data(f, _, feature_only=True).tolist()
        output_img = S.map(rv_list[-392:], f[-392:])
        # print(output_img)
        output_img = np.asarray(output_img) * 255
        tmp_out.append(output_img)
        # show_images(output_img)
    show_images(np.asarray(tmp_out))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]) == 1)
    else:
        main()