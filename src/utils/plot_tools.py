from matplotlib import pyplot as plt


def plot_img_and_mask(img, mask):
    class_cnt = mask.max() + 1
    figure, ax = plt.subplots(1, class_cnt + 1)
    ax[0].set_title("Input image")
    ax[1].imshow(img)
    for i in class_cnt:
        ax[i + 1].set_title(f"Mask (calss {i + 1})")
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
