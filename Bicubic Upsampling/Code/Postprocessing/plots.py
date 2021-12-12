import random

import matplotlib.pyplot as plt


# TO-D0 Make a Plot with Bicubic and SRCNN PSNR Evaluation Values with each Epoch
def evaluation_plot():
    epoch_eval1 = {
        'psnr_srcnn': [],
        'psnr_bicubic': [],
        'ssim_srcnn': [],
        'ssim_bicubic': []
    }

    # Extracting SSIM and PSNR Scores from txt file
    with open('epoch_20.txt', 'rt') as eval_file:
        for line in eval_file:
            word_list = line.split()
            if word_list[0] == 'PSNR':
                epoch_eval1['psnr_bicubic'].append(float(word_list[-5]))
                epoch_eval1['psnr_srcnn'].append(float(word_list[-1]))
            elif word_list[0] == 'SSIM':
                epoch_eval1['ssim_bicubic'].append(float(word_list[-5]))
                epoch_eval1['ssim_srcnn'].append(float(word_list[-1]))

    # Global Epoch Value
    num_epoch = len(epoch_eval1['psnr_srcnn'])
    # Figure and Subplots
    eval_fig, (ax_psnr, ax_ssim) = plt.subplots(2, 1, figsize=(10, 15))
    eval_fig.suptitle('Per Epoch Evaluation Metrics Comparisons', fontsize=18)
    eval_fig.tight_layout(pad=8, h_pad=10)

    # SSIM Subplot
    ax_psnr.set_title("PSNR", fontsize=16)

    # label y
    ax_psnr.set_ylabel("PSNR (dB)", fontsize=12)

    # Plotting SRCNN
    ax_psnr.plot(range(num_epoch), epoch_eval1['psnr_srcnn'], "b", label="SRCNN")

    # Plotting Bicubic
    ax_psnr.plot(range(num_epoch), epoch_eval1['psnr_bicubic'], "r", label="Bicubic")

    # Setting up axis scale - xmin, xmax, ymin, ymax
    ax_psnr.axis([0, num_epoch, 0, 40])

    # SSIM Subplot
    ax_ssim.set_title("SSIM", fontsize=16)

    # label x
    ax_ssim.set_xlabel("Number of Epochs", fontsize=12)

    # label y
    ax_ssim.set_ylabel("SSIM", fontsize=12)

    # Plotting SRCNN
    ax_ssim.plot(range(num_epoch), epoch_eval1['ssim_srcnn'], "b", label="SRCNN")

    # Plotting Bicubic
    ax_ssim.plot(range(num_epoch), epoch_eval1['ssim_bicubic'], "r", label="Bicubic")

    # Setting up axis scale - xmin, xmax, ymin, ymax
    ax_ssim.axis([0, num_epoch, 0, 1])

    # Figure Legends
    handles, labels = ax_ssim.get_legend_handles_labels()
    eval_fig.legend(handles, labels, loc='center', ncol=2)

    # Display all open figures
    plt.show()


def loss_plot():
    epoch_loss1 = {
        'train_loss': [],
        'val_loss': [],
    }

    # Extracting Loss and Val Loss Scores from txt file
    with open('epoch_20.txt', 'rt') as eval_file:
        for line in eval_file:
            word_list = line.split()
            if word_list[0][0] == '[':
                epoch_loss1['train_loss'].append(float(word_list[-1]))
            elif len(word_list) == 5 and word_list[2] == 'validation':
                epoch_loss1['val_loss'].append(float(word_list[-1]))

    # Global Epoch Value
    num_epoch = len(epoch_loss1['val_loss'])
    num_loss = len(epoch_loss1['train_loss'])
    # Figure and Subplots
    eval_fig, (ax_loss, ax_val_loss) = plt.subplots(1, 2, figsize=(20, 10))
    eval_fig.suptitle('Loss Plots', fontsize=18)
    eval_fig.tight_layout(pad=8, h_pad=10)

    # SSIM Subplot
    ax_loss.set_title("Training Loss", fontsize=16)

    ax_loss.set_xlabel("Number of Batches", fontsize=12)

    # label y
    ax_loss.set_ylabel("Loss", fontsize=12)

    # Plotting Training Loss
    ax_loss.plot(range(num_loss), epoch_loss1['train_loss'], "b")

    # Setting up axis scale - xmin, xmax, ymin, ymax
    ax_loss.axis([0, num_loss, 0, 1.2])

    # SSIM Subplot
    ax_val_loss.set_title("Validation Loss", fontsize=16)

    # label x
    ax_val_loss.set_xlabel("Number of Epochs", fontsize=12)

    # label y
    ax_val_loss.set_ylabel("Loss", fontsize=12)

    # Plotting SRCNN
    ax_val_loss.plot(range(num_epoch), epoch_loss1['val_loss'], "b")

    # Setting up axis scale - xmin, xmax, ymin, ymax
    ax_val_loss.axis([0, num_epoch, 0, 1.2])

    # Display all open figures
    plt.show()


def upsampling_plot(metric_results):
    num_sub_images = len(list(metric_results[0].values())[0])
    # Figure and Subplots
    eval_fig, axes = plt.subplots(len(metric_results[0]), 1, figsize=(1.5 * num_sub_images, 5 * len(metric_results[0])))
    eval_fig.suptitle('Evaluation Metric Results', fontsize=20)
    eval_fig.tight_layout(pad=8, h_pad=10)
    metric_sample = random.sample(metric_results, len(metric_results) if len(metric_results) < 3 else 3)
    # Subplots
    '''

    Root mean square error (RMSE),
    Peak signal-to-noise ratio (PSNR),
    Structural Similarity Index (SSIM),
    Feature-based similarity index (FSIM),
    Information theoretic-based Statistic Similarity Measure (ISSM),
    Signal to reconstruction error ratio (SRE),
    Spectral angle mapper (SAM), and
    Universal image quality index (UIQ)

    '''
    plot_parameters = {
        'colours': ['b', 'r', 'g'],
        'psnr': ["PSNR (Peak Signal to Noise Ratio)", (30, 45)],
        'ssim': ["SSIM (Structural Similarity Index Measure)", (0.4, 1)],
        'sre': ["SRE (Signal to Reconstruction Error Ratio)", (40, 60)],
        'sam': ["SAM (Spectral Angle Mapper)", (75, 100)],
        'rmse': ["RMSE (Root Mean Square Error)", (0, 0.03)],
        'cos': ["Cosine Similarity", (1e-09, 9e-08)]
    }

    for i in range(len(metric_sample)):
        for index, (key, value) in enumerate(metric_sample[i].items(), 0):
            axes[index].set_title(f'{plot_parameters[key][0]}', fontsize=16)
            # label y
            axes[index].set_ylabel(f"{key.upper()}", fontsize=12)
            # label x
            axes[index].set_xlabel("Upsampled Sub Image Number", fontsize=12)
            # Plotting Original Image - Upscaled Sub Image Values per metric
            axes[index].plot(range(1, num_sub_images + 1), value, plot_parameters['colours'][i],
                             label=f"HR Image {i + 1}")
            # Setting up axis scale - xmin, xmax, ymin, ymax
            axes[index].axis([1, num_sub_images, plot_parameters[key][1][0], plot_parameters[key][1][1]])
            # Subplot Legends
            axes[index].legend(loc='upper right')

    # Display all open figures
    plt.show()
