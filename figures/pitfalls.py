import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tueplots import figsizes, fonts, fontsizes


def rbf_kernel(x, xp, lengthscale, variance):
    """Compute the RBF kernel between x and xp."""
    return variance * np.exp(-0.5 * ((x - xp) ** 2) / lengthscale**2)


def f(x):
    """Scalar function to be optimized."""
    return 0.3 * np.exp(-(((x - 0.2) / 0.1) ** 2)) + 0.7 * np.exp(-(((x - 0.7) / 0.1) ** 2))


def plot_gp_and_acquisition(X_sample, Y_sample, lengthscale, kernel_amplitude, noise, subplot_base, title_suffix):
    # Points to predict
    X_plot = np.linspace(0, 1, 250)

    # Kernel matrix for the samples
    K = rbf_kernel(X_sample[:, np.newaxis], X_sample, lengthscale, kernel_amplitude)
    K += noise * np.eye(K.shape[0])

    # Kernels between the plot points and the sample points
    K_s = rbf_kernel(X_plot[:, np.newaxis], X_sample, lengthscale, kernel_amplitude)

    # Kernel of the plot points
    K_ss = rbf_kernel(X_plot[:, np.newaxis], X_plot, lengthscale, kernel_amplitude)

    # GP posterior distribution parameters
    K_inv = np.linalg.inv(K)
    mu = K_s @ K_inv @ Y_sample
    cov = K_ss - K_s @ K_inv @ K_s.T
    std_dev = np.sqrt(np.diag(cov))

    # Calculate the Probability of Improvement
    best_y = np.max(Y_sample) + np.sqrt(noise)
    xi = 0.0  # exploration parameter
    z = (mu - best_y + xi) / std_dev
    PI = norm.cdf(z)

    # Plotting the GP regression
    plt.subplot(2, 2, subplot_base)
    plt.plot(X_plot, f(X_plot), "r:", label="True function")
    plt.plot(X_sample, Y_sample, "ko", markersize=5, label="Observations")
    plt.plot(X_plot, mu, "b-", label="GP mean")
    plt.fill_between(
        X_plot, mu - 1.96 * std_dev, mu + 1.96 * std_dev, color="blue", alpha=0.2, label="95% confidence interval"
    )
    plt.title(f"GP Posterior ({title_suffix})")

    # Plotting the Probability of Improvement
    plt.subplot(2, 2, subplot_base + 2)
    plt.plot(X_plot, PI, "g-", label="Probability of Improvement")
    plt.fill_between(X_plot, 0, PI, color="green", alpha=0.2)
    plt.title("PI acquisition function")


if __name__ == "__main__":
    # Set up the plots
    plt.rcParams.update(fontsizes.icml2022())
    plt.rcParams.update(fonts.icml2022())
    plt.rcParams.update(figsizes.icml2022_half())
    plt.rcParams.update(  # Use Times New Roman instead of Times
        {
            "font.serif": ["Times New Roman"],
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )

    # Sample data
    X_sample = np.array([0.18, 0.25])
    Y_sample = f(X_sample)
    noise = 0.05**2

    # Plot 1: just plot the data
    X_plot = np.linspace(0, 1, 250)
    plt.plot(X_plot, f(X_plot), "r:", label="True function")
    plt.plot(X_sample, Y_sample, "ko", markersize=5, label="Observations")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.savefig("data.pdf")
    plt.close()
    del X_plot

    # Plot 2: prior width, fixed lengthscale of 0.05
    plt.subplots(2, 2, sharey="row", sharex=True)
    lengthscale = 0.05
    plot_gp_and_acquisition(
        X_sample,
        Y_sample,
        lengthscale=lengthscale,
        kernel_amplitude=1.0,
        noise=noise,
        subplot_base=1,
        title_suffix="$\\sigma=1.0$",
    )
    plot_gp_and_acquisition(
        X_sample,
        Y_sample,
        lengthscale=lengthscale,
        kernel_amplitude=0.01,
        noise=noise,
        subplot_base=2,
        title_suffix="$\\sigma=0.1$",
    )
    plt.savefig("prior-width.pdf")
    plt.close()

    # Plot 3: lengthscale, fixed prior width of 1.0
    plt.subplots(2, 2, sharey="row", sharex=True)
    amplitude = 0.1
    plot_gp_and_acquisition(
        X_sample,
        Y_sample,
        lengthscale=0.05,
        kernel_amplitude=amplitude,
        noise=noise,
        subplot_base=1,
        title_suffix="$\\ell=0.05$",
    )
    plot_gp_and_acquisition(
        X_sample,
        Y_sample,
        lengthscale=5.0,
        kernel_amplitude=amplitude,
        noise=noise,
        subplot_base=2,
        title_suffix="$\\ell=5.0$",
    )
    plt.savefig("lengthscale.pdf")
    plt.close()
