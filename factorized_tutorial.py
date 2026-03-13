import marimo


__generated_with = "0.8.22"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _():
    import math
    import io
    import torch
    from torchvision import transforms
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
    return (math, io, torch, transforms, np, Image, plt, colors, bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)


@app.cell
def _(torch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = 'mse'  # only pre-trained model for mse are available for now
    quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
    return (device, metric, quality)

# %config InlineBackend.figure_format = 'retina'



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
loading networks
    """
    )
    return

@app.cell
def _(
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    device,
    mbt2018,
    mbt2018_mean,
    quality,
):
    networks = {
        'bmshj2018-factorized': bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device),
        'bmshj2018-hyperprior': bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device),
        'mbt2018-mean': mbt2018_mean(quality=quality, pretrained=True).eval().to(device),
        'mbt2018': mbt2018(quality=quality, pretrained=True).eval().to(device),
        'cheng2020-anchor': cheng2020_anchor(quality=quality, pretrained=True).eval().to(device),
    }
    return (networks,)

@app.cell
def _(Image, device, transforms):
    img = Image.open('./kodim19.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return img, x



@app.cell
def _loadNet(networks, x, quantization_mode):
    net = networks['bmshj2018-factorized']
    return net


@app.cell
def _(device, net, torch):
    dummy_variable = torch.linspace(-5, 5, 201)
    dummy_in = dummy_variable.view(1, 1, 1, 201).expand(1, 320, 1, 201).to(device)
    eb_out = net.entropy_bottleneck(dummy_in, training=True)
    return dummy_in, dummy_variable, eb_out


@app.cell
def _(mo):
    channel_slider = mo.ui.slider(0, 319, value=1, step=1, label='Latent Channel Index ')

    mo.md(
        f"""
        **Select latent channel to visualize:**

        {channel_slider}
        """
    )
    return (channel_slider,)

@app.cell
def _plotFactorizedEntropyModel(eb_out, plt, torch, channel_slider):
    x_vals = torch.linspace(-5, 5, 201).cpu().numpy()
    y_hat_out = eb_out[0][0].detach().cpu().numpy()  # (320, 1, 201)
    likelihoods = eb_out[1][0].detach().cpu().numpy()  # (320, 1, 201)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(x_vals, likelihoods[channel_slider.value, 0], label=f'ch {channel_slider.value}')

    ax.set_xlabel('Input')
    ax.set_ylabel('Likelihood')
    ax.set_title('Channel-wise Factorized Entropy Model')
    ax.legend()
    ax.set_xlim([-6, 6])
    ax.set_ylim([-0.2, 1.2])
    plt.gca()
    return 

@app.cell
def _(mo):
    quantization_mode = mo.ui.checkbox(value=True, label="Do quantization")
    mo.md(
        f"""
        **Toggle between 2 modes of quantization:**

        {quantization_mode}
        """
    )
    return (quantization_mode, )
    
@app.cell
def _encForward(net, x, quantization_mode):
    y = net.g_a(x)
    training = not quantization_mode.value
    y_hat, y_likelihoods = net.entropy_bottleneck(y, training=training)
    return y, y_hat, y_likelihoods



@app.cell
def __(plt, img, y_hat, channel_slider):
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    channel_image = y_hat[0,channel_slider.value,:,:].detach().cpu().numpy()
    im = axes[1].imshow(channel_image)
    axes[1].set_title(f'Latent Channel {channel_slider.value}')
    fig1.colorbar(im, ax=axes[1], shrink=0.8)
    plt.tight_layout()
    plt.gca()
    return



if __name__ == "__main__":
    app.run()

