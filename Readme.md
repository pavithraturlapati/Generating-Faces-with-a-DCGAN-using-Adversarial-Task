# Generating Faces with a DCGAN using Adversarial Task

A Deep Convolutional Generative Adversarial Network (DCGAN) that learns to synthesize realistic human face images by training a **Generator** and a **Discriminator** against each other in an adversarial game.

## Overview

This project explores **generative modeling** — teaching a network to produce new, realistic data rather than classify or predict on existing data. Following Ian Goodfellow's original GAN formulation, two networks are trained simultaneously with opposing goals:

- The **Generator (G)** takes random noise and tries to turn it into a convincing fake image.
- The **Discriminator (D)** looks at real and generated images and tries to tell them apart.

Neither network is ever shown a "correct" target output directly — the Generator only ever learns through the Discriminator's feedback. As training progresses, both networks get better in tandem, ideally reaching a point where the Generator's fakes are indistinguishable from real data.

## The Minimax Objective

```
min_G max_D  E[x~data][log D(x)] + E[z~noise][log(1 - D(G(z)))]
```

D wants to maximize this (correctly separate real from fake); G wants to minimize it (fool D). In practice, this notebook uses the standard **non-saturating loss**: instead of minimizing `log(1 - D(G(z)))` (which gives weak gradients early on), G directly maximizes `log D(G(z)))`. Both losses use log-softmax outputs over `[fake, real]` for numerical stability, and the Discriminator's output layer has L2 regularization to keep it from overpowering the Generator too early.

## Dataset

**Labeled Faces in the Wild (LFW)**, resized to 36×36 and pixel values normalized to `[0, 1]`. The dataset's facial attribute labels (e.g. smiling) are also loaded but unused here — they'd support a future conditional-GAN extension.

## Architecture

**Generator** — noise (256-dim) → `Dense` → reshape to (8,8,10) → stacked `Deconv2D` (transposed convolutions, which *learn* how to upsample) + `UpSampling2D` layers → final `Conv2D` producing a 36×36×3 image. ELU activations are used throughout to keep gradients healthy during unstable early training.

**Discriminator** — stacked `Conv2D` + `MaxPool2D` blocks (16→32→64→128 filters, standard CNN feature-depth progression) → `Dense(256, tanh)` → `Dense(2, log_softmax)` for real/fake classification.

## Training Procedure

GAN training is unstable because both networks chase a moving target. This project uses a few standard stabilization tricks:

- **5:1 update ratio** — the Discriminator is updated 5 times per Generator update each step, so it stays ahead and gives G a useful signal.
- **Different optimizers per network** — Discriminator uses SGD (`lr=1e-3`); Generator uses Adam (`lr=1e-4`), which handles its more complex loss surface better.
- **Batches of 100** real images and 100 fresh noise samples per step.
- **Visual monitoring instead of relying on loss curves** — every 100 epochs, the notebook renders a grid of generated faces and plots `D(x)` vs `D(G(z))` probability histograms. As training succeeds, these two distributions increasingly overlap around 0.5, meaning D can no longer reliably tell real from fake.
- **Duration** — the loop supports up to 50,000 epochs; results here come from ~15,000 iterations. GAN quality keeps improving with more training since there's no traditional convergence point.

## Key Concepts

- **Latent space** — the 256-dim noise space G learns to map into realistic faces.
- **Mode collapse** — a common failure where G produces only a narrow variety of outputs.
- **Non-saturating loss** — the gradient-friendly loss reformulation used for G.
- **Transposed convolution** — learned upsampling, as opposed to fixed interpolation.

## Flow of the Project

1. Load and preprocess LFW faces (resize, normalize).
2. Build Generator and Discriminator as Keras `Sequential` models.
3. Wire both networks together so real and generated images pass through the same Discriminator.
4. Define separate losses/optimizers for D and G.
5. Train in alternating steps, visualizing samples and probability distributions periodically.
6. Inspect the final generated faces.

## Results

The Generator learns to produce recognizably face-like images purely from noise, using only adversarial feedback — no labels, no reconstruction loss.

<img width="1236" height="640" alt="noise_to_face" src="https://github.com/user-attachments/assets/bff9f367-ddd4-4637-9a64-accfa36d3891" />
