# Generating Faces with a DCGAN - Adversarial Task

A from-scratch implementation of a Generative Adversarial Network (GAN) that learns to synthesize human face images by training a **Generator** and a **Discriminator** against each other in a minimax game.

## What Is This Project About?

This project explores **generative modeling** — teaching a neural network to produce realistic data (in this case, human faces) rather than just classify or predict on existing data. It's built around the core GAN idea introduced by Goodfellow et al.: two networks compete, and that competition drives both to improve.

- The **Generator** starts with random noise and learns to turn it into a face.
- The **Discriminator** looks at real faces and generated (fake) faces, and learns to tell them apart.
- As training progresses, the Generator gets better at fooling the Discriminator, and the Discriminator gets better at catching fakes — pushing the Generator toward increasingly realistic output.

## Dataset

Faces are trained on the **Labeled Faces in the Wild (LFW)** dataset, downscaled to **36×36 grayscale/RGB** images for faster experimentation. The dataset also ships with facial attribute labels (usable for attribute-conditioned generation, e.g. adding a smile), though this notebook focuses on unconditional face generation.

## Architecture

### Generator
Maps a 256-dimensional latent noise vector `z ~ N(0, 1)` to a 36×36 image:

| Layer | Details |
|---|---|
| Input | Noise vector, size 256 |
| Dense | 10 × 8 × 8, ELU activation |
| Reshape | (8, 8, 10) |
| Deconv2D ×2 | 64 filters, 5×5 kernel, ELU |
| UpSampling2D | 2× |
| Deconv2D ×3 | 32 filters, 3×3 kernel, ELU |
| Conv2D (output) | 3 filters, 3×3 kernel, linear activation |

### Discriminator
A convolutional binary classifier that outputs log-probabilities for `real` vs `fake`:

| Layer | Details |
|---|---|
| Conv2D ×2 | 16 → 32 filters, 2×2, ReLU |
| MaxPool2D | |
| Conv2D ×2 | 64 → 128 filters, 2×2, ReLU |
| MaxPool2D | |
| Flatten → Dense | 256 units, tanh |
| Dense (output) | 2 units, log-softmax |


## Training

- **Framework:** TensorFlow 1.x (Keras Sequential API), trained inside a Google Colab environment.
- **Discriminator loss:** negative log-likelihood of correctly classifying real vs. generated samples, with L2 regularization on the final layer's weights. Optimized with **SGD** (lr = 1e-3).
- **Generator loss:** negative log-likelihood of the Discriminator classifying generated samples as real. Optimized with **Adam** (lr = 1e-4).
- **Training loop:** for each epoch, the Discriminator is updated 5 times per 1 Generator update (a common GAN stabilization trick), using batches of 100 real and 100 noise-generated samples.
- **Duration:** run for up to 50,000 epochs, with progress (sample generations + real-vs-fake probability histograms) visualized every 100 epochs. The final results below come from roughly **15,000 iterations**.

## Flow of the Project

1. Load and preprocess the LFW face dataset (resize, normalize to `[0, 1]`).
2. Build the Generator and Discriminator networks.
3. Wire up the adversarial training graph: real images and generator-produced images both pass through the Discriminator.
4. Define separate loss functions and optimizers for the Generator and Discriminator.
5. Train iteratively, alternating Discriminator and Generator updates.
6. Periodically sample generated faces and plot the Discriminator's output distribution on real vs. fake data to monitor convergence.
7. Visualize the final batch of generated faces.

## Results

After training, the Generator produces recognizably face-like images from pure noise — evidence that it learned the underlying structure of human faces (position of eyes, nose, mouth, general shading) purely from adversarial feedback, with no explicit labels or reconstruction loss involved.

The notebook also tracks the Discriminator's confidence scores on real vs. generated batches (`D(x)` vs `D(G(z))`) over time, showing the two distributions converge as the Generator improves — a visual signal of the adversarial game approaching balance. As noted in the notebook, longer training (well beyond 15k iterations) yields noticeably sharper, more realistic results.
