# PINN: Damped Harmonic Oscillator

A Physics-Informed Neural Network (PINN) trained in PyTorch to solve the damped harmonic oscillator ODE, conditioned on the damping ratio xi so that a single trained model covers the full parameter range.

---

## Problem Statement

The ODE being solved is:

```
d²x/dz² + 2*xi*dx/dz + x = 0
```

with fixed initial conditions:

```
x(0)  = x0 = 0.7
dx/dz(0) = v0 = 1.2
```

The domain is `z in [0, 20]` and the network is conditioned on `xi in [0.1, 0.4]`, covering the under-damped regime.

---

## Method

### Network Architecture

The network takes the pair `(z, xi)` as input and outputs the predicted displacement `x_hat(z; xi)`. Both inputs are normalised to `[0, 1]` before entering the MLP.

```
Input:   [z_norm, xi_norm]   (2 features)
Hidden:  5 layers x 64 neurons, tanh activation
Output:  scalar N(z, xi)
```

### Hard Initial Condition Enforcement

Rather than adding an IC penalty term to the loss, the network output is transformed analytically:

```
x_hat(z; xi) = x0 + v0*z + z^2 * N(z, xi)
```

This construction guarantees `x_hat(0) = x0` and `dx_hat/dz(0) = v0` for any network weights. The IC constraints are therefore satisfied exactly throughout training, not approximately.

### Loss Function

Only a single physics residual loss is minimised:

```
L = mean[ (x'' + 2*xi*x' + x)^2 ]
```

Derivatives with respect to `z` are computed via automatic differentiation using `torch.autograd.grad` with `create_graph=True`, enabling backpropagation through the differential operators.

### Training Details

| Setting | Value |
|---|---|
| Collocation points per epoch | 10,000 |
| Optimiser | Adam |
| Initial learning rate | 1e-3 |
| Scheduler | ReduceLROnPlateau (factor 0.5) |
| Epochs | 20,000 |
| Gradient clipping | max_norm = 1.0 |
| Weight initialisation | Xavier normal |

Collocation points `(z, xi)` are resampled uniformly at every epoch, which acts as a stochastic regulariser and prevents the network from overfitting to a fixed quadrature grid.

---

## Validation

The exact closed-form solution for the under-damped case (`xi < 1`) is:

```
x(z) = exp(-xi*z) * [ A*cos(wd*z) + B*sin(wd*z) ]

where:
  wd = sqrt(1 - xi^2)
  A  = x0
  B  = (v0 + xi*x0) / wd
```

The PINN prediction is compared against this analytic solution using the relative L2 error:

```
err = sqrt( mean((x_pred - x_exact)^2) ) / sqrt( mean(x_exact^2) )
```

### Observed Results

| xi | Relative L2 Error |
|---|---|
| 0.10 | 53.08 % |
| 0.15 | 39.05 % |
| 0.20 | 28.86 % |
| 0.25 | 22.32 % |
| 0.30 | 18.56 % |
| 0.35 | 16.56 % |
| 0.40 | 16.06 % |

The higher errors at small xi reflect the well-known spectral bias of neural networks: low-damping solutions are more oscillatory over the long domain `[0, 20]`, which MLPs with smooth activations struggle to resolve without additional techniques. Accuracy improves monotonically with xi because heavier damping reduces oscillation frequency and amplitude.

---

## Known Limitations and Improvements

**Spectral bias.** Standard MLPs with tanh activations preferentially learn low-frequency components. For low xi values the solution contains many oscillations over `z in [0, 20]`, which explains the elevated errors. Fourier feature embeddings or sinusoidal activations (SIREN) would mitigate this.

**Causal training.** The residual is sampled uniformly in z, so the network receives no gradient signal enforcing causality. Causal weighting schemes that progressively expand the training window from z=0 outward tend to improve accuracy on long-time domains.

**Adaptive sampling.** Replacing uniform collocation sampling with residual-based adaptive refinement (RAR) concentrates points where the PDE residual is largest, improving efficiency.

**Loss weighting.** Although ICs are hard-constrained here, adding a separate boundary loss with tunable weights (or using NTK-based balancing) can help when extending to problems where hard constraints are unavailable.

---

## File Structure

```
pinn_damped_oscillator.py   main script: model definition, training, evaluation, plotting
pinn_dho_weights.pt         saved model weights after training
pinn_dho_results.png        diagnostic figure (4 panels)
README.md                   this file
```

---

## Requirements

```
torch >= 2.0
numpy
matplotlib
```

---

## Usage

```bash
python pinn_damped_oscillator.py
```

Training runs for 20,000 epochs and prints loss every 500 epochs. On a modern CPU this takes roughly 10 to 20 minutes. On a CUDA GPU it is substantially faster. The script saves the weights and the diagnostic figure automatically on completion.

