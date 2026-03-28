"""
PINN for the Damped Harmonic Oscillator
ODE  : d²x/dz² + 2ξ·dx/dz + x = 0
ICs  : x(0) = 0.7,  dx/dz(0) = 1.2
Domain: z ∈ [0, 20],  ξ ∈ [0.1, 0.4]

The network takes (z, ξ) as input and predicts x(z; ξ).
ICs are enforced exactly via output transformation:
    x_hat = x0 + v0*z + z^2 * N(z, ξ)
so x_hat(0) = x0 and dx_hat/dz(0) = v0 by construction.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X0 = 0.7
V0 = 1.2
Z_MAX = 20.0
XI_LO = 0.1
XI_HI = 0.4

N_COLLOCATION = 10_000
EPOCHS = 20_000
LR = 1e-3
HIDDEN = 64
N_LAYERS = 5


def exact_solution(z, xi):
    wd = np.sqrt(1.0 - xi**2)
    A = X0
    B = (V0 + xi * X0) / wd
    return np.exp(-xi * z) * (A * np.cos(wd * z) + B * np.sin(wd * z))


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, HIDDEN), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
        layers.append(nn.Linear(HIDDEN, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, xi):
        z_n = z / Z_MAX
        xi_n = (xi - XI_LO) / (XI_HI - XI_LO)
        inp = torch.cat([z_n, xi_n], dim=1)
        N = self.net(inp)
        return X0 + V0 * z + z**2 * N


def physics_loss(model, z, xi):
    z = z.requires_grad_(True)
    xi = xi.requires_grad_(True)

    x = model(z, xi)

    x_z = torch.autograd.grad(
        x, z, grad_outputs=torch.ones_like(x),
        create_graph=True, retain_graph=True
    )[0]

    x_zz = torch.autograd.grad(
        x_z, z, grad_outputs=torch.ones_like(x_z),
        create_graph=True, retain_graph=True
    )[0]

    residual = x_zz + 2.0 * xi * x_z + x
    return torch.mean(residual**2)


def train():
    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, factor=0.5, min_lr=1e-5
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        z = (torch.rand(N_COLLOCATION, 1) * Z_MAX).to(DEVICE)
        xi = (torch.rand(N_COLLOCATION, 1) * (XI_HI - XI_LO) + XI_LO).to(DEVICE)

        optimizer.zero_grad()
        loss = physics_loss(model, z, xi)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if epoch % 500 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:>6}  loss {loss.item():.4e}  lr {lr_now:.2e}")
            history.append((epoch, loss.item()))

    return model, history


@torch.no_grad()
def predict(model, z_np, xi_val):
    model.eval()
    z = torch.tensor(z_np[:, None], dtype=torch.float32).to(DEVICE)
    xi = torch.full_like(z, xi_val)
    return model(z, xi).cpu().numpy().ravel()


def relative_l2(model, xi_vals, n_pts=500):
    z = np.linspace(0, Z_MAX, n_pts)
    errors = {}
    for xi in xi_vals:
        pred = predict(model, z, xi)
        exact = exact_solution(z, xi)
        errors[xi] = np.sqrt(np.mean((pred - exact)**2)) / (np.sqrt(np.mean(exact**2)) + 1e-12)
    return errors


def plot_results(model, history):
    xi_test = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    z = np.linspace(0, Z_MAX, 500)
    colors = plt.cm.tab10(np.linspace(0, 1, len(xi_test)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PINN — Damped Harmonic Oscillator", fontsize=14)

    # solution curves
    ax = axes[0, 0]
    for xi, c in zip(xi_test, colors):
        ax.plot(z, exact_solution(z, xi), "--", color=c, lw=1.2, alpha=0.6)
        ax.plot(z, predict(model, z, xi), "-", color=c, lw=1.8, label=f"ξ={xi:.2f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("x(z; ξ)")
    ax.set_title("PINN (solid) vs Exact (dashed)")
    ax.legend(ncol=2, fontsize=8)

    # pointwise error
    ax = axes[0, 1]
    for xi, c in zip(xi_test, colors):
        err = np.abs(predict(model, z, xi) - exact_solution(z, xi)) + 1e-10
        ax.semilogy(z, err, color=c, lw=1.4, label=f"ξ={xi:.2f}")
    ax.set_xlabel("z")
    ax.set_ylabel("|error|")
    ax.set_title("Pointwise Absolute Error")
    ax.legend(ncol=2, fontsize=8)

    # training loss
    ax = axes[1, 0]
    epochs, losses = zip(*history)
    ax.semilogy(epochs, losses, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ODE residual loss")
    ax.set_title("Training Loss")

    # relative L2 by xi
    ax = axes[1, 1]
    errors = relative_l2(model, xi_test)
    xi_labels = [f"{xi:.2f}" for xi in xi_test]
    vals = [errors[xi] * 100 for xi in xi_test]
    bars = ax.bar(xi_labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("ξ")
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Accuracy by Damping Ratio")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/pinn_dho_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved → pinn_dho_results.png")


if __name__ == "__main__":
    model, history = train()

    print("\nRelative L2 errors:")
    errors = relative_l2(model, [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    for xi, err in errors.items():
        print(f"  ξ={xi:.2f}  →  {err*100:.4f}%")

    plot_results(model, history)

    torch.save(model.state_dict(), "/mnt/user-data/outputs/pinn_dho_weights.pt")
    print("saved → pinn_dho_weights.pt")
