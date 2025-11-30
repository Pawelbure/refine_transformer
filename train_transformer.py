#!/usr/bin/env python
# train_transformer.py
#
# Script 3:
# - loads pre-generated two-body data from data/two_body_dataset_*.npz
# - loads latest trained KoopmanAE (encoder/decoder) from train_koopman_ae.py checkpoints
# - builds windowed datasets on normalized trajectories
# - trains a Transformer in latent space to predict next latent state
# - evaluates one-step MSE and long rollout
# - saves model + loss curves + a 2D orbit rollout plot

import os
import glob
import math
from datetime import datetime

import argparse

from experiment_configs import get_experiment_config, DEFAULT_EXPERIMENT
from train_koopman_ae import KoopmanAE

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

from utils import *
    
# ============================================================
# Dataset for Transformer: windowed x_in / x_out
# ============================================================
class WindowedSequenceDataset(Dataset):
    """
    data: np.ndarray (N_traj, T_total, x_dim), normalized
    Each sample:
      x_in  : (SEQ_LEN, x_dim)
      x_out : (HORIZON, x_dim)
    """
    def __init__(self, data, seq_len, horizon):
        assert data.ndim == 3
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

        self.index = []  # list of (traj_idx, start_t)
        N, T, _ = data.shape
        for n in range(N):
            max_start = T - seq_len - horizon
            if max_start <= 0:
                continue
            for t0 in range(max_start):
                self.index.append((n, t0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        n, t0 = self.index[idx]
        x_traj = self.data[n]  # (T_total, x_dim)
        x_in  = x_traj[t0:t0 + self.seq_len]                  # (SEQ_LEN, x_dim)
        x_out = x_traj[t0 + self.seq_len:t0 + self.seq_len + self.horizon]  # (HORIZON, x_dim)
        return {
            "x_in":  torch.from_numpy(x_in.astype(np.float32)),
            "x_out": torch.from_numpy(x_out.astype(np.float32)),
        }

# ============================================================
# Transformer in latent space
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LatentTransformer(nn.Module):
    def __init__(self,
                 latent_dim=8,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.0,
                 max_len=500):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_norm = nn.LayerNorm(latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.readout = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_seq):
        """
        z_seq: (B, T, latent_dim)
        returns z_next_pred: (B, latent_dim)
        """
        z_seq = self.input_norm(z_seq)
        z_seq = self.pos_encoder(z_seq)
        h = self.transformer(z_seq)      # (B, T, latent_dim)
        h_last = h[:, -1, :]
        return self.readout(h_last)

# ============================================================
# Training utilities
# ============================================================
def train_transformer(dyn_model, encoder, decoder,
                      train_loader, val_loader,
                      num_epochs, lr, device, out_dir,
                      test_norm, state_mean, state_std,
                      t_eval, seq_len, rollout_steps):
    dyn_model.to(device)
    encoder.to(device)
    decoder.to(device)

    # freeze encoder/decoder
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    encoder.eval()
    decoder.eval()

    optimizer = torch.optim.Adam(dyn_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        # -------------------
        # Train
        # -------------------
        dyn_model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_in  = batch["x_in"].to(device)   # (B, T, x_dim)
            x_out = batch["x_out"].to(device)  # (B, H, x_dim), H=1 here

            optimizer.zero_grad()

            with torch.no_grad():
                z_seq = encoder(x_in)                 # (B,T,d)
                z_next_true = encoder(x_out[:, 0, :]) # (B,d)

            z_next_pred = dyn_model(z_seq)            # (B,d)

            # latent loss
            loss_latent = mse(z_next_pred, z_next_true)

            # x-space loss (decoded)
            x_next_pred = decoder(z_next_pred)        # (B,x_dim)
            x_next_true = x_out[:, 0, :]              # (B,x_dim)
            loss_x = mse(x_next_pred, x_next_true)

            loss = loss_latent + 0.5 * loss_x
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_in.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------
        # Validation
        # -------------------
        dyn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_in  = batch["x_in"].to(device)
                x_out = batch["x_out"].to(device)

                z_seq = encoder(x_in)
                z_next_true = encoder(x_out[:, 0, :])
                z_next_pred = dyn_model(z_seq)

                loss_latent = mse(z_next_pred, z_next_true)
                x_next_pred = decoder(z_next_pred)
                x_next_true = x_out[:, 0, :]
                loss_x = mse(x_next_pred, x_next_true)

                loss = loss_latent + 0.5 * loss_x
                val_loss += loss.item() * x_in.size(0)

        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": dyn_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(out_dir, "transformer_best.pt"),
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"[Transformer] Epoch {epoch:03d} | Train: {train_loss:.4e} | Val: {val_loss:.4e}")

        if epoch % 2 == 0:
            plot_rollout_example_2d_orbit(
                test_norm=test_norm,
                state_mean=state_mean,
                state_std=state_std,
                t_eval=t_eval,
                encoder=encoder,
                decoder=decoder,
                dyn_model=dyn_model,
                seq_len=seq_len,
                n_future=rollout_steps,
                out_dir=out_dir,
                device=device,
                epoch=epoch
            )
            print(f"  -> Saved Transformer rollout orbit plot for epoch {epoch}.")

    return best_val_loss, history

# ============================================================
# Evaluation utilities
# ============================================================
def one_step_mse_xspace(encoder, decoder, dyn_model, data_loader, device):
    encoder.eval()
    decoder.eval()
    dyn_model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in data_loader:
            x_in  = batch["x_in"].to(device)
            x_out = batch["x_out"].to(device)

            z_seq = encoder(x_in)
            z_next_true = encoder(x_out[:, 0, :])
            z_next_pred = dyn_model(z_seq)

            x_next_true = x_out[:, 0, :]
            x_next_pred = decoder(z_next_pred)

            loss = mse(x_next_pred, x_next_true)
            total_loss += loss.item() * x_in.size(0)
            n += x_in.size(0)

    return total_loss / n


def rollout_transformer(x_init_norm, n_steps, encoder, decoder, dyn_model, device):
    """
    x_init_norm: (T0, x_dim) normalized initial physical window
    returns:
        x_pred_all_norm: (T0 + n_steps, x_dim) normalized
        first T0 rows = given window
        next n_steps = autoregressive transformer predictions
    """
    encoder.eval()
    decoder.eval()
    dyn_model.eval()

    x_init_t = torch.from_numpy(x_init_norm.astype(np.float32)).to(device)  # (T0, x_dim)
    T0, x_dim = x_init_t.shape

    with torch.no_grad():
        z_seq = encoder(x_init_t).unsqueeze(0)  # (1, T0, d)
        x_list = [x_init_t]  # first block: (T0, x_dim)

        for _ in range(n_steps):
            z_next = dyn_model(z_seq)      # (1,d)
            x_next = decoder(z_next)       # (1,x_dim)
            x_list.append(x_next)          # keep 2D

            # update latent sequence, keep last T0 states
            z_seq = torch.cat([z_seq, z_next.unsqueeze(1)], dim=1)  # (1,T0+1,d)
            z_seq = z_seq[:, -T0:, :]                              # (1,T0,d)

        x_pred_all = torch.cat(x_list, dim=0)  # (T0 + n_steps, x_dim)

    return x_pred_all.cpu().numpy()


def plot_rollout_example_2d_orbit(test_norm, state_mean, state_std, t_eval,
                                  encoder, decoder, dyn_model,
                                  seq_len, n_future, out_dir, device, epoch):
    """
    Use first test trajectory:
      - take an initial window of length seq_len
      - autoregressively roll out n_future steps with Transformer
      - plot 2D orbit (x-y for each mass), true vs rollout
    """
    if test_norm.shape[0] == 0:
        return

    x_traj_norm = test_norm[0]  # (T_total, 4)
    T_total = x_traj_norm.shape[0]
    if T_total < seq_len + n_future + 1:
        n_future = max(1, T_total - seq_len - 1)

    start_idx = 0
    x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]

    # rollout in normalized space
    x_pred_all_norm = rollout_transformer(
        x_init_norm, n_future, encoder, decoder, dyn_model, device
    )  # (seq_len + n_future, 4)

    x_true_seg_norm = x_traj_norm[start_idx:start_idx + seq_len + n_future]

    # denormalize
    x_pred_all = x_pred_all_norm * state_std + state_mean
    x_true_seg = x_true_seg_norm * state_std + state_mean

    # --- 2D orbit plot ---
    x1_true, y1_true = x_true_seg[:, 0], x_true_seg[:, 1]
    x2_true, y2_true = x_true_seg[:, 2], x_true_seg[:, 3]

    x1_pred, y1_pred = x_pred_all[:, 0], x_pred_all[:, 1]
    x2_pred, y2_pred = x_pred_all[:, 2], x_pred_all[:, 3]

    plt.figure(figsize=(7, 7))
    plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
    plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

    plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (pred)", linewidth=1.5, color="C0")
    plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (pred)", linewidth=1.5, color="C1")

    idx_boundary = seq_len - 1
    plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                color="C0", marker="o", s=40, label="Start pred M1")
    plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                color="C1", marker="o", s=40, label="Start pred M2")

    plt.title("Transformer rollout: 2D orbit, true vs pred (test example)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"transformer_rollout_orbit_{epoch}.png"))
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help="Name of experiment configuration to use.",
    )
    parser.add_argument(
        "--reuse_transformer",
        action="store_true",
        help="Reuse latest trained Transformer instead of training a new one.",
    )
    args = parser.parse_args()

    cfg     = get_experiment_config(args.experiment)
    ds_cfg  = cfg.dataset
    k_cfg   = cfg.koopman
    tf_cfg  = cfg.transformer

    EXP_DATA_ROOT = f"{cfg.name}/{cfg.DATA_ROOT}"
    EXP_OUTPUT_ROOT = f"{cfg.name}/outputs"
    
    SEQ_LEN    = ds_cfg.SEQ_LEN
    HORIZON    = ds_cfg.HORIZON

    LATENT_DIM = k_cfg.LATENT_DIM
    HIDDEN_DIM = k_cfg.HIDDEN_DIM

    NHEAD      = tf_cfg.NHEAD
    NUM_LAYERS = tf_cfg.NUM_LAYERS
    DIM_FEEDFORWARD = tf_cfg.DIM_FEEDFORWARD
    DROPOUT    = tf_cfg.DROPOUT
    BATCH_SIZE = tf_cfg.BATCH_SIZE
    EPOCHS     = tf_cfg.EPOCHS
    LR         = tf_cfg.LR

    ROLLOUT_STEPS = tf_cfg.ROLLOUT_STEPS
    max_len_tf    = SEQ_LEN + ROLLOUT_STEPS + tf_cfg.MAX_LEN_EXTRA
    
    # 1) Load dataset
    ds_file = find_latest_dataset(data_dir=EXP_DATA_ROOT)
    print(f"Loading dataset from: {ds_file}")
    data = np.load(ds_file)

    t_eval     = data["t_eval"]
    state_mean = data["state_mean"]
    state_std  = data["state_std"]

    train_norm = data["train_norm"]   # (N_train, T, 4)
    val_norm   = data["val_norm"]     # (N_val,   T, 4)
    test_norm  = data["test_norm"]    # (N_test,  T, 4)

    N_train, T_total, x_dim = train_norm.shape
    print(f"Train_norm shape: {train_norm.shape}")
    print(f"Val_norm shape:   {val_norm.shape}")
    print(f"Test_norm shape:  {test_norm.shape}")

    # 2) Load latest KoopmanAE (from train_koopman_ae.py)
    koopman_dir, koopman_ckpt_path = find_latest_koopman(output_dir=EXP_OUTPUT_ROOT)
    print(f"Loading KoopmanAE from: {koopman_ckpt_path}")
    ckpt = torch.load(koopman_ckpt_path, map_location=DEVICE)

    koopman_model = KoopmanAE(x_dim=x_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    koopman_model.load_state_dict(ckpt["model_state_dict"])
    koopman_model.to(DEVICE)

    encoder = koopman_model.encoder
    decoder = koopman_model.decoder

    # 3) Build windowed datasets/loaders for Transformer
    train_ds = WindowedSequenceDataset(train_norm, seq_len=SEQ_LEN, horizon=HORIZON)
    val_ds   = WindowedSequenceDataset(val_norm,   seq_len=SEQ_LEN, horizon=HORIZON)
    test_ds  = WindowedSequenceDataset(test_norm,  seq_len=SEQ_LEN, horizon=HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Transformer train samples: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # 4) Prepare output dir
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EXP_OUTPUT_ROOT, f"transformer_{time_tag}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Dataset file: {ds_file}\n")
        f.write(f"KoopmanAE dir: {koopman_dir}\n")
        f.write(f"SEQ_LEN: {SEQ_LEN}\n")
        f.write(f"HORIZON: {HORIZON}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"LATENT_DIM: {LATENT_DIM}\n")
        f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
        f.write(f"NHEAD: {NHEAD}\n")
        f.write(f"NUM_LAYERS: {NUM_LAYERS}\n")
        f.write(f"DIM_FEEDFORWARD: {DIM_FEEDFORWARD}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"ROLLOUT_STEPS: {ROLLOUT_STEPS}\n")
    
    # 5) Initialize (or reuse) Transformer
    if args.reuse_transformer:
        print("\n--reuse_transformer was specified. Trying to load latest Transformer model...")

        last_tf_dir, last_tf_ckpt_path = find_latest_transformer()
        print(f"Loading Transformer from: {last_tf_ckpt_path}")

        # Load checkpoint
        ckpt_tf = torch.load(last_tf_ckpt_path, map_location=DEVICE)

        # Instantiate Transformer using config hyperparameters
        dyn_model = LatentTransformer(
            latent_dim=tf_cfg.LATENT_DIM,
            nhead=tf_cfg.NHEAD,
            num_layers=tf_cfg.NUM_LAYERS,
            dim_feedforward=tf_cfg.DIM_FEEDFORWARD,
            dropout=tf_cfg.DROPOUT,
            max_len=max_len_tf,
        ).to(DEVICE)

        dyn_model.load_state_dict(ckpt_tf["model_state_dict"], strict=False)
        dyn_model.eval()

        # best_val_loss comes from the reused checkpoint (may be NaN if missing)
        best_val_loss = float(ckpt_tf.get("val_loss", float("nan")))
        print(f"Reusing Transformer model — skipping training. "
              f"(stored val_loss = {best_val_loss:.4e})\n")

        # Note: no per-epoch plots here (no training loop), but we’ll still
        # create a fresh rollout orbit plot later in this script.
    else:
        print("\nTraining new Transformer model...\n")

        dyn_model = LatentTransformer(
            latent_dim=tf_cfg.LATENT_DIM,
            nhead=tf_cfg.NHEAD,
            num_layers=tf_cfg.NUM_LAYERS,
            dim_feedforward=tf_cfg.DIM_FEEDFORWARD,
            dropout=tf_cfg.DROPOUT,
            max_len=max_len_tf,
        ).to(DEVICE)

        # train_transformer will:
        #   - log train/val loss
        #   - save transformer_best.pt
        #   - plot 2D orbit every 2 epochs (using test_norm, etc.)
        best_val_loss, history = train_transformer(
            dyn_model, encoder, decoder,
            train_loader, val_loader,
            num_epochs=EPOCHS, lr=LR,
            device=DEVICE, out_dir=out_dir,
            test_norm=test_norm,
            state_mean=state_mean,
            state_std=state_std,
            t_eval=t_eval,
            seq_len=SEQ_LEN,
            rollout_steps=ROLLOUT_STEPS,
        )

        # After training, reload the best checkpoint (for final eval/plots)
        tf_ckpt = torch.load(os.path.join(out_dir, "transformer_best.pt"),
                             map_location=DEVICE)
        dyn_model.load_state_dict(tf_ckpt["model_state_dict"])
        dyn_model.to(DEVICE)

    print(f"Best Transformer val loss: {best_val_loss:.4e}")

    # 7) One-step MSE in x-space on test set
    test_one_step_mse = one_step_mse_xspace(
        encoder, decoder, dyn_model, test_loader, DEVICE
    )
    print(f"One-step MSE in x-space (test): {test_one_step_mse:.4e}")
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"best_val_loss: {best_val_loss:.6e}\n")
        f.write(f"test_one_step_mse_xspace: {test_one_step_mse:.6e}\n")

    # 8) Long rollout example on first test trajectory – 2D orbit only
    plot_rollout_example_2d_orbit(
        test_norm=test_norm,
        state_mean=state_mean,
        state_std=state_std,
        t_eval=t_eval,
        encoder=encoder,
        decoder=decoder,
        dyn_model=dyn_model,
        seq_len=SEQ_LEN,
        n_future=ROLLOUT_STEPS,
        out_dir=out_dir,
        device=DEVICE,
        epoch="final"
    )

    print(f"Transformer training and evaluation finished. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()