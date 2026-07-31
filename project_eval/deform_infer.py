#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deform_infer.py

Standalone inference entry point for DEFORM: not part of the upstream
roahmlab/DEFORM release (which only ships train_DEFORM.py's training loop, with
an eval loop inlined inside train() for validation-during-training use). This
adds a "load one checkpoint, roll one episode forward, dump every frame's
state" CLI, generalizing that same inlined eval loop (see train_DEFORM.py's
`if save_steps % evaluate_period == 0:` block) to an arbitrary-length single
episode instead of a fixed eval_time_horizon batch.

Recurrence (matches train_DEFORM.py's own eval loop exactly, just generalized
over T frames instead of a fixed window -- see deform_notes.md,
~/ros_ws/docs_n_papers/, for the derivation):

  frame 0,1 (ground truth): seed current_v = (f1 - f0) / dt, m_u0 = compute_u0(f1's
    first edge, init_direction).
  step k=0: forward(current_vert=f1 (ground truth), ..., input=f2's boundary
    (ground truth, clamped_selection slice), mode="evaluation_numpy")
    -> prediction for f2.
  step k=1: m_u0 parallel-transported from f1's edge to predicted-f2's edge;
    forward(current_vert=predicted f2, ..., input=f3's boundary, ...)
    -> prediction for f3.
  step k>=2: m_u0 parallel-transported from predicted-f(k)'s edge to
    predicted-f(k+1)'s edge; forward(current_vert=predicted f(k+1), ...,
    input=f(k+2)'s boundary, ...) -> prediction for f(k+2).

Only nodes clamped_selection=(0,1,-2,-1) ever come from ground truth (every
frame, not just the seed) -- see deform_notes.md's "Boundary-node convention"
section for why that's not cheating: this harness evaluates already-recorded
episodes, and DEFORM's own released training/eval code reads the boundary the
same way, straight out of the same ground-truth array, at both train and eval
time.

Input: one episode's pickled (T, 3, n_vert) array (prep_deform_dataset.py's own
output format -- same file used for training, so eval just replays a held-out
episode's own pickle), and a trained checkpoint's state_dict.

Output: one CSV, one row per predicted frame (frame index 2..T-1), columns
dlo_d01_x.. dlo_dNN_z (positions -- ground truth for the 4 clamped nodes,
predicted for the rest), dlo_d01_vx..dlo_dNN_vz (velocities, current_v),
dlo_e01_theta..dlo_e{n_edge}_theta (material twist angle per edge), proc_time_s,
status (PASS/FAIL -- FAIL if any NaN/Inf appears in the predicted state).
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DEFORM_func import DEFORM_func
from DEFORM_sim import DEFORM_sim
from util import computeEdges, computeLengths

INIT_DIRECTION = torch.tensor(((0., 0.6, 0.8), (0., 0., 1.)))  # matches train_DEFORM.py's
                                                                 # own global constant


def log(msg):
    print(msg)
    sys.stderr.flush()
    sys.stdout.flush()


def load_episode_pickle(path):
    """(T, 3, n_vert) -> (T, n_vert, 3) torch tensor, matching how
    Train_DeformData/Eval_DeformData transpose it on load."""
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return torch.from_numpy(np.asarray(arr)).float().transpose(1, 2)  # (T, n_vert, 3)


def build_model(dlo_type_config_path, checkpoint_path, device):
    with open(dlo_type_config_path, "r") as f:
        cfg = json.load(f)
    n_vert = cfg["n_vert"]
    n_edge = n_vert - 1
    deform_func = DEFORM_func(n_vert=n_vert, n_edge=n_edge, device=device)
    deform_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_edge, pbd_iter=10, device=device)
    deform_sim.DEFORM_func = deform_func
    # DEFORM_sim.__init__ hardcodes a 13-node (DLO1-shaped) placeholder rest_vert
    # internally, regardless of the n_vert passed in above -- train_DEFORM.py's own
    # per-DLO_type branches always overwrite it right after construction (before any
    # training/saving happens), so every real checkpoint's rest_vert is actually
    # (1, n_vert, 3). Resize the placeholder the same way before load_state_dict, or
    # loading any non-13-node checkpoint fails on a shape mismatch.
    import torch.nn as nn
    deform_sim.rest_vert = nn.Parameter(torch.zeros(1, n_vert, 3, device=device))
    state_dict = torch.load(checkpoint_path, map_location=device)
    deform_sim.load_state_dict(state_dict)
    # m_restEdgeL/m_restRegionL are plain attributes (not nn.Parameters, so NOT restored
    # by load_state_dict above) -- __init__ set them from the same wrong 13-node
    # placeholder rest_vert. Recompute them from the just-loaded REAL rest_vert (train_
    # DEFORM.py's own per-DLO_type branches do this same computeLengths(computeEdges(...))
    # call right after (re)assigning rest_vert, before any training/saving).
    deform_sim.m_restEdgeL, deform_sim.m_restRegionL = computeLengths(
        computeEdges(deform_sim.rest_vert.clone()))
    deform_sim.to(device)
    deform_sim.eval()
    return deform_sim, n_vert, n_edge


def run_episode(deform_sim, n_vert, n_edge, vertices, device, max_frames=None):
    """vertices: (T, n_vert, 3) ground-truth-ish node positions for this
    episode (interior nodes are dlo_perception's own estimate, clamped nodes
    are what forward() will overwrite with each step's true boundary anyway).
    Returns a list of per-frame dicts, one per predicted frame (index 2..T-1)."""
    T = vertices.size(0) if max_frames is None else min(vertices.size(0), max_frames)
    if T < 3:
        raise ValueError("episode has only %d usable frames, need >= 3" % T)

    clamped_selection = torch.tensor((0, 1, n_vert - 2, n_vert - 1))
    clamped_index = torch.zeros(n_vert)
    clamped_index[clamped_selection] = 1.0
    init_direction = INIT_DIRECTION.to(device).unsqueeze(dim=0)

    vertices = vertices.to(device)
    boundary = vertices[:, clamped_selection]  # (T, 4, 3) -- this episode's own boundary trajectory

    dt = deform_sim.dt
    frame0, frame1 = vertices[0:1], vertices[1:2]  # (1, n_vert, 3) each -- batch=1
    current_v = (frame1 - frame0) / dt
    m_restEdgeL = deform_sim.m_restEdgeL
    deform_sim.m_restWprev, deform_sim.m_restWnext, deform_sim.learned_pmass = \
        deform_sim.Rod_Init(1, init_direction, m_restEdgeL, clamped_index)

    rest_edges = computeEdges(frame1)
    m_u0 = deform_sim.DEFORM_func.compute_u0(rest_edges[:, 0].float(), init_direction[:, 0])
    theta_full = torch.zeros(1, n_edge, device=device)

    rows = []
    vert = None  # holds the PREVIOUS step's predicted vertex array (traj_num >= 1)
    pred_vert = None
    with torch.no_grad():
        for k in range(T - 2):  # k=0 predicts frame 2, k=T-3 predicts frame T-1
            t0 = time.time()
            input_boundary = boundary[k + 2:k + 3]  # this step's true boundary (frame k+2)

            if k == 0:
                current_vert = frame1
            elif k == 1:
                previous_edge = computeEdges(frame1)
                current_edges = computeEdges(pred_vert)
                m_u0 = deform_sim.DEFORM_func.parallelTransportFrame(
                    previous_edge[:, 0], current_edges[:, 0], m_u0)
                vert = pred_vert.clone()
                current_vert = vert
            else:
                previous_vert = vert.clone()
                vert = pred_vert.clone()
                previous_edge = computeEdges(previous_vert)
                current_edges = computeEdges(vert)
                m_u0 = deform_sim.DEFORM_func.parallelTransportFrame(
                    previous_edge[:, 0], current_edges[:, 0], m_u0)
                current_vert = vert

            pred_vert, current_v, theta_full = deform_sim(
                current_vert, current_v, init_direction, clamped_index, m_u0,
                input_boundary, clamped_selection, theta_full, mode="evaluation_numpy")

            proc_time_s = time.time() - t0
            pos = pred_vert[0].cpu().numpy()
            vel = current_v[0].cpu().numpy()
            theta = theta_full[0].cpu().numpy()
            finite = np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(theta).all()
            rows.append({
                "frame_no": k + 2,
                "status": "PASS" if finite else "FAIL",
                "pos": pos, "vel": vel, "theta": theta,
                "proc_time_s": proc_time_s,
            })
            if not finite:
                log("  frame %d: non-finite state, stopping rollout (ERR_ROLLOUT_DIVERGED)" % (k + 2))
                break
    return rows


def write_output_csv(rows, n_vert, n_edge, out_path):
    pos_cols = []
    for i in range(1, n_vert + 1):
        pos_cols += ["dlo_d%02d_x" % i, "dlo_d%02d_y" % i, "dlo_d%02d_z" % i]
    vel_cols = []
    for i in range(1, n_vert + 1):
        vel_cols += ["dlo_d%02d_vx" % i, "dlo_d%02d_vy" % i, "dlo_d%02d_vz" % i]
    theta_cols = ["dlo_e%02d_theta" % i for i in range(1, n_edge + 1)]
    header = ["frame_no", "status", "proc_time_s"] + pos_cols + vel_cols + theta_cols

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            row = [r["frame_no"], r["status"], "%.6g" % r["proc_time_s"]]
            row += ["%.6g" % v for v in r["pos"].ravel().tolist()]
            row += ["%.6g" % v for v in r["vel"].ravel().tolist()]
            row += ["%.6g" % v for v in r["theta"].ravel().tolist()]
            writer.writerow(row)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dlo_type_config", required=True,
                         help="data_set/<dlo_traj_tag>/dlo_type_config.json")
    parser.add_argument("--checkpoint", required=True, help="save_model/<tag>_<step>.pth")
    parser.add_argument("--episode_pickle", required=True,
                         help="data_set/<tag>/{train,eval}/<episode>.pkl -- the SAME (T,3,n_vert) "
                              "pickle format prep_deform_dataset.py writes for training")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    # NOT cuda by default despite it being available: "evaluation_numpy" mode's
    # non_linear_opt_theta_full() builds its theseus optimization objective/layer
    # fresh from numpy-converted (CPU) tensors every call, which throws a device
    # mismatch ("kb ... inconsistent with objective's expected (cuda, ...)") the
    # instant self.device is cuda -- confirmed reproducible, likely never exercised
    # by the authors themselves since their own training entry point hardcoded
    # device="cpu" unconditionally (see train_DEFORM.py's own patched comment).
    # A single-episode autoregressive rollout is cheap enough that CPU is fine here
    # regardless -- training (batch=32, theseus in "train" mode) is where the GPU
    # patch actually matters.
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    deform_sim, n_vert, n_edge = build_model(args.dlo_type_config, args.checkpoint, device)
    vertices = load_episode_pickle(args.episode_pickle)
    log("Loaded episode %s: %d frames, n_vert=%d" % (args.episode_pickle, vertices.size(0), n_vert))

    rows = run_episode(deform_sim, n_vert, n_edge, vertices, device, max_frames=args.max_frames)
    write_output_csv(rows, n_vert, n_edge, args.output_csv)
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    log("Wrote %d frame(s) (%d pass) to %s" % (len(rows), n_pass, args.output_csv))


if __name__ == "__main__":
    main()
