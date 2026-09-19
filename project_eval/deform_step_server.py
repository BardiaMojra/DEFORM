#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deform_step_server.py

DEFORM as a persistent, state-settable stepper, for the dynamics-informed perception prior
(dlo_data_001/dev/studies/dyn_prior_tracking). deform_infer.py rolls one whole episode from its
own pickle; this exposes the SAME recurrence one step at a time, from a state the caller supplies,
so a tracker can hand DEFORM its current node chain and get back the predicted next one.

Why a server and not a call per step: building the model and loading the checkpoint costs far more
than a step, and m_u0 / theta_full / the previous vertex array must stay continuous across steps
(deform_infer.py's parallel-transport recurrence) -- restarting would silently reset the material
frame every frame.

Protocol, one command per line on stdin, one reply line on stdout (all floats plain text):

  SET <3*n_vert verts> <3*n_vert vels>   set the state; replies OK
  STEP <n> <12 boundary floats>          advance n steps holding that boundary
                                         (nodes 0, 1, n_vert-2, n_vert-1, xyz each);
                                         replies OK <3*n_vert predicted verts>
  QUIT                                   replies BYE and exits

Anything else replies ERR <reason>. A non-finite state replies ERR nonfinite and keeps the last
good state, so the driver can fall back to its own prediction for that frame.

Runs in DEFORM's own venv (DEFORM_PYTHON / <repo>/.venv), like deform_infer.py.
"""

import os

# Tiny model (n_vert = 10) and "evaluation_numpy" mode converts to numpy every step, so the BLAS
# and torch thread pools -- 20 threads each on this box -- spend more time synchronizing than
# computing. Measured 2026-09-19, 600 frames of t004_d003_v070: 12.3 s at the default 20 threads
# against 4.1 s at 1 (2 threads 4.3 s, 4 threads 5.2 s), with node coordinates bit-identical, so
# this is speed only. Must run BEFORE numpy/torch are imported, which is why it sits up here.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("DEFORM_TORCH_THREADS", "1"))

import argparse
import sys

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deform_infer import INIT_DIRECTION, build_model            # noqa: E402
from util import computeEdges                                   # noqa: E402


class Stepper(object):
    """One episode's worth of DEFORM state, advanced one step at a time."""

    def __init__(self, dlo_type_config, checkpoint, device):
        self.device = torch.device(device)
        self.sim, self.n_vert, self.n_edge = build_model(dlo_type_config, checkpoint, self.device)
        self.dt = self.sim.dt
        self.clamped_selection = torch.tensor((0, 1, self.n_vert - 2, self.n_vert - 1))
        self.clamped_index = torch.zeros(self.n_vert)
        self.clamped_index[self.clamped_selection] = 1.0
        self.init_direction = INIT_DIRECTION.to(self.device).unsqueeze(dim=0)
        self.sim.m_restWprev, self.sim.m_restWnext, self.sim.learned_pmass = \
            self.sim.Rod_Init(1, self.init_direction, self.sim.m_restEdgeL, self.clamped_index)
        self.vert = self.vel = self.m_u0 = self.theta_full = None

    def set_state(self, verts, vels):
        """Adopt an externally supplied state (e.g. a tracker's node chain and its velocity).
        m_u0 is parallel-transported from the previous first edge when there is one, and seeded
        from init_direction when there is not -- the same two cases deform_infer.py's k==0 and
        k>=1 branches handle."""
        new = torch.tensor(verts, dtype=torch.float32, device=self.device).view(1, self.n_vert, 3)
        if self.vert is None:
            self.m_u0 = self.sim.DEFORM_func.compute_u0(
                computeEdges(new)[:, 0].float(), self.init_direction[:, 0])
            self.theta_full = torch.zeros(1, self.n_edge, device=self.device)
        else:
            self.m_u0 = self.sim.DEFORM_func.parallelTransportFrame(
                computeEdges(self.vert)[:, 0], computeEdges(new)[:, 0], self.m_u0)
        self.vert = new
        self.vel = torch.tensor(vels, dtype=torch.float32, device=self.device).view(1, self.n_vert, 3)

    def step(self, n_steps, boundary):
        """n_steps of dt, holding `boundary` (4x3) fixed at the clamped nodes. Returns the
        predicted vertices as a flat list."""
        b = torch.tensor(boundary, dtype=torch.float32, device=self.device).view(1, 4, 3)
        with torch.no_grad():
            for _ in range(int(n_steps)):
                prev = self.vert
                pred, self.vel, self.theta_full = self.sim(
                    self.vert, self.vel, self.init_direction, self.clamped_index, self.m_u0,
                    b, self.clamped_selection, self.theta_full, mode="evaluation_numpy")
                if not torch.isfinite(pred).all():
                    self.vert = prev                       # keep the last good state
                    raise ValueError("nonfinite")
                self.m_u0 = self.sim.DEFORM_func.parallelTransportFrame(
                    computeEdges(prev)[:, 0], computeEdges(pred)[:, 0], self.m_u0)
                self.vert = pred
        return self.vert[0].cpu().numpy().reshape(-1)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dlo_type_config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    st = Stepper(a.dlo_type_config, a.checkpoint, a.device)
    sys.stderr.write("[deform_step_server] ready: n_vert=%d dt=%g device=%s\n"
                     % (st.n_vert, st.dt, a.device))
    sys.stderr.flush()
    print("READY %d %g" % (st.n_vert, st.dt))
    sys.stdout.flush()

    fmt = "%.9g".__mod__
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        cmd, args = parts[0].upper(), parts[1:]
        try:
            if cmd == "QUIT":
                print("BYE")
                sys.stdout.flush()
                return 0
            if cmd == "SET":
                n = 3 * st.n_vert
                if len(args) != 2 * n:
                    raise ValueError("SET wants %d values, got %d" % (2 * n, len(args)))
                v = np.array(args, dtype=np.float64)
                st.set_state(v[:n], v[n:])
                print("OK")
            elif cmd == "STEP":
                if st.vert is None:
                    raise ValueError("STEP before SET")
                if len(args) != 13:
                    raise ValueError("STEP wants n + 12 boundary values, got %d" % len(args))
                out = st.step(int(args[0]), np.array(args[1:], dtype=np.float64))
                print("OK " + " ".join(fmt(x) for x in out))
            else:
                raise ValueError("unknown command %r" % cmd)
        except Exception as exc:                            # never die on one bad frame
            print("ERR %s" % str(exc).replace("\n", " "))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
