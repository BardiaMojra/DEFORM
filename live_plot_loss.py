#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_plot_loss.py

Live view of train_DEFORM.py's own loss_record/*.pkl files (train_loss_<tag>.pkl,
train_epoch_<tag>.pkl, eval_loss_<tag>.pkl, eval_epoch_<tag>.pkl -- see its save_pickle()
calls) -- one figure per dlo_traj_tag, re-read and redrawn every --interval seconds while
training runs, and saved to <tag>/save_model/<tag>_loss.png on every refresh (the
session's own per-tag models/ tree, alongside that tag's save_model/<tag>_<step>.pth
checkpoints -- same convention dataset_quartermaster.py's MODELS_DIR/der's own
models/DER/<tag>/<tag>.json already use, one method-owned artifact store per method) --
so there's always an on-disk copy, not just the live window.

Usage: ./live_plot_loss.py [--dlo_types d003_v050 d003_v070 ...] [--interval 3] [--log]
  (no --dlo_types: auto-discovers every "d0NN_..." tag with a loss record on disk)
"""

import argparse
import glob
import os
import pickle
import re
import time

import matplotlib
import matplotlib.pyplot as plt

MODELS_DIR = os.path.expanduser("~/bags/test_session_001/models")

_TAG_RE = re.compile(r'^d\d{3}(_v\d{3}|_all)$')

# dataviz skill's validated categorical palette, slots 1 (blue) and 2 (orange) --
# fixed per-series identity, not cycled
TRAIN_COLOR = "#2a78d6"
EVAL_COLOR = "#eb6834"


def loss_dir(tag):
    return os.path.join(MODELS_DIR, "DEFORM", tag, "loss_record")


def plots_dir(tag):
    return os.path.join(MODELS_DIR, "DEFORM", tag, "save_model")


def discover_tags():
    tags = set()
    for path in glob.glob(os.path.join(MODELS_DIR, "DEFORM", "*", "loss_record",
                                        "train_loss_*.pkl")):
        name = os.path.basename(path)[len("train_loss_"):-len(".pkl")]
        if _TAG_RE.match(name):
            tags.add(name)
    return sorted(tags)


def load_pickle_if_exists(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None  # tolerate a read racing train_DEFORM.py's own write


class TagPlot:
    def __init__(self, tag, log_scale):
        self.tag = tag
        os.makedirs(plots_dir(tag), exist_ok=True)
        self.out_path = os.path.join(plots_dir(tag), "%s_loss.png" % tag)
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.ax.set_title("%s -- DEFORM training loss" % tag)
        self.ax.set_xlabel("update step")
        self.ax.set_ylabel("loss")
        if log_scale:
            self.ax.set_yscale("log")
        self.ax.grid(True, alpha=0.25, linewidth=0.5)
        (self.train_line,) = self.ax.plot([], [], color=TRAIN_COLOR, linewidth=2,
                                           label="train loss")
        (self.eval_line,) = self.ax.plot([], [], color=EVAL_COLOR, linewidth=2,
                                          marker="o", markersize=5, label="eval loss")
        self.ax.legend(loc="upper right", frameon=False)
        self.fig.tight_layout()

    def refresh(self):
        tag_loss_dir = loss_dir(self.tag)
        train_epochs = load_pickle_if_exists(
            os.path.join(tag_loss_dir, "train_epoch_%s.pkl" % self.tag)) or []
        train_losses = load_pickle_if_exists(
            os.path.join(tag_loss_dir, "train_loss_%s.pkl" % self.tag)) or []
        eval_epochs = load_pickle_if_exists(
            os.path.join(tag_loss_dir, "eval_epoch_%s.pkl" % self.tag)) or []
        eval_losses = load_pickle_if_exists(
            os.path.join(tag_loss_dir, "eval_loss_%s.pkl" % self.tag)) or []

        n = min(len(train_epochs), len(train_losses))
        self.train_line.set_data(train_epochs[:n], train_losses[:n])
        m = min(len(eval_epochs), len(eval_losses))
        self.eval_line.set_data(eval_epochs[:m], eval_losses[:m])

        if n or m:
            self.ax.relim()
            self.ax.autoscale_view()
        latest_step = max(train_epochs[n - 1] if n else 0, eval_epochs[m - 1] if m else 0)
        self.ax.set_title("%s -- DEFORM training loss  (last step=%d)" % (self.tag, latest_step))

        self.fig.canvas.draw_idle()
        self.fig.savefig(self.out_path, dpi=110)
        return n, m


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dlo_types", nargs="+", default=None,
                         help="tags to watch, e.g. d003_v050 d003_all -- default: "
                              "auto-discover every tag with a loss record on disk")
    parser.add_argument("--interval", type=float, default=3.0,
                         help="refresh period in seconds [%(default)s]")
    parser.add_argument("--log", action="store_true", help="log-scale y-axis")
    args = parser.parse_args()

    tags = args.dlo_types or discover_tags()
    if not tags:
        print("No <tag>/loss_record/train_loss_*.pkl found yet under %s/DEFORM -- nothing "
              "to watch (this fills in once train_DEFORM.py starts logging steps)."
              % MODELS_DIR)
        return
    print("Watching %d tag(s): %s" % (len(tags), ", ".join(tags)))
    print("Saving to %s/DEFORM/<tag>/save_model/<tag>_loss.png every %.1fs -- Ctrl+C to stop."
          % (MODELS_DIR, args.interval))

    plt.ion()
    plots = [TagPlot(tag, args.log) for tag in tags]
    plt.show(block=False)

    try:
        while True:
            for p in plots:
                n, m = p.refresh()
                print("  [%s] train points=%d  eval points=%d" % (p.tag, n, m), end="\r")
            plt.pause(args.interval)
    except KeyboardInterrupt:
        print("\nStopped -- last snapshot already saved to each <tag>_loss.png.")


if __name__ == "__main__":
    main()
