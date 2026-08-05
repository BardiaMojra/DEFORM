import concurrent.futures
import glob
import os
import argparse
import re
import shutil
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import pandas as pd
from tqdm import tqdm
from DEFORM_func import DEFORM_func
from DEFORM_sim import DEFORM_sim
from util import computeLengths, computeEdges, compute_u0, parallelTransportFrame
import pickle
import random
import torch.nn as nn

random.seed(0)
torch.manual_seed(0)
"initial release of DEFORM"

# ── ANSI colours -- same palette/names as dynamic_dlo_evaluation's batch_eval_dynamic.py,
# kept consistent across the whole harness ──────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GRN = "\033[32m"
YLW = "\033[33m"
CYN = "\033[36m"
BRED = "\033[91m"
BGRN = "\033[92m"
BYLW = "\033[93m"
BCYN = "\033[96m"
BAR_FULL, BAR_EMPTY = "█", "░"
_TTY = os.isatty(1)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def c(text, *codes):
    """No-op (returns plain text) when stdout isn't a real terminal, so redirected/piped
    output and the log file below never end up with raw escape codes in them."""
    if not codes or not _TTY:
        return str(text)
    return "".join(codes) + str(text) + RST


def term_cols(default=80):
    """Current terminal width, re-queried on every call (not cached) so bars stay correctly
    sized across a resize instead of freezing at whatever the width was when training
    started. Falls back to `default` when stdout isn't a real terminal (e.g. redirected to
    a file/pipe/nohup)."""
    return shutil.get_terminal_size((default, 24)).columns


def pbar(pct, width=None, color=BGRN):
    if width is None:
        # scale the bar itself to the terminal instead of a fixed 20 chars, so it can't
        # overflow (and wrap, breaking in-place redraw) in a narrow window
        width = max(10, min(30, term_cols() // 4))
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    inner = c(BAR_FULL * filled, color) + c(BAR_EMPTY * (width - filled), DIM)
    return "[" + inner + "] " + c("%5.1f%%" % pct, BOLD)


def _default_log(msg, color=None, in_place=False):
    """Fallback log_fn for Train_DeformData/Eval_DeformData when used outside train()'s own
    colorized+file-backed logger (e.g. imported standalone) -- same (msg, color, in_place)
    signature, just prints plainly (in_place is a no-op here -- no log file to keep in sync
    with, so there's nothing gained by overwriting)."""
    print(msg, flush=True)


def make_logger(dlo_type):
    """Additive (not upstream): mirror every print() to logs/train_<DLO_type>.log so
    progress survives a detached/backgrounded run instead of only going to whatever
    terminal happens to be attached -- see deform_notes.md "slow/looks-frozen training"
    writeup. Line-buffered (buffering=1) so a periodic `tail` sees fresh lines without
    waiting on process exit. The log file always gets the plain (un-colored) line -- only
    the terminal copy is colorized -- so grep/tail -f stay readable either way."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "train_%s.log" % dlo_type)
    log_file = open(log_path, "a", buffering=1)
    cursor = {"open": False}   # True while an in-place line is sitting unterminated on the tty

    def log(msg, color=None, in_place=False):
        # msg may already contain inline c(...) color tags (e.g. a colored "[data]" prefix
        # mixed with plain text) -- strip ANSI codes for the file line regardless, so the
        # log file is always plain/grep-able no matter how the caller built the string.
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        term_msg = c(msg, color) if color else msg
        plain_line = "[%s] %s" % (ts, _ANSI_RE.sub("", msg))
        full_term_line = "%s %s" % (c("[%s]" % ts, DIM), term_msg)

        if in_place and _TTY:
            # \r + \x1b[K (erase to end of line) makes each tick overwrite the previous one
            # in place instead of scrolling, and the erase -- not just relying on the new
            # text being >= as long as the old -- means leftover characters don't linger if
            # the terminal was narrowed since the last tick. Re-measure cols on every call
            # (not cached) so a live resize is picked up immediately.
            visible = _ANSI_RE.sub("", full_term_line)
            cols = term_cols()
            if len(visible) > cols - 1:
                full_term_line = visible[:cols - 1]   # plain fallback once colored won't fit
            sys.stdout.write("\r\x1b[K" + full_term_line)
            sys.stdout.flush()
            cursor["open"] = True
        else:
            if cursor["open"]:
                sys.stdout.write("\n")   # close out the in-place line before a normal one
                cursor["open"] = False
            print(full_term_line, flush=True)

        # the log file always gets a full discrete line per call, in_place or not -- so
        # `tail -f logs/train_<DLO_type>.log` still shows every tick even though the tty
        # only ever shows the latest one
        log_file.write(plain_line + "\n")
    return log, log_path


def find_latest_checkpoint(dlo_type, save_model_dir="save_model"):
    """Mirrors dynamic_dlo_evaluation/dev/deform_adapter.py's own find_latest_checkpoint()
    (same glob + numeric-step-suffix-parse logic -- filenames are "<tag>_<step>.pth", and
    sorting them as plain strings orders "100" before "80") -- reimplemented locally rather
    than imported cross-repo, since this runs inside DEFORM's own pinned venv (torch/
    theseus) and that module pulls in a different script's own dependency chain. Returns
    (path, step) or (None, None)."""
    candidates = glob.glob(os.path.join(save_model_dir, "%s_*.pth" % dlo_type))
    if not candidates:
        return None, None

    def step_of(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    best = max(candidates, key=step_of)
    return best, step_of(best)


def load_pickle_if_exists(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _compute_episode_mu0(rope_verts, length):
    """Runs in a worker process, one call per episode (see Train_DeformData.__init__).
    The bishop-frame recurrence is sequential frame-to-frame WITHIN an episode (each
    depends on the previous one via parallelTransportFrame) so it can't be vectorized
    across time -- but it's fully independent ACROSS episodes (no shared state), so
    farming episodes out to a process pool parallelizes what used to be a single-core
    loop. Returns (z-clipped (T,3,N) array, (T-2,3) mu_0 tensor, window_count) for this
    one episode."""
    torch.set_num_threads(1)   # avoid oversubscription -- N worker processes each also
                                # trying to multithread these (tiny, batch-of-1) ops would
                                # thrash the CPU instead of speeding anything up
    rope_arr = np.array(rope_verts)
    n_frames = len(rope_arr) - 1 - 1
    mu_0_list = torch.zeros(n_frames, 3)
    init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).unsqueeze(dim=0)
    for i in range(n_frames):
        if i == 0:
            vertices = torch.transpose(torch.tensor(rope_arr[i + 1: i + 1 + 1]), 1, 2).float()
            rest_edges = computeEdges(vertices)
            m_u0 = compute_u0(rest_edges.float()[:, 0], init_direction.repeat(1, 1, 1)[:, 0])
            mu_0_list[i] = m_u0
        else:
            previous_vertices = torch.transpose(torch.tensor(rope_arr[i: i + 1]), 1, 2).float()
            current_vertices = torch.transpose(torch.tensor(rope_arr[i + 1: i + 1 + 1]), 1, 2).float()
            previous_edge = computeEdges(previous_vertices)
            current_edges = computeEdges(current_vertices)
            m_u0 = parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0.clone())
            mu_0_list[i] = m_u0

    # z-clip once on the full base array -- equivalent to clipping each of the three
    # offset windows separately (upstream's approach): clipping is elementwise, and
    # previous/vertices/target are just 0/+1/+2-shifted slices of this same array, so
    # every element gets the identical clip either way.
    rope_arr[:, -1] = np.clip(rope_arr[:, -1], a_min=2e-3 + 1e-6, a_max=10000.)
    window_count = max(0, len(rope_arr) - 1 - length)
    return rope_arr, mu_0_list, window_count


class Train_DeformData(Dataset):
    """Additive perf/memory fix (not upstream, see deform_notes.md "OOM / swap-thrashing"
    writeup): upstream eagerly materialized EVERY overlapping time_horizon-length sliding
    window for the whole dataset into self.previous_vertices/vertices/target_vertices/mu_0
    lists at __init__ time -- O(total_frames * time_horizon) memory (each frame duplicated
    into up to time_horizon windows, x3 for the previous/current/target offset copies).
    d003_v050's ~8.07M-frame/78-episode dataset OOM'd this machine (13GB+ RSS, swap-
    thrashing to the point of needing a hard kill) even though d003_v070's smaller 62-episode
    set had stayed just under the ceiling. Fixed by storing each episode's base (T,3,N) array
    ONCE (O(total_frames) memory total) and slicing out each of the three offset windows
    lazily in __getitem__ instead -- previous/vertices/target_vertices are just 0/+1/+2-frame-
    shifted views of the identical underlying sequence, so there's nothing to precompute."""

    def __init__(self, DLO_type, train_set_number, time_horizon, device, log_fn=_default_log):
        super(Train_DeformData, self).__init__()
        '''
        change the root dir based in your dir
        '''
        self.root_dir = "data_set/%s/train/" %DLO_type
        inputs_file_list = glob.glob(self.root_dir + "*")
        self.device = device
        chosen_files = random.choices(inputs_file_list, k=train_set_number)
        length = self.length = time_horizon
        self.episodes = []      # per-episode (T,3,N) numpy array, z-clipped once
        self.episode_mu0 = []   # per-episode (T-2,3) tensor on self.device
        window_counts = []      # windows contributed by each episode, for index mapping

        # Additive perf fix (not upstream): the mu_0 (bishop-frame) recurrence is inherently
        # sequential WITHIN an episode -- each frame depends on the previous one -- so it
        # can't be vectorized across time. But it's fully independent ACROSS episodes (no
        # shared state), so instead of running all of them serially on a single core (see
        # _compute_episode_mu0's own docstring for why it's a worker-process function),
        # farm episodes out to a process pool. Also keeps it off the GPU (compute_u0/
        # parallelTransportFrame are device-agnostic -- see util.py, they read io_u.device
        # dynamically): upstream ran every single-frame step through .to(device), a real
        # GPU transfer + kernel launch for a batch-of-1 op, ~300k times for this project's
        # episode counts/lengths -- pure latency with zero compute benefit from the GPU,
        # and produced no output until the whole precompute finished, so a multi-hour run
        # stuck here was indistinguishable from hung. See deform_notes.md "slow/looks-frozen
        # training" writeup.
        episodes_raw = [pd.read_pickle(r'%s' % str(p)) for p in chosen_files]
        total_frames = sum(len(rv) - 1 - 1 for rv in episodes_raw)
        n_workers = min(len(episodes_raw), max(1, (os.cpu_count() or 2) - 1))
        frames_done = 0
        episodes_done = 0
        precompute_start = time.time()
        log_fn(c("[data]", BCYN) + " precomputing mu_0 (bishop frames) for %d train episodes, "
               "%d frames total, on CPU x%d workers" % (train_set_number, total_frames, n_workers))

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_compute_episode_mu0, rv, length) for rv in episodes_raw]
            for fut in concurrent.futures.as_completed(futures):
                rope_arr, mu_0_list, window_count = fut.result()
                self.episodes.append(rope_arr)
                self.episode_mu0.append(mu_0_list)
                window_counts.append(window_count)

                episodes_done += 1
                frames_done += len(mu_0_list)
                pct = 100.0 * frames_done / max(total_frames, 1)
                elapsed = time.time() - precompute_start
                rate = frames_done / elapsed if elapsed > 0 else 0.0
                eta = (total_frames - frames_done) / rate if rate > 0 else float("nan")
                log_fn(c("[data]", BCYN) + " %s frames %s eps %s  %s frames/s  ETA %s"
                       % (pbar(pct), c("%d/%d" % (frames_done, total_frames), BOLD),
                          c("%d/%d" % (episodes_done, train_set_number), BOLD),
                          c("%.0f" % rate, CYN), c("%.0fs" % eta, YLW)),
                       in_place=(episodes_done != train_set_number))

        log_fn(c("[data]", BCYN) + " mu_0 precompute done in " + c("%.1fs" % (time.time() - precompute_start), BGRN))

        self._cum_windows = np.concatenate(([0], np.cumsum(window_counts)))

    def __len__(self):
        return int(self._cum_windows[-1])

    def __getitem__(self, index):
        # searchsorted(..., side="right") - 1 maps a flat window index back to which episode
        # it falls in (cum_windows[ep] <= index < cum_windows[ep+1]) and the in-episode frame
        # offset -- see the class docstring for why windows aren't precomputed/stored directly.
        ep_idx = int(np.searchsorted(self._cum_windows, index, side="right") - 1)
        i = index - int(self._cum_windows[ep_idx])
        rope_arr = self.episodes[ep_idx]
        length = self.length
        previous_vertices = torch.transpose(
            torch.tensor(rope_arr[i:i + length]).to(self.device), 1, 2).float()
        vertices = torch.transpose(
            torch.tensor(rope_arr[i + 1:i + 1 + length]).to(self.device), 1, 2).float()
        target_vertices = torch.transpose(
            torch.tensor(rope_arr[i + 2:i + 2 + length]).to(self.device), 1, 2).float()
        mu_0 = self.episode_mu0[ep_idx][i:i + length].to(self.device)
        return previous_vertices.clone().detach(), vertices.clone().detach(), target_vertices.clone().detach(), mu_0.clone().detach()

class Eval_DeformData(Dataset):
    def __init__(self, DLO_type, eval_set_number, time_horizon, device, log_fn=_default_log):
        super(Eval_DeformData, self).__init__()
        self.root_dir = "data_set/%s/eval/" %DLO_type
        inputs_file_list = glob.glob(self.root_dir + "*")
        self.device = device
        chosen_files = random.choices(inputs_file_list, k=eval_set_number)
        log_fn(c("[data]", BCYN) + " loading %d eval episodes" % eval_set_number)
        bar = tqdm(chosen_files)
        length = time_horizon

        self.previous_vertices = []
        self.vertices = []
        self.target_vertices = []
        self.gt_m0 = []
        for rope_data in bar:
            rope_verts = pd.read_pickle(r'%s' % str(rope_data))
            self.previous_vertices.append(rope_verts[:0 + length])
            self.vertices.append(rope_verts[1:1 + length])
            self.target_vertices.append(rope_verts[2:2 + length])

        self.previous_vertices = np.array(self.previous_vertices)
        self.previous_vertices[:, :, 2] = np.clip(self.previous_vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

        self.vertices = np.array(self.vertices)
        self.vertices[:, :, 2] = np.clip(self.vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

        self.target_vertices = np.array(self.target_vertices)
        self.target_vertices[:, :, 2] = np.clip(self.target_vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        previous_vertices = torch.transpose(torch.tensor(np.array(self.previous_vertices[index])).to(self.device), 1, 2).float()
        vertices = torch.transpose(torch.tensor(np.array(self.vertices[index])).to(self.device), 1, 2).float()
        target_vertices = torch.transpose(torch.tensor(np.array(self.target_vertices[index])).to(self.device),1, 2).float()
        return previous_vertices, vertices, target_vertices

def save_pickle(data, myfile):
    with open(myfile, "wb") as f:
        pickle.dump(data, f)

def train(DLO_type, train_set_number, eval_set_number, train_time_horizon, eval_time_horizon, batch, DEFORM_func, DEFORM_sim, device, max_update_steps=None, on_existing_checkpoint="ask"):
    log, log_path = make_logger(DLO_type)
    log(c("=== starting DEFORM training: DLO_type=%s train_time_horizon=%d eval_time_horizon=%d "
        "max_update_steps=%s ===" % (DLO_type, train_time_horizon, eval_time_horizon, max_update_steps), BOLD, BCYN))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        gpu_idx = int(str(device).split(":")[1]) if ":" in str(device) else torch.cuda.current_device()
        log(c("GPU:", BGRN) + " %s (%s)" % (torch.cuda.get_device_name(gpu_idx), device))
    else:
        log(c("WARNING: running on CPU", BRED, BOLD) + " -- this will be far slower than GPU")
    log(c("logging to %s" % log_path, DIM))
    '''
    Dataset Loading
    '''
    train_dataset = Train_DeformData(DLO_type, train_set_number, train_time_horizon, device, log_fn=log)
    eval_dataset = Eval_DeformData(DLO_type, eval_set_number, eval_time_horizon, device, log_fn=log)
    eval_data_len = len(eval_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    '''
    pre set for DLO:
    n_vert: number of vertices
    n_edge: number of edges
    '''
    if DLO_type == "DLO1":
        n_vert = 13
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        '''pbd itr: inextensibility enforcement loop. number > 5 should able to satisfy the condition'''
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert-1, pbd_iter=10, device=device)
        '''
        rest_vert: undeformed states. Dependent on wires. In simulation, it is typically initialized with a straight wire that is segemented equally.
        '''
        rest_vert = (torch.tensor(((0.893471, -0.133465, 0.018059),
                                   (0.880771, -0.119666, 0.017733),
                                   (0.791946, -0.084258, 0.009944),
                                   (0.680462, -0.102366, 0.018528),
                                   (0.590795, -0.144219, 0.021808),
                                   (0.494905, -0.156384, 0.017816),
                                   (0.396916, -0.143114, 0.021549),
                                   (0.299291, -0.148755, 0.014955),
                                   (0.200583, -0.146497, 0.01727),
                                   (0.09586, -0.142385, 0.016456),
                                   (-0.000782, -0.147084, 0.016081),
                                   (-0.071514, -0.17382, 0.015446),
                                   (-0.094659, -0.186181, 0.012403)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        # vis_rest_vert = torch.Tensor.numpy(rest_vert.to('cpu'))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        # ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        # ax.set_xlim(-.5, 1.)
        # ax.set_ylim(-1, .5)
        # ax.set_zlim(0, 1.)
        # plt.legend()
        # plt.show()
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        '''
        stiffness of bending and twisting: dependent on wires. 
        '''
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(2e-5 * torch.ones((1, n_edge), device=device))
        '''
        load trained model. comment following when train first time.
        '''
        # DEFORM_sim.load_state_dict(torch.load("save_model/DLO1_0.pth"))


    elif DLO_type == "DLO2":
        n_vert = 12
        """clamped start and end"""
        clamped_index = torch.zeros(n_vert)
        clamped_selection = torch.tensor((0, 1, -2, -1))
        clamped_index[clamped_selection] = torch.tensor((1.))
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((0.725862, -0.196132, 0.013556),
                                   (0.719875, -0.165722, 0.009538),
                                   (0.697891, -0.068908, 0.013519),
                                   (0.642622, 0.006184, 0.008588),
                                   (0.559875, 0.054215, 0.008419),
                                   (0.468611, 0.075446, 0.009509),
                                   (0.376396, 0.07341, 0.010467),
                                   (0.289067, 0.041016, 0.008857),
                                   (0.214187, -0.019351, 0.017508),
                                   (0.170766, -0.099437, 0.006587),
                                   (0.161013, -0.200349, 0.007841),
                                   (0.161086, -0.228518, 0.007807)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        # vis_rest_vert = torch.Tensor.numpy(rest_vert.to('cpu'))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        # ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        # ax.set_xlim(-.5, 1.)
        # ax.set_ylim(-1, .5)
        # ax.set_zlim(0, 1.)
        # plt.legend()
        # plt.show()
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(5e-4 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(3e-5 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))


    elif DLO_type == "DLO3":
        n_vert = 12
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((0.704214, -0.046593, 0.020496),
                                   (0.712317, -0.078647, 0.025723),
                                   (0.727923, -0.180886, 0.032423),
                                   (0.702225, -0.273037, 0.031611),
                                   (0.634172, -0.347682, 0.027974),
                                   (0.53685, -0.373692, 0.035285),
                                   (0.430097, -0.379901, 0.029374),
                                   (0.337156, -0.366995, 0.030347),
                                   (0.258182, -0.311241, 0.021588),
                                   (0.2192, -0.209264, 0.022677),
                                   (0.199719, -0.120685, 0.019185),
                                   (0.190919, -0.082036, 0.018718)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)

        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1),
                               rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        # ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        # ax.set_xlim(-.5, 1.)
        # ax.set_ylim(-1, .5)
        # ax.set_zlim(0, 1.)
        # plt.legend()
        # plt.show()
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(8e-4 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))

    elif DLO_type == "DLO4":
        n_vert = 12
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((0.920048, -0.055981, 0.021565),
                                   (0.899931, -0.068992, 0.01902),
                                   (0.800974, -0.091743, 0.014608),
                                   (0.705552, -0.123076, 0.01362),
                                   (0.604248, -0.108163, 0.014673),
                                   (0.506436, -0.115882, 0.014896),
                                   (0.408701, -0.101447, 0.011098),
                                   (0.313047, -0.089462, 0.007723),
                                   (0.231587, -0.10213, 0.007496),
                                   (0.159452, -0.16659, 0.017735),
                                   (0.070979, -0.178956, 0.01519),
                                   (0.062259, -0.202573, 0.013681)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        # vis_rest_vert = torch.Tensor.numpy(rest_vert.to('cpu'))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        # ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        # ax.set_xlim(-.5, 1.)
        # ax.set_ylim(-1, .5)
        # ax.set_zlim(0, 1.)
        # plt.legend()
        # plt.show()
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(8e-5 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))


    elif DLO_type == "DLO5":
        n_vert = 12
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((1.081046, -0.394121, 0.023486),
                                   (1.056035, -0.384787, 0.023537),
                                   (0.961936, -0.393094, 0.023699),
                                   (0.859469, -0.389925, 0.021839),
                                   (0.76015, -0.379264, 0.022267),
                                   (0.658647, -0.37746, 0.016315),
                                   (0.559766, -0.388966, 0.022272),
                                   (0.457995, -0.40327, 0.021107),
                                   (0.355937, -0.394938, 0.01998),
                                   (0.251256, -0.40417, 0.020634),
                                   (0.160682, -0.424936, 0.021145),
                                   (0.140942, -0.420546, 0.020377)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        vis_rest_vert = torch.Tensor.numpy(rest_vert.to('cpu'))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        # ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        # ax.set_xlim(-.5, 1.)
        # ax.set_ylim(-1, .5)
        # ax.set_zlim(0, 1.)
        # plt.legend()
        # plt.show()
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(8e-5 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))

    else:
        # Data-driven fallback for any DLO_type not in the hardcoded list above: read
        # n_vert/rest_vert/initial stiffness from data_set/<DLO_type>/dlo_type_config.json
        # (written by ~/git/dynamic_dlo_evaluation/dev/prep_deform_dataset.py) instead of
        # requiring a new hand-written elif per trajectory tag. See deform_notes.md
        # (~/ros_ws/docs_n_papers/) for the fuller design rationale.
        import json
        cfg_path = os.path.join("data_set", DLO_type, "dlo_type_config.json")
        if not os.path.isfile(cfg_path):
            raise ValueError("No matching DLO type, and no data-driven config at %s" % cfg_path)
        with open(cfg_path, "r") as f:
            dlo_cfg = json.load(f)
        n_vert = dlo_cfg["n_vert"]
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)

        # rest_vert here is already in this project's own robot/arm frame (straight from
        # dlo_perception's own dlo_dyn_xyz, frame 0 of the first episode processed) -- no
        # axis remap needed, unlike the DLO1-5 blocks above which remap from the authors'
        # own mocap capture frame.
        rest_vert = torch.tensor(dlo_cfg["rest_vert"], device=device).unsqueeze(dim=0)
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(
            dlo_cfg["bend_stiffness_init"] * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(
            dlo_cfg["twist_stiffness_init"] * torch.ones((1, n_edge), device=device))

    """clamped start edge and end edge"""
    clamped_index = torch.zeros(n_vert)
    clamped_selection = torch.tensor((0, 1, -2, -1))
    clamped_index[clamped_selection] = torch.tensor((1.))

    # Resume/restart (additive, not upstream -- upstream always trained from a fresh random
    # init on every invocation, silently overwriting save_model/<tag>_<step>.pth and
    # loss_record/*_<tag>.pkl at matching step numbers on any restart, with no way to
    # continue a stopped run). See deform_notes.md.
    resume_step = 0
    resumed_losses = resumed_epochs = resumed_eval_losses = resumed_eval_epochs = None
    existing_ckpt, existing_step = find_latest_checkpoint(DLO_type)
    if existing_ckpt is not None:
        log(c("[resume]", BYLW) + " found existing checkpoint %s (step=%d)"
            % (existing_ckpt, existing_step))
        decision = on_existing_checkpoint
        if decision == "ask":
            try:
                while True:
                    ans = input(c("  Resume training from step %d, or start over? "
                                   "[r]esume / [s]tart over: " % existing_step, BOLD)).strip().lower()
                    if ans in ("r", "resume"):
                        decision = "resume"
                        break
                    if ans in ("s", "start", "fresh", "start over"):
                        decision = "fresh"
                        break
                    print(c("  please answer 'r' (resume) or 's' (start over)", DIM))
            except EOFError:
                decision = "fresh"
                log(c("[resume]", BRED) + " no interactive stdin available -- defaulting to "
                    "start-over (pass --on_existing_checkpoint=resume to resume non-interactively)")
        if decision == "resume":
            DEFORM_sim.load_state_dict(torch.load(existing_ckpt, map_location=device))
            resume_step = existing_step
            resumed_losses = load_pickle_if_exists("loss_record/train_loss_%s.pkl" % DLO_type)
            resumed_epochs = load_pickle_if_exists("loss_record/train_epoch_%s.pkl" % DLO_type)
            resumed_eval_losses = load_pickle_if_exists("loss_record/eval_loss_%s.pkl" % DLO_type)
            resumed_eval_epochs = load_pickle_if_exists("loss_record/eval_epoch_%s.pkl" % DLO_type)
            log(c("[resume]", BGRN) + " resuming from step %d (loss history %s)"
                % (resume_step, "restored" if resumed_losses is not None else "not found, starting empty"))
        else:
            log(c("[resume]", YLW) + " starting fresh -- ignoring existing checkpoint (this will "
                "overwrite save_model/loss_record files at matching step numbers as training proceeds)")
    if max_update_steps is not None and resume_step >= max_update_steps:
        log(c("[train] checkpoint is already at/past max_update_steps=%d -- nothing to do"
            % max_update_steps, BOLD, BGRN))
        return

    """learning setup"""
    loss_func = torch.nn.L1Loss()
    network_lr = 1e-4
    lr_scale = 0.1
    parameters_to_update = [
        {"params": DEFORM_sim.integration_ratio, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.velocity_ratio, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.rest_vert, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.mocap_mass, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.DEFORM_func.bend_stiffness, "lr": 1e-11 * lr_scale},
        {"params": DEFORM_sim.DEFORM_func.twist_stiffness, "lr": 1e-11 * lr_scale},
        {"params": DEFORM_sim.vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.delta_vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.delta_vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.fc.parameters(), "lr": network_lr * lr_scale},
    ]
    # Create an optimizer with different learning rates
    optimizer = torch.optim.SGD(parameters_to_update)

    """record steps and losses"""
    epochs = []
    losses = []
    eval_epochs = []
    eval_losses = []

    """evaluate the model after each 20 training iterations"""
    train_epoch = 100
    save_steps = 0
    evaluate_period = 20
    save_period = 20
    update_steps = 0
    train_start = time.time()

    def progress_suffix(steps):
        """Colored progress-bar + ETA tag appended to per-step log lines -- only meaningful
        when max_update_steps is set (an unbounded run has no known total)."""
        if not max_update_steps:
            return ""
        pct = 100.0 * steps / max_update_steps
        elapsed = time.time() - train_start
        rate = steps / elapsed if elapsed > 0 else 0.0
        eta = (max_update_steps - steps) / rate if rate > 0 else float("nan")
        return "  %s  ETA %s" % (pbar(pct), c("%.0fs" % eta, YLW))

    for epoch in range(train_epoch):
        bar = tqdm(train_data_loader)
        for data in bar:
            if save_steps % evaluate_period == 0:
                log(c("[eval]", BYLW) + " starting eval @ step=%d%s" % (update_steps, progress_suffix(update_steps)))
                eval_batch = eval_set_number
                part_eval = eval_set_number
                eval_set, test_set = torch.utils.data.random_split(eval_dataset, [part_eval, eval_data_len - part_eval])
                eval_data_loader = DataLoader(eval_set, batch_size=eval_batch, shuffle=True, drop_last=True)
                torch.save(DEFORM_sim.state_dict(),os.path.join("save_model/", "%s_%s.pth" % (DLO_type, str(update_steps))))
                eval_loss = 0
                eval_bar = tqdm(eval_data_loader)
                """evaluation: for faster evaluation, use DEFORM_sim(..., mode = "evaluation_numpy")"""
                with torch.no_grad():
                    eval_time = 0
                    for eval_data in eval_bar:
                        init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).to(device).unsqueeze(dim=0)
                        eval_previous_vertices, eval_vertices, eval_target_vertices = eval_data
                        inputs = eval_target_vertices[:, :, clamped_selection]
                        """
                        initialize all theta = 0
                        """
                        theta_full = torch.zeros(eval_batch, n_vert - 1).to(device)
                        for traj_num in range(eval_target_vertices.size()[1]):
                            with torch.no_grad():
                                if traj_num == 0:
                                    rest_edges = computeEdges(eval_vertices[:, traj_num])
                                    m_u0 = DEFORM_func.compute_u0(rest_edges[:, 0].float(), init_direction.repeat(eval_batch, 1, 1)[:, 0])
                                    current_v = (eval_vertices[:, traj_num] - eval_previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                                    m_restEdgeL = DEFORM_sim.m_restEdgeL.repeat(eval_batch, 1)
                                    DEFORM_sim.m_restWprev, DEFORM_sim.m_restWnext, DEFORM_sim.learned_pmass = DEFORM_sim.Rod_Init(eval_batch, init_direction.repeat(eval_batch, 1, 1), m_restEdgeL, clamped_index)
                                    init_pred_vert_0, current_v, theta_full = DEFORM_sim(eval_vertices[:, traj_num], current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")
                                    traj_loss = loss_func(init_pred_vert_0, eval_target_vertices[:, traj_num].float())
                                    eval_loss += traj_loss

                                    """visualization: store image into local file for visualization"""
                                    # init_vis_vert = torch.Tensor.numpy(init_pred_vert_0.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(init_vis_vert[0, :, 0], init_vis_vert[0, :, 1], init_vis_vert[0, :, 2], label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                                if traj_num == 1:
                                    previous_edge = computeEdges(eval_previous_vertices[:, traj_num])
                                    current_edges = computeEdges(init_pred_vert_0)
                                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                                    pred_vert, current_v, theta_full = DEFORM_sim(init_pred_vert_0, current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")
                                    vert = init_pred_vert_0.clone()
                                    traj_loss = loss_func(pred_vert, eval_target_vertices[:, traj_num])
                                    eval_loss += traj_loss

                                    # vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2], label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                                if traj_num >= 2:
                                    previous_vert = vert.clone()
                                    vert = pred_vert.clone()
                                    current_v = current_v.clone()
                                    m_u0 = m_u0.clone()
                                    previous_edge = computeEdges(previous_vert)
                                    current_edges = computeEdges(vert)
                                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0],m_u0)
                                    pred_vert, current_v, theta_full = DEFORM_sim(vert, current_v,init_direction.repeat(eval_batch, 1, 1),clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")
                                    traj_loss = loss_func(pred_vert, eval_target_vertices[:, traj_num])
                                    eval_loss += traj_loss

                                    # vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2],label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                            eval_time += 1
                eval_losses.append(eval_loss.cpu().detach().numpy() / (eval_time_horizon * part_eval // eval_batch))
                log(c("[eval]", BYLW) + " step=%d eval_loss=%s  (history: %s)"
                    % (update_steps, c("%.6f" % eval_losses[-1], BOLD), eval_losses))
                eval_epochs.append(update_steps)
                """save loss into local files. to do: tensor board"""
                save_pickle(eval_losses, "loss_record/eval_loss_%s.pkl" % (DLO_type))
                save_pickle(eval_epochs, "loss_record/eval_epoch_%s.pkl" % (DLO_type))
            """"""

            """training"""
            theta_full = torch.zeros(batch, n_vert - 1).to(device)
            traj_loss_record = 0

            if train_time_horizon == 1:
                previous_vertices, vertices, target_vertices, m_u0 = data
                inputs = target_vertices[:, :, clamped_selection]

                traj_num = 0
                optimizer.zero_grad()
                current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                pred_vertice, pred_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                v_loss = loss_func(pred_v, target_v)
                (traj_loss + v_loss).backward(retain_graph=True)
                optimizer.step()

                save_steps += 1
                update_steps += 1

                losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                epochs.append(update_steps)
                log(c("[train]", BGRN) + " step=%d epoch=%d loss=%s%s" % (update_steps, epoch, c("%.6f" % losses[-1], BOLD), progress_suffix(update_steps)))
                if save_steps % save_period == 0:
                    save_pickle(losses, "loss_record/train_loss_%s.pkl" %DLO_type)
                    save_pickle(epochs, "loss_record/train_epoch_%s.pkl" %DLO_type)
                if max_update_steps is not None and update_steps >= max_update_steps:
                    torch.save(DEFORM_sim.state_dict(), os.path.join("save_model/", "%s_%s.pth" % (DLO_type, str(update_steps))))
                    log(c("[train] reached max_update_steps=%d, stopping (total wall time %.0fs)"
                        % (max_update_steps, time.time() - train_start), BOLD, BGRN))
                    return

            if train_time_horizon > 1:
                previous_vertices, vertices, target_vertices, m_u0 = data
                if train_time_horizon == 2:
                    inputs = target_vertices[:, :, clamped_selection]
                    optimizer.zero_grad()
                    loss = 0
                    for traj_num in range(2):
                        if traj_num == 0:
                            current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vertice, current_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss

                        if traj_num == 1:
                            previous_edge = computeEdges(previous_vertices[:, traj_num])
                            current_edges = computeEdges(pred_vertice)
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0],m_u0[:, traj_num])
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vertice, current_v, theta_full = DEFORM_sim(pred_vertice.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num],
                                clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_steps += 1
                    update_steps += 1
                    losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                    epochs.append(update_steps)
                    log(c("[train]", BGRN) + " step=%d epoch=%d loss=%s%s" % (update_steps, epoch, c("%.6f" % losses[-1], BOLD), progress_suffix(update_steps)))
                    if save_steps % save_period == 0:
                        save_pickle(losses, "loss_record/train_loss_%s.pkl" % DLO_type)
                        save_pickle(epochs, "loss_record/train_epoch_%s.pkl" % DLO_type)
                    if max_update_steps is not None and update_steps >= max_update_steps:
                        torch.save(DEFORM_sim.state_dict(), os.path.join("save_model/", "%s_%s.pth" % (DLO_type, str(update_steps))))
                        log(c("[train] reached max_update_steps=%d, stopping (total wall time %.0fs)"
                        % (max_update_steps, time.time() - train_start), BOLD, BGRN))
                        return

                else:
                    inputs = target_vertices[:, :, clamped_selection]
                    optimizer.zero_grad()
                    loss = 0
                    for traj_num in range(train_time_horizon):
                        if traj_num == 0:
                            current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                            traj_loss_record += traj_loss

                        if traj_num == 1:
                            previous_edge = computeEdges(previous_vertices[:, traj_num])
                            current_edges = computeEdges(pred_vert)
                            vert = pred_vert.clone()
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0[:, traj_num])
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(pred_vert.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                            traj_loss_record += traj_loss

                        if traj_num >= 2:
                            previous_vert = vert.clone()
                            vert = pred_vert.clone()
                            current_v = current_v.clone()
                            m_u0 = m_u0.clone()
                            previous_edge = computeEdges(previous_vert)
                            current_edges = computeEdges(vert)
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(vert.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            traj_loss_record += traj_loss
                            loss += traj_loss + v_loss

                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_steps += 1
                    update_steps += 1
                    losses.append(traj_loss_record.cpu().detach().numpy() / train_time_horizon)
                    epochs.append(update_steps)
                    log(c("[train]", BGRN) + " step=%d epoch=%d loss=%s%s" % (update_steps, epoch, c("%.6f" % losses[-1], BOLD), progress_suffix(update_steps)))
                    if save_steps % save_period == 0:
                        save_pickle(losses, "loss_record/train_loss_%s.pkl" % DLO_type)
                        save_pickle(epochs, "loss_record/train_epoch_%s.pkl" % DLO_type)
                    if max_update_steps is not None and update_steps >= max_update_steps:
                        torch.save(DEFORM_sim.state_dict(), os.path.join("save_model/", "%s_%s.pth" % (DLO_type, str(update_steps))))
                        log(c("[train] reached max_update_steps=%d, stopping (total wall time %.0fs)"
                        % (max_update_steps, time.time() - train_start), BOLD, BGRN))
                        return

if __name__ == "__main__":
    '''
    DLO_type: DLO type name, related to training dataset folder, saved model name and loss record. For loss record, 
        try to explore using tensor board
    DLO_type: DLO1/DLO2/DLO3/DLO4/DLO5
    eval/train set number = number of pickle file
    eval/train time horizon: in this case, FPS = 100 hz. change self.dt in DEFROM_sim but test it stability first 
    batch: training batch. eval batch default = eval set number
    device: cuda:0/CPU switchable
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DLO_type", type=str, default="DLO1")
    parser.add_argument("--train_set_number", type=int, default=56)
    parser.add_argument("--eval_set_number", type=int, default=14)
    parser.add_argument("--train_time_horizon", type=int, default=100)
    parser.add_argument("--eval_time_horizon", type=int, default=500)
    # upstream hardcoded device="cpu" below regardless of CUDA availability, despite
    # this docstring saying "cuda:0/CPU switchable" -- confirmed via nvidia-smi showing
    # zero GPU utilization during a real training run (~15-17s/step on CPU). Default to
    # CUDA when available; override with --device cpu if ever needed.
    parser.add_argument("--device", type=str,
                         default="cuda:0" if torch.cuda.is_available() else "cpu")
    # additive: upstream's train() loops train_epoch=100 * len(train_data_loader) steps
    # with no way to stop early or bound wall-clock time; None preserves original
    # (unbounded) behavior.
    parser.add_argument("--max_update_steps", type=int, default=None,
                         help="stop after this many training steps (checkpoints/eval "
                              "already happen every 20 steps regardless) [unbounded]")
    parser.add_argument("--on_existing_checkpoint", type=str, default="ask",
                         choices=["ask", "resume", "fresh"],
                         help="what to do when save_model/<DLO_type>_*.pth already exists: "
                              "ask interactively, resume from it, or start fresh (overwriting "
                              "matching step numbers) [ask]")
    args = parser.parse_args()
    # NOTE: upstream previously ignored args.train_set_number/eval_set_number/
    # train_time_horizon/eval_time_horizon here, hardcoding 56/14/100/500 regardless of
    # what was passed on the CLI -- now actually wired through.
    train(DLO_type=args.DLO_type, train_set_number=args.train_set_number,
          eval_set_number=args.eval_set_number, train_time_horizon=args.train_time_horizon,
          eval_time_horizon=args.eval_time_horizon, batch=32, DEFORM_func=DEFORM_func,
          DEFORM_sim=DEFORM_sim, device=args.device, max_update_steps=args.max_update_steps,
          on_existing_checkpoint=args.on_existing_checkpoint)

