"""Resumable sample cache shared by the sampling scripts."""
import os
import sys
import time
import argparse
import numpy as np

from paths import cache_path


def parse(default_bs):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 samples, tiny batch, separate cache")
    ap.add_argument("--bs", type=int, default=default_bs)
    ap.add_argument("--wall-guard", type=float, default=float(os.environ.get("WALL_GUARD_S", 46.0 * 3600)))
    return ap


class Cache:
    def __init__(self, slug, M0, smoke, t0):
        self.path = cache_path(slug, smoke)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.M0 = M0
        self.t0 = t0
        self.X32 = np.zeros((M0, 32, 32, 3), dtype=np.uint8)
        self.done = 0
        if os.path.exists(self.path):
            prev = np.load(self.path)["X32"]
            prev = prev.reshape(len(prev), 32, 32, 3)
            self.done = min(len(prev), M0)
            self.X32[:self.done] = prev[:self.done]
            print(f"resumed cache {self.path}: {self.done} samples", flush=True)
        self.t_sample = None

    def start(self):
        self.t_sample = time.time()
        self.n_start = self.done

    def add(self, hwc_uint8):
        nb = len(hwc_uint8)
        self.X32[self.done:self.done + nb] = hwc_uint8
        self.done += nb

    def save(self):
        np.savez_compressed(self.path, X32=self.X32[:self.done])

    def progress(self):
        dt = time.time() - self.t_sample + 1e-9
        rate = (self.done - self.n_start) / dt
        print(f"  {self.done}/{self.M0} sampled ({rate:.1f} img/s, "
              f"{rate*3600:.0f} img/h)  ({time.time()-self.t0:.0f}s)", flush=True)

    def finish(self, smoke=False, full_M=None):
        self.save()
        dt = time.time() - self.t_sample + 1e-9
        n_new = self.done - self.n_start
        rate = n_new / dt
        state = "finished" if self.done >= self.M0 else "paused"
        print(f"sampling {state} at {self.done}/{self.M0}  ({dt:.0f}s sampling, "
              f"{rate:.2f} img/s)  cache={self.path}", flush=True)
        if smoke and rate > 0:
            Mf = full_M or self.M0
            print(f"SMOKE OK: projected full run at this rate = "
                  f"{Mf/rate/3600:.2f} h  (M={Mf}, tiny-batch rate, pessimistic)", flush=True)
        return self.done >= self.M0
