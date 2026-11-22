"""Generic PINN trainer: Adam (with lr decay + periodic collocation resampling), then L-BFGS.

Used by every stage: the stage script supplies
  modules      dict name -> nn.Module   (everything trainable; checkpointed by name)
  loss_fn()    -> (total_loss_tensor, {term_name: tensor})
  resample()   optional, called every `resample_every` Adam steps (new collocation points)
  evaluate()   optional, -> {metric: float}, called every `eval_every` steps and at the end
  snapshot(it) optional, store intermediate predictions (for the training GIF)

Kaggle-friendliness:
  * hard wall-clock budget (`time_budget_min`) across both phases
  * checkpoint every `ckpt_every` Adam steps and after every L-BFGS chunk (atomic writes);
    re-running the same command resumes; a finished run is detected and not retrained
  * L-BFGS NaN guard: if a chunk blows up, restore the last good weights and stop L-BFGS
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch

from common.checkpoint import load_checkpoint, save_checkpoint


def _sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()


class Trainer:
    def __init__(self, modules: dict, loss_fn, tcfg, out_dir: Path, device,
                 resample=None, evaluate=None, snapshot=None, ckpt_name="ckpt.pt",
                 signature: str = ""):
        self.modules, self.loss_fn, self.t = modules, loss_fn, tcfg
        self.signature = signature                   # config fingerprint; resume only if equal
        self.resample, self.evaluate, self.snapshot = resample, evaluate, snapshot
        self.device = device
        self.ckpt_path = Path(out_dir) / ckpt_name
        self.params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
        self.history: list[dict] = []
        self.phase, self.step = "adam", 0           # step = global iteration count
        self.adam_done, self.lbfgs_done = 0, 0
        self.elapsed = 0.0                           # training seconds, summed across restarts
        self.adam = torch.optim.Adam(self.params, lr=float(tcfg.adam_lr))
        n_adam = max(int(tcfg.adam_iters), 1)
        gamma = float(tcfg.get("lr_final_ratio", 0.05)) ** (1.0 / n_adam)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.adam, gamma=gamma)

    # ---------------------------------------------------------------- checkpointing ----------
    def _save(self):
        save_checkpoint(self.ckpt_path, signature=self.signature, phase=self.phase, step=self.step, adam_done=self.adam_done,
                        lbfgs_done=self.lbfgs_done, elapsed=self.elapsed, history=self.history,
                        adam=self.adam, sched=self.sched,
                        modules={k: m.state_dict() for k, m in self.modules.items()})

    def _try_resume(self) -> bool:
        if not bool(self.t.get("resume", True)):
            return False
        ck = load_checkpoint(self.ckpt_path, map_location=self.device)
        if ck is None:
            return False
        if ck.get("signature", "") != self.signature:
            print(f"[train] {self.ckpt_path.name} was made with a different config — starting fresh")
            return False
        for k, m in self.modules.items():
            m.load_state_dict(ck["modules"][k])
        self.adam.load_state_dict(ck["adam"]); self.sched.load_state_dict(ck["sched"])
        self.phase, self.step = ck["phase"], ck["step"]
        self.adam_done, self.lbfgs_done = ck["adam_done"], ck["lbfgs_done"]
        self.elapsed, self.history = ck["elapsed"], ck["history"]
        print(f"[train] resumed from {self.ckpt_path.name}: phase={self.phase} step={self.step} "
              f"elapsed={self.elapsed:.0f}s")
        return True

    # ---------------------------------------------------------------- logging ----------------
    def _log(self, terms: dict, lr: float, do_eval: bool):
        row = {"step": self.step, "phase": self.phase, "time_s": round(self.elapsed, 2), "lr": lr}
        row.update({k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()})
        if do_eval and self.evaluate is not None:
            row.update(self.evaluate())
        self.history.append(row)
        msg = " ".join(f"{k}={v:.3e}" for k, v in row.items()
                       if isinstance(v, float) and k not in ("time_s", "lr"))
        print(f"[{self.phase:5s} {self.step:6d} {self.elapsed:6.0f}s] {msg}", flush=True)

    def _budget_left(self, t_start) -> bool:
        return self.elapsed + (time.perf_counter() - t_start) < 60.0 * float(self.t.time_budget_min)

    # ---------------------------------------------------------------- main loop --------------
    def run(self):
        self._try_resume()
        if self.phase == "done":
            print("[train] checkpoint says training is complete — skipping to evaluation")
            return self.history
        t = self.t
        log_every, eval_every = int(t.log_every), int(t.get("eval_every", t.log_every))
        snap_every = int(t.get("snapshot_every", 0))
        resample_every = int(t.get("resample_every", 0))

        # ---- Adam ----
        t0 = time.perf_counter()
        if self.phase == "adam":
            if self.resample is not None:
                self.resample()
            while self.adam_done < int(t.adam_iters):
                if resample_every and self.resample is not None and self.adam_done % resample_every == 0:
                    self.resample()
                self.adam.zero_grad(set_to_none=True)
                loss, terms = self.loss_fn()
                loss.backward()
                self.adam.step(); self.sched.step()
                self.adam_done += 1; self.step += 1
                if self.step % log_every == 0 or self.adam_done == int(t.adam_iters):
                    _sync(self.device)
                    self.elapsed += time.perf_counter() - t0; t0 = time.perf_counter()
                    terms = {"loss": loss, **terms}
                    self._log(terms, self.sched.get_last_lr()[0], self.step % eval_every == 0)
                    if not math.isfinite(self.history[-1]["loss"]):
                        raise FloatingPointError("Adam loss became non-finite — check scaling / lr")
                if snap_every and self.snapshot is not None and self.step % snap_every == 0:
                    self.snapshot(self.step)
                if self.step % int(t.ckpt_every) == 0:
                    self.elapsed += time.perf_counter() - t0; t0 = time.perf_counter()
                    self._save()
                if not self._budget_left(t0):
                    print("[train] time budget reached during Adam")
                    break
            self.elapsed += time.perf_counter() - t0; t0 = time.perf_counter()
            self.phase = "lbfgs"
            self._save()

        # ---- L-BFGS (fixed collocation set: line search needs a deterministic objective) ----
        if self.phase == "lbfgs" and int(t.lbfgs_iters) > 0:
            if self.resample is not None:
                self.resample()
            chunk = min(int(t.get("lbfgs_chunk", 50)), int(t.lbfgs_iters))
            opt = torch.optim.LBFGS(self.params, lr=1.0, max_iter=chunk, history_size=50,
                                    tolerance_grad=1e-9, tolerance_change=1e-12,
                                    line_search_fn="strong_wolfe")
            last = {"loss": None, "terms": None}

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, terms = self.loss_fn()
                loss.backward()
                last["loss"], last["terms"] = loss, terms
                return loss

            prev_loss = float("inf")
            while self.lbfgs_done < int(t.lbfgs_iters) and self._budget_left(t0):
                good = {k: copy.deepcopy(m.state_dict()) for k, m in self.modules.items()}
                opt.step(closure)
                n_it = opt.state[opt._params[0]].get("n_iter", chunk)
                done_now = n_it - getattr(self, "_lbfgs_n_seen", 0)
                self._lbfgs_n_seen = n_it
                self.lbfgs_done += max(done_now, 1); self.step += max(done_now, 1)
                _sync(self.device)
                self.elapsed += time.perf_counter() - t0; t0 = time.perf_counter()
                cur = float(last["loss"].detach())
                if not math.isfinite(cur):
                    for k, m in self.modules.items():
                        m.load_state_dict(good[k])
                    print("[train] L-BFGS produced a non-finite loss — restored last good weights")
                    break
                self._log({"loss": last["loss"], **last["terms"]}, 1.0, True)
                if self.snapshot is not None:
                    self.snapshot(self.step)
                self._save()
                if done_now <= 0 or abs(prev_loss - cur) <= 1e-12 * max(abs(cur), 1e-30):
                    print("[train] L-BFGS converged (no further progress)")
                    break
                prev_loss = cur
        self.phase = "done"
        self.elapsed += time.perf_counter() - t0
        self._save()
        return self.history
