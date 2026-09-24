"""Shared progress and ETA logging for research experiments."""
import time

class ResearchProgress:
    def __init__(self, experiment_id, total_runs):
        self.experiment_id=experiment_id
        self.total_runs=total_runs
        self.started=time.monotonic()
        self.run_started=None
        self.completed=0

    @staticmethod
    def _fmt(seconds):
        seconds=max(0,int(seconds))
        h,rem=divmod(seconds,3600); m,s=divmod(rem,60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    def start_run(self, model, seed):
        self.run_started=time.monotonic()
        print(f"\n[{self.experiment_id}] Starting run {self.completed+1}/{self.total_runs} | model={model} | seed={seed}",flush=True)

    def epoch(self, epoch, epochs, model, seed):
        elapsed=time.monotonic()-self.started
        print(f"[{self.experiment_id}] model={model} | seed={seed} | epoch {epoch}/{epochs} complete | total elapsed {self._fmt(elapsed)}",flush=True)

    def finish_run(self, model, seed, accuracy=None):
        now=time.monotonic(); self.completed+=1
        elapsed=now-self.started
        avg=elapsed/self.completed
        eta=avg*(self.total_runs-self.completed)
        pct=100*self.completed/self.total_runs
        width=20; filled=round(width*self.completed/self.total_runs)
        bar="█"*filled+"░"*(width-filled)
        run_time=now-self.run_started if self.run_started is not None else 0
        acc="" if accuracy is None else f" | accuracy={accuracy:.4f}"
        print(f"[{self.experiment_id}] Run {self.completed}/{self.total_runs} complete | model={model} | seed={seed}{acc}",flush=True)
        print(f"Progress [{bar}] {self.completed}/{self.total_runs} ({pct:.1f}%) | run {self._fmt(run_time)} | elapsed {self._fmt(elapsed)} | ETA ~{self._fmt(eta)}",flush=True)
