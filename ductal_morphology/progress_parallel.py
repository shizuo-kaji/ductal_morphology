from tqdm.auto import tqdm
import joblib

# parallel execution with progress bar
class ProgressParallel(joblib.Parallel):
    def __init__(self, total=None, *args, **kwargs):
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(total=self._total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()