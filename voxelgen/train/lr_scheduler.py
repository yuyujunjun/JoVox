# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from torch.optim.lr_scheduler import LRScheduler
from accelerate.logging import get_logger


logger = get_logger(__name__)


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, initial_lr: float = 1e-10,
                 min_lr: float = 0.0, last_iter: int = -1, final_anneal_start: int = None,
                 final_anneal_iters: int = 0, final_min_lr: float = 0.0):
        self.final_anneal_start = final_anneal_start
        self.final_anneal_iters = final_anneal_iters
        self.final_min_lr = final_min_lr
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_iter)

    def get_lr(self):
        if self.final_anneal_start is not None and self._step_count > self.final_anneal_start + 1:
            k = self.final_anneal_start + 1
            hi = self._cosine_lr(k)
            theta = min((self._step_count - k) / self.final_anneal_iters * math.pi, math.pi)
            return [self.final_min_lr + (h - self.final_min_lr) * (1 + math.cos(theta)) / 2 for h in hi]
        return self._cosine_lr(self._step_count)

    def _cosine_lr(self, step):
        if step <= self.warmup_iters:
            return [
                self.initial_lr + (base_lr - self.initial_lr) * step / self.warmup_iters
                for base_lr in self.base_lrs]
        else:
            cos_iter = step - self.warmup_iters
            cos_max_iter = self.max_iters - self.warmup_iters
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_theta = min(cos_theta, math.pi)
            cos_lr = [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(cos_theta)) / 2
                      for base_lr in self.base_lrs]
            return cos_lr
