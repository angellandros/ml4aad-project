import os

import numpy as np

from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


class ESOptimizer(object):

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 rng: np.random.RandomState,
                 traj_logger: TrajLogger):
        
        self.incumbent = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.traj_logger = traj_logger or TrajLogger('./smac-output/run_1/', stats)

    def run(self, time_left=100, count=5):
        for i in range(count):
            best = self.race(inc_id=i)
        return best

    def race(self, time_left=100, count=1, inc_id=0):
        random_confs = self.scenario.cs.sample_configuration(count)
        if count == 1:
            random_confs = [random_confs]
        best, inc_perf = self.intensifier.intensify(
            challengers=random_confs,
            incumbent=self.incumbent,
            run_history=self.runhistory,
            aggregate_func=self.aggregate_func,
            time_bound=time_left
        )
        self.incumbent = best
        print('=================================', inc_perf)
        self.traj_logger.add_entry(inc_perf, inc_id, self.incumbent)
        return best
