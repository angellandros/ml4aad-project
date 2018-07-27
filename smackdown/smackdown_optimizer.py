import os
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
        self.default_configuration = scenario.cs.get_default_configuration()
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.rng = rng
        self.traj_logger = traj_logger or TrajLogger('./smac-output/run_1/', stats)
        self.keys = list(self.default_configuration.keys())

    def run(self, time_left=100, count=20):
        for i in range(count):
            best = self.race(inc_id=i)
        return best

    def race(self, time_left=100, count=100, inc_id=0, train_model=True):
        random_confs = self.scenario.cs.sample_configuration(count)
        if count == 1:
            random_confs = [random_confs]

        if train_model and not self.runhistory.empty():
            then = time.time()
            regr = RandomForestRegressor(max_depth=2, random_state=0)
            conf_ids = [k for k in self.runhistory.data.keys()]
            X = [self.__unserialize__(self.runhistory.ids_config[id[0]]) for id in conf_ids]
            y = [self.runhistory.data[id][0] for id in conf_ids]
            regr.fit(X, y)
            print('model fit with %d points in' % len(X), time.time() - then)
            then = time.time()
            best_pred_perf = np.infty
            best_pred_conf = random_confs[0]
            for conf in random_confs:
                pred_perf = regr.predict([self.__unserialize__(conf)])
                print('.', end='')
                if pred_perf < best_pred_perf:
                    best_pred_conf = conf
            print('\nconfigurations predicted in', time.time() - then)
            random_confs = [best_pred_conf]  # not random any more!
        if train_model:
            random_confs = [random_confs[0]]

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

    def __unserialize__(self, conf):
        vector = []
        for k in self.keys:
            if k in conf.keys():
                vector.append(conf[k])
            else:
                vector.append(self.default_configuration[k])
        return vector
