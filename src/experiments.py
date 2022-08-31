from src.events import Events
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class ExperimentalDesign:

    def __init__(self, events: Events, seed=0):

        self._seed = seed
        self._events = events

        # Set the seed value
        self._set_seed(seed=seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_events(self, events: Events):

        self._events = events

    def construct_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
            bins_num=bins_num, subsampling=subsampling, init_time=init_time, last_time=last_time, with_time=with_time
        )

        time_gen = np.random.default_rng()
        chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
        neg_samples = list(map(
            lambda idx: (
                possible_neg_samples[idx][0],
                possible_neg_samples[idx][1],
                (possible_neg_samples[idx][3] - possible_neg_samples[idx][2])*time_gen.random()+possible_neg_samples[idx][2]
            ), chosen_idx
        ))

        all_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        all_samples = pos_samples + neg_samples

        return all_labels, all_samples

    def _pos_and_pos_neg_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        all_pos_samples, all_possible_neg_samples = [], []
        if bins_num > 1:

            bounds = np.linspace(init_time, last_time, bins_num + 1)
            for b in range(bins_num):
                '''
                labels, samples = self.construct_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b], last_time=bounds[b+1]
                )
                all_labels += labels
                all_samples += samples
                '''
                pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b],
                    last_time=bounds[b + 1]
                )

                all_pos_samples += pos_samples
                all_possible_neg_samples += possible_neg_samples

        else:

            if init_time is None and last_time is None:
                subevents = self.__events.get_data()
            else:
                subevents = self.__events.get_subevents(init_time=init_time, last_time=last_time).get_data()

            # Sample positive instances
            if with_time:
                pos_samples = [(i, j, t) for i, j in self.__pairs() for t in subevents[i][j]]
            else:
                pos_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) > 0]

            if subsampling > 0:
                pos_samples = np.random.choice(pos_samples, size=subsampling, replace=False).tolist()

            possible_neg_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) == 0]
            if with_time:
                # time_list = (last_time - init_time) * np.random.random_sample(len(possible_neg_samples)) + init_time
                # possible_neg_samples = [(sample[0], sample[1], t) for sample, t in zip(possible_neg_samples, time_list)]
                possible_neg_samples = [(sample[0], sample[1], init_time, last_time) for sample in possible_neg_samples]

            all_pos_samples = pos_samples
            all_possible_neg_samples = possible_neg_samples

        return all_pos_samples, all_possible_neg_samples


class LinkRemovalExperiment:

    def __init__(self, events: Events, seed=0):

        self._seed = seed
        self._events = events

        # Set the seed value
        self._set_seed(seed=seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_pos_neg_samples(self, num=None):

        pairs_list = self._events.get_pairs()
        events_list = self._events.get_events()

        n = len(pairs_list)

        indices = np.random.choice(np.arange(n), replace=False)

        print(indices)









# class ExperimentalDesign:
#
#     def __init__(self, events: Events, seed=0):
#
#         self._seed = seed
#         self._events = events
#
#         # Set the seed value
#         self._set_seed(seed=seed)
#
#     def _set_seed(self, seed):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#
#     def set_events(self, events: Events):
#
#         self._events = events
#
#     def construct_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):
#
#         pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
#             bins_num=bins_num, subsampling=subsampling, init_time=init_time, last_time=last_time, with_time=with_time
#         )
#
#         time_gen = np.random.default_rng()
#         chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
#         neg_samples = list(map(
#             lambda idx: (
#                 possible_neg_samples[idx][0],
#                 possible_neg_samples[idx][1],
#                 (possible_neg_samples[idx][3] - possible_neg_samples[idx][2])*time_gen.random()+possible_neg_samples[idx][2]
#             ), chosen_idx
#         ))
#
#         all_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
#         all_samples = pos_samples + neg_samples
#
#         return all_labels, all_samples
#
#     def _pos_and_pos_neg_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):
#
#         all_pos_samples, all_possible_neg_samples = [], []
#         if bins_num > 1:
#
#             bounds = np.linspace(init_time, last_time, bins_num + 1)
#             for b in range(bins_num):
#                 '''
#                 labels, samples = self.construct_samples(
#                     bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b], last_time=bounds[b+1]
#                 )
#                 all_labels += labels
#                 all_samples += samples
#                 '''
#                 pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
#                     bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b],
#                     last_time=bounds[b + 1]
#                 )
#
#                 all_pos_samples += pos_samples
#                 all_possible_neg_samples += possible_neg_samples
#
#         else:
#
#             if init_time is None and last_time is None:
#                 subevents = self.__events.get_data()
#             else:
#                 subevents = self.__events.get_subevents(init_time=init_time, last_time=last_time).get_data()
#
#             # Sample positive instances
#             if with_time:
#                 pos_samples = [(i, j, t) for i, j in self.__pairs() for t in subevents[i][j]]
#             else:
#                 pos_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) > 0]
#
#             if subsampling > 0:
#                 pos_samples = np.random.choice(pos_samples, size=subsampling, replace=False).tolist()
#
#             possible_neg_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) == 0]
#             if with_time:
#                 # time_list = (last_time - init_time) * np.random.random_sample(len(possible_neg_samples)) + init_time
#                 # possible_neg_samples = [(sample[0], sample[1], t) for sample, t in zip(possible_neg_samples, time_list)]
#                 possible_neg_samples = [(sample[0], sample[1], init_time, last_time) for sample in possible_neg_samples]
#
#             all_pos_samples = pos_samples
#             all_possible_neg_samples = possible_neg_samples
#
#         return all_pos_samples, all_possible_neg_samples


