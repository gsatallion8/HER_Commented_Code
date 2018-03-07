import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        # Size of the buffer is only measured in terms of number of episodes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is a dict with {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    #Checking if the Buffer is full
    def full(self):
        with self.lock:
            return self.current_size == self.size

    #Sampling a minibatch
    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        # state (o), reward (r), goal (g), alternate goal (ag)
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # next state (o_2), next alternate goal (ag_2) but not sure why there's ag_2
        # offset the experience by one time instant to get the next state (o_2) sequence
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # The following is what we must change. We can add another attribute to the ReplayBuffer object
        # A priority value i.e., TD error for each transition
        # Or alternatively we can locally just define an array to store priority values
        # We might end up recalculating though. 
        transitions = self.sample_transitions(buffers, batch_size)

        # Sanity check I guess!
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    # Store episodes. batch_size is the number of episodes being stored (more than one when in parallel)
    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    # Size in terms of number of episodes stored in buffer
    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    # Size in terms of number of time steps of interaction with env
    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    # Number of transitions stored
    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    # Function to get the indices (episode_index) at which the new batch will be stored
    # To remind, buffer is characterized by ['key_of_interest', episode_index, time_inside_episode, vector_of_interest]
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # Add until you fill the buffer, and then start replacing episodes randomly
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size: # If new bactches make the buffer overflow
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
