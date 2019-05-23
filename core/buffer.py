import random, numpy as np


class Buffer():
    """Cyclic Buffer stores experience tuples from the rollouts

        Parameters:
            save_freq (int): Period for saving data to drive
            save_folder (str): Folder to save data to
            capacity (int): Maximum number of experiences to hold in cyclic buffer

        """

    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done = []
        self.counter = 0

    def push(self, s, a, ns, r, done): #f: FITNESS, t: TIMESTEP, done: DONE
        """Add an experience to the buffer

            Parameters:
                s (ndarray): Current State
                ns (ndarray): Next State
                a (ndarray): Action
                r (ndarray): Reward
                done_dist (ndarray): Temporal distance to done (#action steps after which the skselton fell over)
                done (ndarray): Done
                shaped_r (ndarray): Shaped Reward (includes both temporal and behavioral shaping)


            Returns:
                None
        """


        if self.__len__() < self.capacity:
            self.s.append(None); self.ns.append(None); self.a.append(None); self.r.append(None); self.done.append(None)

        #Append new tuple
        ind = self.counter % self.capacity
        self.s[ind] = s; self.ns[ind] = ns; self.a[ind] = a; self.r[ind] = r; self.done[ind] = done
        self.counter += 1


    def __len__(self):
        return len(self.s)


    def sample(self, batch_size):
        """Sample a batch of experiences from memory with uniform probability

               Parameters:
                   batch_size (int): Size of the batch to sample

               Returns:
                   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
           """
        ind = random.sample(range(self.__len__()), batch_size)
        return np.vstack([self.s[i] for i in ind]), np.vstack([self.ns[i] for i in ind]), np.vstack([self.a[i] for i in ind]), np.vstack([self.r[i] for i in ind]), np.vstack([self.done[i] for i in ind])



    # def save(self):
    #     """Method to save experiences to drive
    #
    #            Parameters:
    #                None
    #
    #            Returns:
    #                None
    #        """
    #
    #     tag = str(int(self.counter / self.save_freq))
    #     list_files = os.listdir(self.folder)
    #     while True:
    #         save_fname = self.folder + SAVE_TAG + tag
    #         if save_fname in list_files:
    #             tag += 1
    #             continue
    #         break
    #
    #
    #     end_ind = self.counter % self.capacity
    #     start_ind = end_ind - self.save_freq
    #
    #     try:
    #         np.savez_compressed(save_fname,
    #                         state=np.vstack(self.s[start_ind:end_ind]),
    #                         next_state=np.vstack(self.ns[start_ind:end_ind]),
    #                         action = np.vstack(self.a[start_ind:end_ind]),
    #                         reward = np.vstack(self.r[start_ind:end_ind]),
    #                         done_dist = np.vstack(self.done_dist[start_ind:end_ind]),
    #                         done=np.vstack(self.done[start_ind:end_ind]))
    #         print ('MEMORY BUFFER WITH INDEXES', str(start_ind), str(end_ind),  'SAVED WITH TAG', tag)
    #     except:
    #         print()
    #         print()
    #         print()
    #         print()
    #         print('############ WARNING! FAILED TO SAVE FROM INDEX ', str(start_ind), 'to', str(end_ind), '################')
    #         print()
    #         print()
    #         print()
    #         print()