from parameters import Parameters as p
import numpy as np
import random


cpdef one_of_each_type(c_number, current_type, rover_dist):
    cdef int i
    cdef int ntypes = int(p.num_types)
    cdef double[:, :] partners = np.zeros((c_number, 2))

    for i in range(c_number):
        partners[i, 0] = rover_dist  # Place counterfactual partner at target rover's location
        partners[i, 1] = float(i)  # Type of rover

        if ntypes > 1:
            while partners[i, 1] == current_type:  # Do not suggest same type if there are multiple
                partners[i, 1] = float(random.randint(0, (ntypes-1)))

    return partners
