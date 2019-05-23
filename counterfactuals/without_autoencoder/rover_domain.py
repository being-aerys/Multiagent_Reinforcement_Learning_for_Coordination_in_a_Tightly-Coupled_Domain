import random
from random import randint
import numpy as np
#import math, cPickle
import math, pickle

class Task_Rovers:

    def __init__(self, parameters):
        self.params = parameters
        self.dim_x = parameters.dim_x
        self.dim_y = parameters.dim_y
        self.observation_space = np.zeros((2*360 // self.params.angle_resolution + 4, 1))  # +4 is for incorporating the 4 walls info
        self.action_space = np.zeros((self.params.action_dim,1))

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_pois)]  # FORMAT: [item] = [x, y] coordinate

        self.poi_value = self.set_poi_values()
        self.poi_status = [False for _ in range(self.params.num_pois)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_rovers)]  # Track each rover's position
        self.ledger_closest = [[0.0, 0.0] for _ in range(self.params.num_rovers)]  # Track each rover's ledger call

        #Macro Action trackers
        self.util_macro = [[False, False, False] for _ in range(self.params.num_rovers)] #Macro utilities to track [Is_currently_active?, Is_activated_now?, Is_reached_destination?]

        #Rover path trace (viz)
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_rovers)]



    def set_poi_values(self):  # POI values set to fixed value
        poi_vals = [1.0 for _ in range(self.params.num_pois)]

        for poi_id in range(self.params.num_pois):
            poi_vals[poi_id] =poi_vals[poi_id] * 5

        return poi_vals


    def reset_poi_pos(self):

        if self.params.unit_test == 1: #Unit_test
            self.poi_pos[0] = [0,1]
            return

        if self.params.unit_test == 2: #Unit_test
            if random.random()<0.5: self.poi_pos[0] = [4,0]
            else: self.poi_pos[0] = [4,9]
            return

        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)


        # for reseting the environment with random or non-random positions of POIs
        if self.params.poi_rand: #Random
            for i in range(self.params.num_pois):
                if i % 3 == 0:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif i % 3 == 1:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif i % 3 == 2:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad - 1)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]

        else: #Not_random
            for i in range(self.params.num_pois):
                if i % 3 == 0:
                    x = start + i/4 #randint(start, center - rad - 1)
                    y = start + i/3
                elif i % 3 == 1:
                    x = center + i/4 #randint(center + rad + 1, end)
                    y = start + i/4#randint(start, end)
                elif i % 3 == 2:
                    x = center+i/4#randint(center - rad, center + rad)
                    y = start + i/4#randint(start, center - rad - 1)
                else:
                    x = center+i/4#randint(center - rad, center + rad)
                    y = center+i/4#randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]

    def reset_rover_pos(self):
        start = 1.0; end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        if self.params.unit_test == 1: #Unit test
            self.rover_pos[0] = [end,0];
            return

        for rover_id in range(self.params.num_rovers):
                quadrant = rover_id % 4
                if quadrant == 0:
                    x = center - 1 - (rover_id / 4) % (center - rad)
                    y = center - (rover_id / (4 * center - rad)) % (center - rad)
                if quadrant == 1:
                    x = center + (rover_id / (4 * center - rad)) % (center - rad)-1
                    y = center - 1 + (rover_id / 4) % (center - rad)
                if quadrant == 2:
                    x = center + 1 + (rover_id / 4) % (center - rad)
                    y = center + (rover_id / (4 * center - rad)) % (center - rad)
                if quadrant == 3:
                    x = center - (rover_id / (4 * center - rad)) % (center - rad)
                    y = center + 1 - (rover_id / 4) % (center - rad)
                self.rover_pos[rover_id] = [x, y]

    def reset(self):
        self.reset_poi_pos()
        self.reset_rover_pos()
        self.poi_status = self.poi_status = [False for _ in range(self.params.num_pois)]
        self.util_macro = [[False, False, False] for _ in range(self.params.num_rovers)]  # Macro utilities to track [Is_currently_active?, Is_activated_now?, Is_reached_destination?]
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_rovers)]
        return self.get_joint_state()



    # TODO: compress the state space using VAEs
    def get_joint_state(self):
        joint_state = []
        for rover_id in range(self.params.num_rovers):
            if self.util_macro[rover_id][0]: #If currently active
                if self.util_macro[rover_id][1] == False: #Not first time activate (Not Is-activated-now)
                    return np.zeros((720/self.params.angle_resolution + 5, 1)) -10000 #If macro return none
                else:
                    self.util_macro[rover_id][1] = False  # Turn off is_activated_now?

            self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1] # rover's own positions (x, y)

            rover_state = [0.0 for _ in range(360 // self.params.angle_resolution)]
            poi_state = [0.0 for _ in range(360 // self.params.angle_resolution)]
            temp_poi_dist_list = [[] for _ in range(360 // self.params.angle_resolution)]
            temp_rover_dist_list = [[] for _ in range(360 // self.params.angle_resolution)]

            # Log all distance into brackets for POIs
            for loc, status, value in zip(self.poi_pos, self.poi_status, self.poi_value):
                if status == True: continue #If the POI has been accessed, ignore

                #x1 = loc[0] - self_x; y1 = loc[1] - self_y
                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist > self.params.observation_radius: continue #Observability radius

                if dist == 0: dist = 0.001 # to avoid division by 0

                bracket = int(angle / self.params.angle_resolution)
                temp_poi_dist_list[bracket].append(value/(dist*dist))

            # Log all distance into brackets for other drones
            for id, loc, in enumerate(self.rover_pos):
                if id == rover_id: continue #Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist > self.params.observation_radius: continue #Observability radius

                if dist == 0: dist = 0.001 # to avoid division by 0

                bracket = int(angle / self.params.angle_resolution)
                temp_rover_dist_list[bracket].append(1/(dist*dist))


            ####Encode the information onto the state
            for bracket in range(int(360 / self.params.angle_resolution)):
                # POIs
                num_pois = len(temp_poi_dist_list[bracket])
                if num_pois > 0:
                    if self.params.sensor_model == 1: poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_pois #Density Sensor
                    else: poi_state[bracket] = max(temp_poi_dist_list[bracket])  #closest Sensor
                else: poi_state[bracket] = -1 # \todo:initially it was -1

                #Rovers
                num_rovers = len(temp_rover_dist_list[bracket])
                if num_rovers > 0:
                    if self.params.sensor_model == 1: rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers #Density Sensor
                    else: rover_state[bracket] = max(temp_rover_dist_list[bracket]) #Minimum Sensor
                else: rover_state[bracket] = -1 # \todo:initially it was -1


            # rover_state: state of all rovers within its each angle (resolution)
            # poi_state: state of all rovers within its each angle (resolution)

            state = rover_state + poi_state #Append rover and poi to form the full state



            #Append wall info
            # if x and y coordinates of rover are at or less than a distance of observation radius from the 4 walls,
            # add it to state, to know that the rovers are within the boundary of the walls
            # TODO: explain this
            state = state + [-1.0, -1.0, -1.0, -1.0] ##### extra information for wall info
            if self_x <= self.params.observation_radius: state[-4] = self_x  # if x pos is within observation radius, add it to state
            if self.params.dim_x - self_x <= self.params.observation_radius: state[-3] = self.params.dim_x - self_x
            if self_y <= self.params.observation_radius :state[-2] = self_y # if y pos is within observation radius, add it to state
            if self.params.dim_y - self_y <= self.params.observation_radius: state[-1] = self.params.dim_y - self_y

            state[-4] = state[-4] / self.dim_x #max(state[-4], state[-3], state[-2], state[-1])
            state[-3] = state[-3] / self.dim_x #max(state[-4], state[-3], state[-2], state[-1])
            state[-2] = state[-2] / self.dim_x #max(state[-4], state[-3], state[-2], state[-1])
            state[-1] = state[-1] / self.dim_x #max(state[-4], state[-3], state[-2], state[-1])

            #state = np.array(state)
            joint_state.append(state)

        return joint_state

    def get_angle_dist(self, x1, y1, x2,y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)

        v1 = x2 - x1
        v2 = y2 - y1
        angle = np.rad2deg(np.arctan2(v1, v2))
        if angle < 0: angle += 360

        dist = v1 * v1 + v2 * v2
        dist = math.sqrt(dist)

        return angle, dist

    '''
        dot = x2 * x1 + y2 * y1  # dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist
    '''

    def step(self, joint_action):

        for rover_id in range(self.params.num_rovers):
            action = joint_action[rover_id]
            new_pos = [self.rover_pos[rover_id][0]+action[0], self.rover_pos[rover_id][1]+action[1]]

            #Check if action is legal
            if not(new_pos[0] >= self.dim_x or new_pos[0] < 0 or new_pos[1] >= self.dim_y or new_pos[1] < 0):  #If legal
                self.rover_pos[rover_id] = [new_pos[0], new_pos[1]] #Execute action

        #Append rover path
        for rover_id in range(self.params.num_rovers):
            self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]))

        return self.get_joint_state(), self.get_reward()



    # Here each rover is getting its own reward for observing a POI
    # but it does not depend on quadrant (just the location of the rover and the POI)
    def get_reward(self):
        #Update POI's visibility
        poi_visitors = [[] for _ in range(self.params.num_pois)]

        for i, loc in enumerate(self.poi_pos): #For all POIs
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            for rover_id in range(self.params.num_rovers): #For each rover
                x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
                dist = math.sqrt(x1 * x1 + y1 * y1)
                if dist <= self.params.activation_dist: poi_visitors[i].append(rover_id) #Add rover id to that POI's visitor list

        #Compute reward
        rewards = [0.0 for _ in range(self.params.num_rovers)]

        # while calculating rewards, check for each POI and the number of rovers within obs radius of that POI.
        # if that rover is observing a POI along with other rovers (minimum number of POIs required to observe some POI)
        # then that rover gets the reward.
        for poi_id, rovers in enumerate(poi_visitors):
            if len(rovers) >= self.params.coupling:  # if more than the specified number of rovers are observing any of POI, that rover gets a rewards
                self.poi_status[poi_id] = True       # that POI is observed
                lucky_rovers = random.sample(rovers, self.params.coupling)  # randomly sample required (2) rovers from the visitors

                for rover_id in lucky_rovers:
                    rewards[rover_id] += self.poi_value[poi_id]/self.params.num_pois # TODO: explain this- how much that rover is contributing in observing all avilable POIs


        return rewards


    def visualize(self):

        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        # Draw in hive
        drone_symbol_bank = ["0", "1", '2', '3', '4', '5']
        for rover_pos, symbol in zip(self.rover_pos, drone_symbol_bank):
            x = int(rover_pos[0]); y = int(rover_pos[1])
            #print x,y
            grid[x][y] = symbol


        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]); y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print(row)


    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # Draw in rover path
        for rover_id in range(self.params.num_rovers):
            for time in range(self.params.num_timesteps):
                x = int(self.rover_path[rover_id][time][0]);
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]);
            y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print (row)


        print ('------------------------------------------------------------------------')



#Functions
def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        #cPickle.dump(obj, output, -1)
        pickle.dump(obj, output, -1)

def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))

def to_numpy(var):
    return var.cpu().data.numpy()