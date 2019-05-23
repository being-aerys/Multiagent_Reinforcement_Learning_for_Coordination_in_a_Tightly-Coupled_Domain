from core import TD3 as pg
from core import models
from core.mod_utils import list_mean, pprint, str2bool
import numpy as np, os, time, random, torch
from core import mod_utils as utils
from core.runner import rollout_worker
import core.ounoise as OU_handle
from torch.multiprocessing import Process, Pipe, Manager
import argparse
from core.buffer import Buffer


parser = argparse.ArgumentParser()
parser.add_argument('-num_workers', type=int,  help='#Rollout workers',  default=6)
parser.add_argument('-mem_cuda', type=str2bool,  help='#Store buffer in GPU?',  default=False)
parser.add_argument('-render', type=str2bool,  help='#Render?',  default=False)
parser.add_argument('-savetag', help='Saved tag',  default='def')
parser.add_argument('-gamma', type=float,  help='#Gamma',  default=0.97)
parser.add_argument('-seed', type=float,  help='#Seed',  default=7)
parser.add_argument('-dpp', type=str2bool,  help='#Use_DPP?',  default=True)


SEED = vars(parser.parse_args())['seed']
NUM_WORKERS = vars(parser.parse_args())['num_workers']
MEM_CUDA = vars(parser.parse_args())['mem_cuda']
SAVE_TAG = vars(parser.parse_args())['savetag']
GAMMA = vars(parser.parse_args())['gamma']
RENDER = vars(parser.parse_args())['render']
DPP = vars(parser.parse_args())['dpp']
CUDA = True



class Parameters:
    def __init__(self):

        #TD3 params
        self.gamma = GAMMA
        self.tau = 0.001
        self.batch_size = 128
        self.buffer_size = 1000000
        self.updates_per_step = 1
        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.2

        #NN specifics
        self.num_episodes = 100000
        self.is_dpp = True


        #Rover domain
        self.dim_x = self.dim_y = 15; self.obs_radius = 100; self.act_dist = 1.5; self.angle_res = 20
        self.num_poi = 5; self.num_rover = 12; self.ep_len = 25
        self.poi_rand = False; self.coupling = 5; self.rover_speed = 1
        self.render = RENDER
        self.is_homogeneous = False  #False --> Heterogenenous Actors
        self.sensor_model = 'closest'  #Closest VS Density

        #Dependents
        self.state_dim = int(720 / self.angle_res + 4)
        self.action_dim = 2
        self.test_frequency = 10

        #Save Filenames
        self.save_foldername = 'R_idpp/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.aux_save = self.save_foldername + 'auxiliary/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)
        if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

        self.critic_fname = 'critic_'+ 'dpp_' if DPP else '' + '_' +SAVE_TAG
        self.actor_fname = 'actor_'+ 'dpp_' if DPP else ''+  '_' + SAVE_TAG
        self.log_fname = 'reward_'+ 'dpp_' if DPP else ''+  '_' + SAVE_TAG
        self.best_fname = 'best_'+'dpp_' if DPP else ''+ '_' + SAVE_TAG

        #Unit tests (Simply changes the rover/poi init locations)
        self.unit_test = 0 #0: None
                           #1: Single Agent
                           #2: Multiagent 2-coupled



class IDPP:
    """Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
       Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

            Parameters:
                args (int): Parameter class with all the parameters

            """

    def __init__(self, args):
        self.args = args
        self.agents = [pg.TD3(args) for _ in range(self.args.num_rover)]


        #Load to GPU
        for ag in self.agents: ag.to_cuda()


        ###### Buffer is agent's own data self generated via its rollouts #########
        self.buffers = [Buffer() for _ in range(self.args.num_rover)]
        self.noise_gen = OU_handle.get_list_generators(NUM_WORKERS, args.action_dim)


        ######### Multiprocessing TOOLS #########
        self.manager = Manager()
        self.data_bucket = [self.manager.list() for _ in range(args.num_rover)] #Experience list stores experiences from all processes


        ######### TRAIN ROLLOUTS WITH ACTION NOISE ############
        self.models_bucket = self.manager.list()
        model_template = models.Actor(args)
        for _ in range(self.args.num_rover): self.models_bucket.append(models.Actor(args))
        self.task_pipes = [Pipe() for _ in range(NUM_WORKERS)]
        self.result_pipes = [Pipe() for _ in range(NUM_WORKERS)]
        self.train_workers = [Process(target=rollout_worker, args=(self.args, i, self.task_pipes[i][1], self.result_pipes[i][0], self.noise_gen[i], self.data_bucket, self.models_bucket, model_template)) for i in range(NUM_WORKERS)]
        for worker in self.train_workers: worker.start()

        ######## TEST ROLLOUT POLICY ############
        self.test_task_pipe = Pipe()
        self.test_result_pipe = Pipe()
        self.test_worker = Process(target=rollout_worker, args=(self.args, 0, self.test_task_pipe[1], self.test_result_pipe[0], None, self.data_bucket, self.models_bucket, model_template))
        self.test_worker.start()


        #### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
        self.best_policy = models.Actor(args) #Best policy found by PF yet
        self.best_score = -999; self.test_score = None
        self.test_eval_flag = True; self.rollout_scores = [None for _ in range(NUM_WORKERS)]
        self.best_rollout_score = -999
        self.train_eval_flag = [True for _ in range(NUM_WORKERS)]
        self.update_budget = 0


    def train(self, gen, tracker):
        """Main training loop to do rollouts and run policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """

        #Sync models
        for ag, bucket_model in zip(self.agents, self.models_bucket):
            ag.actor.cpu()
            ag.hard_update(bucket_model, ag.actor)
            ag.actor.cuda()


        ########### START TEST ROLLOUT ##########
        if self.test_eval_flag: #ALL DATA TEST
            self.test_eval_flag = False
            render = random.random() < 0.1
            self.test_task_pipe[0].send(render)


        ########## START TRAIN (ACTION NOISE) ROLLOUTS ##########
        for i in range(NUM_WORKERS):
            if self.train_eval_flag[i]:
                self.train_eval_flag[i] = False
                self.task_pipes[i][0].send(False)

        ############ POLICY GRADIENT #########
        if self.buffers[0].counter > 1000:
            while self.update_budget > 0:

                for buffer, agent in zip(self.buffers, self.agents):
                    s, ns, a, r, done = buffer.sample(self.args.batch_size)

                    s = torch.Tensor(s); ns = torch.Tensor(ns); a = torch.Tensor(a); r = torch.Tensor(r); done = torch.Tensor(done)
                    if not MEM_CUDA and CUDA: s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda(); done = done.cuda()

                    agent.update_parameters(s, ns, a, r, done, DPP, num_epoch=1)
                    self.update_budget -= 1


        #Save models periodically
        if gen % 20 == 0:
            for rover_id in range(self.args.num_rover):
                torch.save(self.agents[rover_id].critic.state_dict(), self.args.model_save + self.args.critic_fname + '_'+ str(rover_id))
                torch.save(self.agents[rover_id].actor.state_dict(), self.args.model_save + self.args.actor_fname + '_'+ str(rover_id))
            print("Models Saved")


        ##### PROCESS TEST ROLLOUT ##########
        if self.test_result_pipe[1].poll():
            entry = self.test_result_pipe[1].recv()
            self.test_eval_flag = True
            self.test_score = entry[0]
            gen_tracker.update([self.test_score], gen)
            if self.test_score > self.best_score: self.best_score = self.test_score


        ####### PROCESS TRAIN ROLLOUTS ########
        while True:
            for i in range(NUM_WORKERS):
                if self.result_pipes[i][1].poll():
                    entry = self.result_pipes[i][1].recv()
                    self.train_eval_flag[i] = True
                    self.rollout_scores[i] = entry[0]
                    if entry[0] > self.best_rollout_score: self.best_rollout_score = entry[0]

            #Hard Join
            if sum(self.train_eval_flag) == len(self.train_eval_flag): break




        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for rover_id in range(args.num_rover):
            for _ in range(len(self.data_bucket[rover_id])):
                exp = self.data_bucket[rover_id].pop()
                self.buffers[rover_id].push(exp[0], exp[1], exp[2], exp[3], exp[4])
                self.update_budget += 1



if __name__ == "__main__":
    args = Parameters()  # Create the Parameters class


    gen_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)    #Seeds

    # INITIALIZE THE MAIN AGENT CLASS
    ai = IDPP(args)
    print(' State_dim:', args.state_dim)
    time_start = time.time()

    ###### TRAINING LOOP ########
    for gen in range(1, 1000000000): #RUN VIRTUALLY FOREVER
        gen_time = time.time()

        #ONE EPOCH OF TRAINING
        ai.train(gen, gen_tracker)

        #PRINT PROGRESS
        print('Ep:', gen, 'Score: cur/best:', pprint(ai.test_score), pprint(ai.best_score),
              'Time:',pprint(time.time()-gen_time),
              'Best_rollout_score', pprint(ai.best_rollout_score),
              'DPP', DPP)




    

