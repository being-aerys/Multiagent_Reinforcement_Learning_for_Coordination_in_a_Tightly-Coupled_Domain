from core.rover_domain import Task_Rovers
from core import mod_utils as utils
import numpy as np, torch


#Rollout evaluate an agent in a complete game
def rollout_worker(args, worker_id, task_pipe, result_pipe, noise, data_bucket, models_bucket, model_template):
    """Rollout Worker runs a simulation in the environment to generate experiences and fitness values

        Parameters:
            worker_id (int): Specific Id unique to each worker spun
            task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
            result_pipe (pipe): Sender end of the pipe used to report back results
            noise (object): A noise generator object
            exp_list (shared list object): A shared list object managed by a manager that is used to store experience tuples
            pop (shared list object): A shared list object managed by a manager used to store all the models (actors)
            difficulty (int): Difficulty of the task
            use_rs (bool): Use behavioral reward shaping?
            store_transition (bool): Log experiences to exp_list?

        Returns:
            None
    """

    worker_id = worker_id; env = Task_Rovers(args)
    models = [model_template for _ in range(args.num_rover)]
    for m in models: m = m.eval()


    while True:
        RENDER = task_pipe.recv() #Wait until a signal is received  to start rollout

        # Get the current model state from the population
        for m, bucket_model in zip(models, models_bucket):
            utils.hard_update(m, bucket_model)

        fitness = 0.0
        joint_state = env.reset(); rollout_trajectory = [[] for _ in range(args.num_rover)]
        joint_state = utils.to_tensor(np.array(joint_state))
        while True: #unless done

            joint_action = [models[i].forward(joint_state[i,:]).detach().numpy() for i in range(args.num_rover)]
            if noise != None:
                for action in joint_action: action += noise.noise()


            next_state, reward, done, info = env.step(joint_action)  # Simulate one step in environment


            next_state = utils.to_tensor(np.array(next_state))
            fitness += sum(reward)/args.coupling

            #If storing transitions
            for i in range(args.num_rover):
                rollout_trajectory[i].append([np.expand_dims(utils.to_numpy(joint_state)[i,:], 0), np.expand_dims(np.array(joint_action)[i,:], 0),
                                              np.expand_dims(utils.to_numpy(next_state)[i, :], 0), np.expand_dims(np.array([reward[i]]), 0),
                                              np.expand_dims(np.array([done]), 0)])

            joint_state = next_state

            #DONE FLAG IS Received
            if done:
                if RENDER: env.render()
                #Push experiences to main
                for rover_id in range(args.num_rover):
                    for entry in rollout_trajectory[rover_id]:
                        for i in range(len(entry[0])):
                            data_bucket[rover_id].append([entry[0], entry[1], entry[2], entry[3], entry[4]])
                break

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([fitness])




