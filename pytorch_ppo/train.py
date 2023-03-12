
#This code for training the proximal policy optimization with actor
#2 critic
#Note: This training is only for continuous action space

#importing the modules
import numpy as np
import torch
import os
import time
import gym
from datetime import datetime
import csv
  

#trail-1 Bipedal Walker-v3

def train():
    #training parameter #may change to achieve maximum reward
    env_name="BipedalWalker-v3"

    #time paramters
    max_episode_len=1500
    max_training_timestep=int(4e6)

    #playing with data
    print_time=max_episode_len*6
    log_time=max_episode_len*3

    #action distribution for better action prediction
    action_dist=0.6

    #policy upgrading
    upgrade_policy=max_episode_len*4
    
    #policy data
    epoch=60
    gamma=0.99
    clip=0.2
    lr_actor=0.0003
    lr_critic=0.002
    seed=0
    if seed:
      np.random.seed(seed)
      env.seed(seed)
      torch.manual_seed(seed)
    print("-----------------------------------------------------------------")
    print("Initialising the continuous action state policy")
    print("training the policy with : " +env_name)

    #initiating the gym
    env=gym.make(env_name)
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    logger_name=env_name+".csv"
    print("logging the data in a csv file")
    print("-----------------------------------------------------------------")
    print("maximum training steps:",max_training_timestep)
    print("maximum episode time:",max_episode_len)
    print("logging frequency:",log_time)
    print("showing reward frequency:",print_time)
    print("action dim:",action_dim)
    print("observations dim:",state_dim)
    print("action distribution or multivariate normal:",action_dist)
    print("policy update frequency:",upgrade_policy)
    print("epoch:",epoch)
    print("policy ratio clip:",clip)
    print("discount factor:",gamma)
    print("actor learning rate:",lr_actor)
    print("critic learning rate:",lr_critic)
    print("-----------------------------------------------------------------")
    
    #initializing the policy
    smart=ppo(action_dim,state_dim,epoch,lr_actor,lr_critic,clip,gamma,action_dist)
    start=datetime.now()
    print("start training time at:",start)
    log=open(logger_name,"w+")
    log.write('episode,timestep,reward\n')
    present_reward=0
    present_episode=0
    run_log_reward=0
    run_log_episodes=0

    time_step=0
    i_episode=0
    while time_step<max_training_timestep:
        state=env.reset()
        current_episode_reward=0
        for t in range(1,max_episode_len+1):
            action=smart.select_action(state)
            state,reward,done,_=env.step(action)
            smart.local.rewards.append(reward)
            smart.local.terminals.append(done)
            time_step+=1
            current_episode_reward+=reward

            if time_step % upgrade_policy==0:
                smart.update()

            if time_step % log_time==0:
                avg_log_reward=run_log_reward/run_log_episodes
                avg_log_reward=round(avg_log_reward,4)

                log.write('{},{},{}\n'.format(i_episode,time_step,avg_log_reward))
                log.flush()
                run_log_reward=0
                run_log_episodes=0

            if time_step % print_time ==0:
                present_reward_avg=present_reward/present_episode
                present_reward_avg=round(present_reward_avg,2)

                print("Episode:{} \t Timestep:{} \t Average Reward:{} \t".format(i_episode,time_step,present_reward_avg))
                present_reward=0
                present_episode=0

            if done:
               break

        present_reward+=current_episode_reward
        present_episode+=1
        run_log_reward+=current_episode_reward
        run_log_episodes+=1

        i_episode+=1

    log.close()
    env.close()
    end=datetime.now()
    print("started training at",start)
    print("ended training at:",end)
    print("total training time:",end-start)


if __name__=='__main__':
    train()
