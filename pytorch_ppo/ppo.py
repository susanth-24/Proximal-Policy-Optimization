#proximal policy optimization implementation
#import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from torch.optim import Adam
#setting up the device
device=torch.device('cpu')

#creating the local store
class store:
    def __init__(self):
        self.actions=[]
        self.states=[]
        self.log_probs=[]
        self.rewards=[]
        self.state_values=[]
        self.terminals=[]

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.terminals[:]

class actor_critic(nn.Module):
    def __init__(self,action_dim,state_dim,action_dist):
        super(actor_critic,self).__init__()

        #creating the neural net for actor
        self.actor=nn.Sequential(
            nn.Linear(state_dim,70),
            nn.ReLU(),
            nn.Linear(70,70),
            nn.ReLU(),
            nn.Linear(70,action_dim),
            nn.Tanh()
        )

        #creating the neural net for critic
        self.critic=nn.Sequential(
            nn.Linear(state_dim,70),
            nn.ReLU(),
            nn.Linear(70,70),
            nn.Tanh(),
            nn.Linear(70,1)
        )

        self.cov=torch.full((action_dim,),action_dist**2)

    def action_pred(self,obs):
        action_mean=self.actor(obs)
        cov_mat=torch.diag(self.cov).unsqueeze(dim=0)
        prob_distribution=MultivariateNormal(action_mean,cov_mat)
        action=prob_distribution.sample()
        log_probs=prob_distribution.log_prob(action)
        state_vals=self.critic(obs)
        
        return action.detach(),log_probs.detach(),state_vals.detach()

    def evaluate(self,obs,action):
        values=self.critic(obs)
        action_mean=self.actor(obs)
        cov_mat=torch.diag_embed(self.cov.expand_as(action_mean))
        prob_distribution=MultivariateNormal(action_mean,cov_mat)
        log_probs=prob_distribution.log_prob(action)

        return log_probs,values

    
class ppo:
    def __init__(self,action_dim,state_dim,epoch,lr_actor,lr_critic,clip,gamma,action_dist):
        self.states_dim=state_dim
        self.action_dim=action_dim
        self.epoch=epoch
        self.clip=clip
        self.gamma=gamma
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.local=store()
        self.policy=actor_critic(self.action_dim,self.states_dim,action_dist)
        self.optimize_actor=Adam(self.policy.actor.parameters(),lr=self.lr_actor)
        self.optimize_critic=Adam(self.policy.critic.parameters(),lr=self.lr_critic)

    def select_action(self,state):
        state=torch.FloatTensor(state).to(device)
        action,log_probs,state_values=self.policy.action_pred(state)
        self.local.actions.append(action)
        self.local.log_probs.append(log_probs)
        self.local.states.append(state)
        self.local.state_values.append(state_values)
        #print(action.item())
        return action.detach().cpu().numpy().flatten()

    def update(self):
        rewards=[]
        discounted_reward=0
        for reward, terminal in zip(reversed(self.local.rewards),reversed(self.local.terminals)):
            if terminal:
                discounted_reward=0
            discounted_reward=reward+(self.gamma*discounted_reward)
            rewards.insert(0,discounted_reward)

        rewards=torch.tensor(rewards,dtype=torch.float32).to(device)
        rewards=(rewards-rewards.mean())/(rewards.std()+1e-7)

        stack_states=torch.squeeze(torch.stack(self.local.states,dim=0)).detach().to(device)
        stack_actions=torch.squeeze(torch.stack(self.local.actions,dim=0)).detach().to(device)
        stack_log_probs=torch.squeeze(torch.stack(self.local.log_probs,dim=0)).detach().to(device)
        stack_state_values=torch.squeeze(torch.stack(self.local.state_values,dim=0)).detach().to(device)

        advantages=rewards.detach()-stack_state_values.detach()

        for t in range(self.epoch):
            log_probs,values=self.policy.evaluate(stack_states,stack_actions)
            values=torch.squeeze(values)
            ratio=torch.exp(log_probs-stack_log_probs)

            #surrogate loss
            s1=ratio*advantages
            s2=torch.clamp(ratio,1-self.clip,1+self.clip)*advantages

            actor_loss=(-torch.min(s1,s2)).mean()
            #print("a",np.shape(values))
            #print("aa",np.shape(rewards))
            ''' print("aa",rewards.size())
            print("a",values.size())
            if torch.isnan(values).any() or torch.isnan(rewards).any():
                raise ValueError("Input or target tensor contains NaN values.")

            if torch.isinf(values).any() or torch.isinf(rewards).any():
                raise ValueError("Input or target tensor contains infinity values.") '''
            critic_loss=nn.MSELoss()(values,rewards)
            #print(critic_loss)

            #calculating the gradients and doing backward propagation
            self.optimize_actor.zero_grad()
            actor_loss.backward()
            self.optimize_actor.step()

            self.optimize_critic.zero_grad()
            critic_loss.backward()
            self.optimize_critic.step()

        self.local.clear()


