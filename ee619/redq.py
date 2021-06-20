from .agent import Agent as SAC
import numpy as np
import os
import torch

from torch.optim import Adam

from .model import GaussianPolicy, DeterministicPolicy


class REDQ:

    def __init__(self, num_q, M, num_inputs, action_space, args):

        self.num_q = num_q
        self.M = M

        self.alpha = args.alpha
        self.gamma = args.gamma
        self.batch_size = args.batch_size

        self.sac_agents = np.array([SAC() for i in range(num_q)])

        self.policy_type = args.policy
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.device = torch.device("cuda" if args.cuda else "cpu")

    def sample_M(self, M):

        random_indices = np.random.randint(0, self.num_q, M)

        return random_indices

    def get_q(self, agent, next_state):

        agent_action = agent.select_action(next_state)
        agent_action = torch.FloatTensor(agent_action).view(1, -1).to(self.device)

        return agent_action

    def update_parameters(self, memory, batch_size, updates):

        # Sample a batch from memory

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        random_indices = self.sample_M(self.M)
        M_agents = self.sac_agents[random_indices]

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)

        next_q_values = []
        for agent in M_agents:
            next_q = self.next_q_value(agent, state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                                       next_state_action, next_state_log_pi)
            next_q_values.append(next_q)

        target_q = torch.cat(next_q_values, 1)
        min_target_q = torch.min(target_q, dim=1, keepdim=True)[0]

        critic_1_total_loss = []
        critic_2_total_loss = []

        for agent in M_agents:
            critic_1_loss, critic_2_loss = agent.update_parameters(state_batch, action_batch, reward_batch,
                                                                   next_state_batch, mask_batch, batch_size,
                                                                   min_target_q, updates)
            critic_1_total_loss.append(critic_1_loss)
            critic_2_total_loss.append(critic_2_loss)

        critic_1_loss = sum(critic_1_total_loss) / len(critic_1_total_loss)
        critic_2_loss = sum(critic_2_total_loss) / len(critic_2_total_loss)

        pi, log_pi, _ = self.policy.sample(state_batch)

        q_a_tilda_list = []

        for agent in self.sac_agents:
            q_1, q_2 = agent.critic(state_batch, pi)
            min_qf_pi = torch.min(q_1, q_2)
            q_a_tilda_list.append(min_qf_pi)

        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)

        policy_loss = ((self.alpha * log_pi) - ave_q).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return critic_1_loss, critic_2_loss, policy_loss.item()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def next_q_value(self, agent, state_batch, action_batch, reward_batch, next_state_batch, mask_batch,
                     next_state_action, next_state_log_pi):

        with torch.no_grad():
            qf1_next_target, qf2_next_target = agent.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        return next_q_value

    def load_model(self, actor_path):
        print('Loading models from {}'.format(actor_path))
        self.policy.load_state_dict(torch.load(actor_path))

    def save_model(self, env_name, suffix):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        actor_path = "models/redq_actor_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(actor_path))
        torch.save(self.policy.state_dict(), actor_path)

