import torch
import numpy as np

from utils import Average_Meter
from .nnet import Neural_Network

class Agent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.nnet = Neural_Network(self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.nnet.parameters(), lr=self.args.lr)
        self.loss_meter = Average_Meter()

        self.epsilon = self.args.epsilon
        self.gamma = self.args.gamma
        self.epsilon_min = 0.05
        self.decay = 1e-4

        self.state_memory = np.empty((self.args.replay_memory, 3, self.args.size, self.args.size))
        self.action_memory = np.empty(self.args.replay_memory)
        self.reward_memory = np.empty(self.args.replay_memory)
        self.next_state_memory = np.empty((self.args.replay_memory, 3, self.args.size, self.args.size))
        self.terminal_memory = np.empty(self.args.replay_memory, dtype=np.bool)
        self.cntr = 0

    def act(self, s):
        if np.random.random() < self.epsilon:
            return np.random.choice(4)

        s = torch.Tensor([s]).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            qs = self.nnet(s)

        return torch.argmax(qs).item()

    def remember(self, s, a, r, s_, t):
        idx = self.cntr % self.args.replay_memory
        self.state_memory[idx] = s
        self.action_memory[idx] = a
        self.reward_memory[idx] = r
        self.next_state_memory[idx] = s_
        self.terminal_memory[idx] = t
        self.cntr += 1

    def learn(self):
        if self.cntr < self.args.batch_size:
            return 

        self.nnet.train()
        self.optimizer.zero_grad()
        loss_meter = Average_Meter()

        max_idx = min(self.cntr, self.args.replay_memory)
        batch = np.random.choice(max_idx, self.args.batch_size, replace=False)
        batch_idx = np.arange(self.args.batch_size)

        state_batch = torch.FloatTensor(self.state_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch].astype(np.int32)
        reward_batch = torch.FloatTensor(self.reward_memory[batch]).to(self.device)
        next_state_batch = torch.FloatTensor(self.next_state_memory[batch]).to(self.device)
        terminal_batch = torch.BoolTensor(self.terminal_memory[batch]).to(self.device)

        Qs = self.nnet(state_batch)
        Qs_copy = Qs.detach().clone()
        Qs_ = self.nnet(next_state_batch)

        Qs_[terminal_batch] = 0.0

        for i in range(len(state_batch)):
            Qs_copy[i,action_batch[i]] = reward_batch[i] + self.gamma * torch.max(Qs_[i]).item()

        loss = self.loss(Qs_copy, Qs).to(self.device)

        loss_meter.update(loss.item(), Qs.shape[0])
        print(loss_meter)

        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon - self.decay, self.epsilon_min)

    def loss(self, target, out):
        return torch.sum((target - out) ** 2) / target.shape[0]