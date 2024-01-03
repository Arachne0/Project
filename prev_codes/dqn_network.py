import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DQN(nn.Module):
    def __init__(self, board_width, board_height, num_actions):
        super(DQN, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # state dimension
        state_dim = board_width * board_height

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        # flatten input
        x = self.flatten(x)

        # common layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values



class DQNNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width      # 9
        self.board_height = board_height    # 4
        self.l2_const = 1e-4  # coef of l2 penalty

        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.dqn_net = DQN(board_width, board_height).to(self.device)

        self.optimizer = optim.Adam(self.dqn_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.dqn_net.load_state_dict(net_params)



    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch_np = np.array(state_batch)
            state_batch = torch.FloatTensor(state_batch_np).to(self.device)
            log_act_probs, value = self.dqn_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.item()
        else:
            state_batch_np = np.array(state_batch)
            state_batch = torch.FloatTensor(state_batch_np)
            log_act_probs, value = self.dqn_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.detach().cpu().numpy()



    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Tensor
        state_batch = torch.from_numpy(np.array(state_batch)).float()
        mcts_probs = torch.from_numpy(np.array(mcts_probs)).float()
        winner_batch = torch.from_numpy(np.array(winner_batch)).float()

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.dqn_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.detach().view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.dqn_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
