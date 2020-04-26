import numpy as np
import torch
import torch.nn as nn
GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 4



class ActorNetwork(nn.Module):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, state_dim, action_dim, learning_rate):
        super(ActorNetwork, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.fc1 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=6),
            nn.ReLU(inplace=True)
        )


        self.fc3 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(128*6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,self.a_dim),
            nn.Softmax()
        )

    def forward(self,x):
        split_0 = self.fc1(x[:, 0:1, -1])
        split_1 = self.fc2(x[:, 1:2, -1])

        split_2 = torch.flatten(self.conv1(x[:, 2:3, :]),1,2)
        split_3 = torch.flatten(self.conv2(x[:, 3:4, :]),1,2)
        split_4 = torch.flatten(self.conv3(x[:, 4:5, :A_DIM]),1,2)
        split_5 = self.fc3(x[:, 5:6, -1])

        # print([split_0.shape, split_1.shape, split_2.shape, split_3.shape, split_4.shape, split_5.shape])
        # exit()
        tensor_merge = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5],dim=1)
        out = self.out_layer(tensor_merge)

        return out

class ActorNetworkLSTM(nn.Module):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, state_dim, action_dim, learning_rate):
        super(ActorNetworkLSTM, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.fc1 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=6),
            nn.ReLU(inplace=True)
        )


        self.fc3 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTMCell(128*6, 128)

        self.out_layer = nn.Sequential(
            nn.Linear(128,self.a_dim),
            nn.Softmax()
        )

    def forward(self,x,hx,cx):
        split_0 = self.fc1(x[:, 0:1, -1])
        split_1 = self.fc2(x[:, 1:2, -1])

        split_2 = torch.flatten(self.conv1(x[:, 2:3, :]),1,2)
        split_3 = torch.flatten(self.conv2(x[:, 3:4, :]),1,2)
        split_4 = torch.flatten(self.conv3(x[:, 4:5, :A_DIM]),1,2)
        split_5 = self.fc3(x[:, 5:6, -1])

        # print([split_0.shape, split_1.shape, split_2.shape, split_3.shape, split_4.shape, split_5.shape])
        # exit()
        tensor_merge = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5],dim=1)
        hx,cx = self.lstm(tensor_merge,(hx,cx))

        out = self.out_layer(hx)

        return out,hx,cx


class CriticNetwork(nn.Module):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, state_dim, learning_rate):
        super(CriticNetwork, self).__init__()
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        self.fc1 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=6),
            nn.ReLU(inplace=True)
        )


        self.fc3 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(128*6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self,x):
        split_0 = self.fc1(x[:, 0:1, -1])
        split_1 = self.fc2(x[:, 1:2, -1])

        split_2 = torch.flatten(self.conv1(x[:, 2:3, :]),1,2)
        split_3 = torch.flatten(self.conv2(x[:, 3:4, :]),1,2)
        split_4 = torch.flatten(self.conv3(x[:, 4:5, :A_DIM]),1,2)
        split_5 = self.fc3(x[:, 5:6, -1])

        # print([split_0.shape, split_1.shape, split_2.shape, split_3.shape, split_4.shape, split_5.shape])
        # exit()
        tensor_merge = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5],dim=1)
        out = self.out_layer(tensor_merge)

        return out





class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, islstm = 0):
        super(ActorCritic, self).__init__()
        self.islstm = islstm
        if islstm == 0:
            self.actor = ActorNetwork(state_dim, action_dim,learning_rate[0])
        else:
            self.actor = ActorNetworkLSTM(state_dim, action_dim,learning_rate[0])
        self.critic = CriticNetwork(state_dim, learning_rate[1])

    def forward(self,x):
        if self.islstm:
            inputs,hx,cx = x
            action,hx,cx = self.actor(inputs,hx,cx)
            score  = self.critic(inputs)
            return action, score,hx,cx
        else:
            inputs = x
            action = self.actor(inputs)
            score  = self.critic(inputs)
            return action, score

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)


class A3C(nn.Module):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, state_dim, action_dim, learning_rate):
        super(A3C, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.fc1 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=4),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels = 128, kernel_size=6),
            nn.ReLU(inplace=True)
        )


        self.fc3 = nn.Sequential(
            nn.Linear(1,128,bias = True),
            nn.ReLU(inplace=True)
        )

        self.actor_out = nn.Sequential(
            nn.Linear(128*6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,self.a_dim),
            nn.ReLU(inplace=True)
        )
        self.critic_out1 = nn.Sequential(
            nn.Linear(128*6, 128),
            nn.ReLU(inplace=True),
        )
        self.critic_out2 = nn.Sequential(
            nn.Linear(128, 1),
            #nn.ReLU(inplace=True)
        )


    def forward(self,x):
        split_0 = self.fc1(x[:, 0:1, -1])
        split_1 = self.fc2(x[:, 1:2, -1])

        split_2 = torch.flatten(self.conv1(x[:, 2:3, :]),1,2)
        split_3 = torch.flatten(self.conv2(x[:, 3:4, :]),1,2)
        split_4 = torch.flatten(self.conv3(x[:, 4:5, :A_DIM]),1,2)
        split_5 = self.fc3(x[:, 5:6, -1])

        # print([split_0.shape, split_1.shape, split_2.shape, split_3.shape, split_4.shape, split_5.shape])
        # exit()
        tensor_merge = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5],dim=1)

        action = self.actor_out(tensor_merge)
        out1   = self.critic_out1(tensor_merge)
        score  = self.critic_out2(out1)

        return action, score


    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.normal_(module.weight,mean=0, std=0.1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)


    def test_param(self):
        print("**************")
        for m in self.critic_out2.parameters():
            print(m)
            break
        print("**************")



def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch



def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

