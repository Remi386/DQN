import torch.nn as nn
import torch
from Envir import *
import torch.optim as optim
import numpy as np
import collections

ReplayBufferSize = 700
BATCH_SIZE = 40  # размер выборки
GAMMA = 0.8  # для уравнения Беллмана
# Для оптимизатора Адама
LEARNING_RATE = 1e-4
MOMENTUM = 0.8
SYNC_TARGET_ITER = 500
# скрытые слои нейронных сетей
hidden_size1 = 320
hidden_size2 = 240
hidden_size3 = 30
TemporaryModel = "TemporaryWeigths-"

EpisodeStep = collections.namedtuple('EpisodeStep', field_names=['state', 'step', 'reward', 'done', 'next_state'])
Episode = collections.namedtuple('Episode', field_names=['totalReward', 'steps'])

StepsDict = {0: "Step up", 1: "Step up right", 2: "Step to the right", 3: "Step down right",
         4: "Step down", 5: "Step down left", 6: "Step to the left", 7: "Step up Left"}

class Tiger(nn.Module):
    _currentEpisode = 0
    _EpisodesCount = MaxEpisodes

    def __init__(self, envName):
        super(Tiger, self).__init__()

        self._replayBuffer = collections.deque(maxlen=ReplayBufferSize)
        self._replayBufferChanged = collections.deque(maxlen=ReplayBufferSize)
        self.env = gym.make(envName)
        self._action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self._state = self.env.reset()
        input_size = self._observation_space.shape[0] * self._observation_space.shape[1]

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, ACTIONS_COUNT)
        )

        self.tgt_net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, ACTIONS_COUNT)
        )

    def forward(self, x):
        return self.net(x)

    def seed(self, seed):
        self.env.seed(seed)

    def educate(self, inputModelName=None, outputModelName=None, educateMore=False):
        if inputModelName is not None:
            self.tgt_net.load_state_dict(torch.load(inputModelName, map_location=lambda storage, loc: storage))
            if educateMore:
                self.tgt_net.train()
            else:
                self.tgt_net.eval()
                return

        self._currentEpisode = 0
        optimizer = optim.SGD(self.net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        for _ in range(self._EpisodesCount):
            done = False
            total_reward = 0
            self._state = self.env.reset()

            # Для отладки
            if self._currentEpisode % 1000 == 0:
                print("{} iteration".format(self._currentEpisode))

            # Временные модели
            # if self._currentEpisode % 100000 == 0:
            #    tstr = TemporaryModel + str(self._currentEpisode) + ".dat"
            #    torch.save(self.tgt_net.state_dict(), tstr)

            if self._currentEpisode % SYNC_TARGET_ITER == 0:
                self.tgt_net.load_state_dict(self.net.state_dict())

            steps = collections.deque()

            changed_flag = False
            while not done:
                current_state = self._state
                next_state, reward, done, [action, changed] = self.step()
                total_reward += reward

                if not changed_flag:
                    if changed:
                        changed_flag = True

                # Запись эпизода в очередь
                step = EpisodeStep(current_state, action, reward, done, next_state)
                steps.append(step)



                # Если размер буфера недостаточно большой, пропускаем итерацию
                if len(self._replayBuffer) < ReplayBufferSize:
                    continue

                optimizer.zero_grad()

                loss = self.loss_function(self._replayBuffer)
                loss.backward()

                if len(self._replayBufferChanged) > 0:
                    loss1 = self.loss_function(self._replayBufferChanged, useAll=True)
                    loss1.backward()

                optimizer.step()

            eps = Episode(total_reward, steps)

            if changed_flag:
                self._replayBufferChanged.append(eps)
            else:
                self._replayBuffer.append(eps)

            self._currentEpisode += 1

        if outputModelName is not None:
            torch.save(self.tgt_net.state_dict(), outputModelName)

    def step(self):
        action = self._egreedy_policy()
        next_state, reward, done, changed = self.env.step(action)
        self._state = next_state
        return self._state, reward, done, [action, changed]

    def loss_function(self, buffer, useAll=False):
        states, actions, rewards, dones, next_states = self._bufferSample(buffer, useAll)
        states_v = torch.FloatTensor(states)
        next_states_v = torch.FloatTensor(next_states)
        rewards_v = torch.FloatTensor(rewards)
        actions_v = torch.tensor(actions, dtype=torch.int64)
        done_mask = torch.tensor(dones, dtype=torch.bool)

        # Извлекаем q values из состояний и примененных действий к ним
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        # Лучшее действия из следующего состояния
        next_state_values = self.tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0  # Если эпизод финальный - нет следующего состояния, как и награды
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def render(self):
        self.env.render()

    def _egreedy_policy(self):
        action = 0
        if random.uniform(0, 1) < float(self._EpisodesCount - self._currentEpisode) / self._EpisodesCount:
            action = self._action_space.sample()
        else:
            state_v = torch.FloatTensor(self._state)
            q_vals_v = self.tgt_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=0)
            action = round(act_v.item())
        return action

    def _bufferSample(self, buffer, useAll):
        # случайная выборка эпизодов
        if useAll:
            total_reward, steps = zip(*[buffer[i] for i in range(len(buffer))])
            states, actions, rewards, dones, next_states = zip(*[steps[i] for i in range(len(steps))])
            return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
                   np.array(dones, dtype=np.bool), np.array(next_states)
        else:
            indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
            total_reward, episode = zip(*[buffer[idx] for idx in indices])
            steps = list([episode[i][j] for i in range(len(episode)) for j in range(len(episode[i]))])
            states, actions, rewards, dones, next_states = zip(*[steps[i] for i in range(len(steps))])
            return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states)

    def play(self):
        done = False
        self._currentEpisode = self._EpisodesCount
        self._state = self.env.reset()
        while not done:
            next_state, reward, done, [action, _] = self.step()
            print("{}, Reward: {}".format(StepsDict[action], reward))
            self.render()
            self._state = next_state
