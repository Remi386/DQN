import random
import gym
from gym import spaces
import numpy as np

FIELD_HEIGHT = 10
FIELD_WIDTH = 10
ACTIONS_COUNT = 8
START_Y_POSITION = 0
START_X_POSITION = 0

TIGER = 1
RABBIT = 2
ENEMY = 3

FIELD = np.zeros((FIELD_HEIGHT, FIELD_WIDTH))
FIELD[START_Y_POSITION, START_X_POSITION] = TIGER

#rabbits
FIELD[9, 3] = RABBIT
FIELD[8, 7] = RABBIT
FIELD[4, 9] = RABBIT

#enemies
FIELD[4, 2] = ENEMY
FIELD[1, 5] = ENEMY
FIELD[3, 7] = ENEMY
FIELD[7, 3] = ENEMY

StepPenalty = -1.0
RabbitReward = 20.0
ReturningReward = 80.0
EnemyPenalty = -100
OutOfBoundsPenalty = -10
TrackingReward = 0.5
TrackingDistance = 3 #дистанция отслеживания добычи
JumpDistance = 5 #дистанция, на которую перемещается кролик
MaxEpisodes = 1000000

# possible actions
StepUp = 0
StepUpRight = 1
StepRight = 2
StepDownRight = 3
StepDown = 4
StepDownLeft = 5
StepLeft = 6
StepUpLeft = 7

# TigerStates
TigerStatesCount = 2
THungry = 0
TFULL = 1
#TSearch = 2

class TigerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    _PosX = 0
    _PosY = 0
    _TField = np.zeros((FIELD_HEIGHT, FIELD_WIDTH))
    _state = 0
    _TigerMastery = 1

    def __init__(self):
        super(TigerEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTIONS_COUNT)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(FIELD_WIDTH, FIELD_HEIGHT, TigerStatesCount), dtype=np.uint16)

    def step(self, action):

        varPosY, varPosX = self._takeStep(action)

        reward, done, OutOfBounds = self._calculateReward(varPosY, varPosX)
        if(not OutOfBounds):
            self._TField[self._PosY, self._PosX] = 0
            self._TField[varPosY, varPosX] = TIGER
            self._PosX, self._PosY = varPosX, varPosY
        else:
            self._TField.fill(0)

        return np.copy(self._TField).flatten(), reward, done, {}

    def seed(self, seed=1234):
        random.seed(seed)

    def reset(self):
        self._PosY = START_Y_POSITION
        self._PosX = START_X_POSITION
        self._TField = np.copy(FIELD)
        self._state = THungry
        return np.copy(self._TField).flatten()

    def render(self, mode='human'):
        for y in range(FIELD_HEIGHT):
            print(self._TField[y])
        print("\n\n")

    def close(self):
        ...

    def _calculateReward(self, NposY, NposX):
        #Выход за границы
        if (not self._isPossibleToStep(NposY, NposX)):
            return OutOfBoundsPenalty, True, True
        #Попытка поймать кролика
        if (self._TField[NposY, NposX] == RABBIT and self._state == THungry):
            if(self._tryToCatch(NposY, NposX)):
                self._state = TFULL
            return RabbitReward, False, False
        #Наткнулись на врага
        if (self._TField[NposY, NposX] == ENEMY):
            return EnemyPenalty, True, False
        #Вернулись домой сытыми
        if (NposX == START_X_POSITION and NposY == START_Y_POSITION and self._state == TFULL):
            return ReturningReward, True, False
        #Добыча поблизости
        if (self._trackingPrey()):
            return TrackingReward, False, False
        #Обычный шаг
        return StepPenalty, False, False

    def _tryToCatch(self, RposY, RposX):
        if(random.uniform(0, 1) < self._TigerMastery):
            return True
        else:
            self._TigerMastery += 0.1
            self._changeRabbitPosition(RposY, RposX)
            return False

    def _changeRabbitPosition(self, RposY, RposX):
        success = False
        while not success:
            x = JumpDistance * random.randint(-1, 1)
            y = JumpDistance * random.randint(-1, 1)
            if RposY + y >= 0 and RposY + y < FIELD_HEIGHT and RposX + x >= 0 and RposX + x < FIELD_WIDTH:
                if self._TField[RposY + y][RposX + x] == 0:
                    self._TField[RposY + y][RposX + x] = RABBIT
                    success = True

    def _trackingPrey(self):
        #Выслеживание добычи
        rabbitPositions = list()
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                if (self._TField[y][x] == RABBIT):
                    rabbitPositions.append([y, x])
        for rabbit in rabbitPositions:
            if abs(self._PosY - rabbit[0]) <= TrackingDistance and \
                    abs(self._PosX - rabbit[1]) <= TrackingDistance:
                return True
        return False

    def _isPossibleToStep(self, NposY, NposX):
        return (NposX >= 0 and NposX < FIELD_WIDTH and NposY >= 0 and NposY < FIELD_HEIGHT)

    def _takeStep(self, action):
        if (action == StepUp):
            return self._PosY - 1, self._PosX
        if (action == StepUpRight):
            return self._PosY - 1, self._PosX + 1
        if (action == StepRight):
            return self._PosY, self._PosX + 1
        if (action == StepDownRight):
            return self._PosY + 1, self._PosX + 1
        if (action == StepDown):
            return self._PosY + 1, self._PosX
        if (action == StepDownLeft):
            return self._PosY + 1, self._PosX - 1
        if (action == StepLeft):
            return self._PosY, self._PosX - 1
        if (action == StepUpLeft):
            return self._PosY - 1, self._PosX - 1


gym.envs.register(
     id='TigerEnv-v2',
     entry_point='__main__:TigerEnv',
     max_episode_steps=1000000
)