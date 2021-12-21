from AgentTiger import *
ModelName = "TrainedModel.dat"

Agent = Tiger("TigerEnv-v1")
Agent.educate(inputModelName="TestModel.dat")
Agent.seed(123414)
Agent.play()



'''
for _ in range(EpisodesCount):
    state = env.reset()
    done = False
    while not done:
        action = ChooseAction(state, QTable, currentEpisode)
        next_state, reward, done, info = env.step(action)
        QTable[state][action] += Alpha * (reward + Gamma * np.max(QTable[next_state]) - QTable[state][action])
        state = next_state
    currentEpisode += 1

state = env.reset()
done = False
print("State: " + str(state))
while not done:
    action = ChooseAction(state, QTable, currentEpisode)
    next_state, reward, done, info = env.step(action)
    print("{}, state: {}, Reward: {}".format(steps[action], next_state, reward))
    env.render()
    state = next_state
'''