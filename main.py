from AgentTiger import *
ModelName = "TrainedModel.dat"

Agent = Tiger("TigerEnv-v2")
Agent.educate(inputModelName=ModelName)
Agent.play()
