from AgentTiger import *
ModelName = "TrainedModel_Indicat.dat"

Agent = Tiger("TigerEnv-v2")
Agent.educate(outputModelName=ModelName)
Agent.play()
