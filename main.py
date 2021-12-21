from AgentTiger import *
ModelName = "TemporaryWeigths-600000.dat"

Agent = Tiger("TigerEnv-v2")
Agent.educate(inputModelName=ModelName)
Agent.play()
