clusterConfigurationDict = {}
clusterConfigurationDict["raw landscape directory"] = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/"
clusterConfigurationDict["tmp directory"] = "/scratch/eecs-share/rhoutman/mcgregse/data/tmp/"
clusterConfigurationDict["landscape summary directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/landscapes/"
clusterConfigurationDict["landscape processing jump"] = -50

clusterConfigurationDict["raw CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/estimatedoutput.csv"
clusterConfigurationDict["processed CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/processed.csv"

clusterConfigurationDict["variances output path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/variances.pkl"

clusterConfigurationDict["horizon"] = 99
clusterConfigurationDict["target trajectory count"] = 30
clusterConfigurationDict["policy parameters ERC"] = [0, 20, 40, 60, 80]
clusterConfigurationDict["policy parameters startIndex"] = [0, 36, 72, 108, 144]

clusterConfigurationDict["experimental outputs directory"] = "/nfs/stak/students/m/mcgregse/Projects/rlpy/experiments/results/"

clusterConfigurationDict["environment"] = "production"

testingConfigurationDict = {}
# Not used since we are not testing this step
#testingConfigurationDict["raw landscape directory"] = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/"
#testingConfigurationDict["landscape summary directory"] = "/nfs/stak/students/m/mcgregse/Projects/rlpy/experiments/data/landscapes/"
#testingConfigurationDict["landscape processing jump"] = 1

testingConfigurationDict["raw CSV path"] = "estimatedoutput.csv"
testingConfigurationDict["processed CSV path"] = "experiments/data/processed.csv"

testingConfigurationDict["variances output path"] = "experiments/data/variances.pkl"

testingConfigurationDict["horizon"] = 99
testingConfigurationDict["target trajectory count"] = 30
testingConfigurationDict["policy parameters ERC"] = [0, 20, 40, 60, 80]
testingConfigurationDict["policy parameters startIndex"] = [0, 36, 72, 108, 144]

testingConfigurationDict["experimental outputs directory"] = "experiments/results/"

testingConfigurationDict["environment"] = "testing"
