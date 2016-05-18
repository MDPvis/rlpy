clusterConfiguration = {}
clusterConfiguration["raw landscape directory"] = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/"
clusterConfiguration["tmp directory"] = "/scratch/eecs-share/rhoutman/mcgregse/data/tmp/"
clusterConfiguration["landscape summary directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/landscapes/"
clusterConfiguration["landscape processing jump"] = -50
clusterConfiguration["raw CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/estimatedoutput.csv"
clusterConfiguration["processed CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/processed.csv"
clusterConfiguration["variances output path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/variances.pkl"
clusterConfiguration["horizon"] = 99
clusterConfiguration["target trajectory count"] = 30
clusterConfiguration["policy parameters ERC"] = [0, 20, 40, 60, 80]
clusterConfiguration["policy parameters startIndex"] = [0, 36, 72, 108, 144]
clusterConfiguration["experimental outputs directory"] = "/nfs/stak/students/m/mcgregse/Projects/rlpy/experiments/results/"
clusterConfiguration["feature selection results directory"] = "experiments/results/feature_selection/"
clusterConfiguration["environment"] = "production"

landscapeConfiguration = {}
landscapeConfiguration["raw landscape directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial/landscapes/"
landscapeConfiguration["tmp directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial/tmp/"
landscapeConfiguration["landscape summary directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial/processed_landscapes/"
landscapeConfiguration["landscape processing jump"] = -5
landscapeConfiguration["raw CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial/estimate/estimatedoutput_split_policy.csv"
landscapeConfiguration["processed CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial/estimate/processed.csv"
landscapeConfiguration["variances output path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/variances.pkl"
landscapeConfiguration["horizon"] = 99
landscapeConfiguration["target trajectory count"] = 30
landscapeConfiguration["policy parameters ERC"] = [0, 470] # hack
landscapeConfiguration["policy parameters startIndex"] = [0] # hack
landscapeConfiguration["experimental outputs directory"] = "/nfs/stak/students/m/mcgregse/Projects/rlpy/experiments/results/"
landscapeConfiguration["environment"] = "production"

landscapeConfiguration2 = {}
landscapeConfiguration2["raw landscape directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial2/landscapes/"
landscapeConfiguration2["tmp directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial2/tmp/"
landscapeConfiguration2["landscape summary directory"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial2/processed_landscapes/"
landscapeConfiguration2["landscape processing jump"] = -5
landscapeConfiguration2["raw CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial2/estimate/estimatedoutput_split_policy.csv"
landscapeConfiguration2["processed CSV path"] = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/spatial2/estimate/processed.csv"
landscapeConfiguration2["variances output path"] = = "/nfs/eecs-fserv/share/rhoutman/mcgregse/data/variances.pkl"
landscapeConfiguration2["horizon"] = 99
landscapeConfiguration2["target trajectory count"] = 30
landscapeConfiguration2["policy parameters ERC"] = [0, 470] # hack
landscapeConfiguration2["policy parameters startIndex"] = [0] # hack
landscapeConfiguration2["experimental outputs directory"] = "/nfs/stak/students/m/mcgregse/Projects/rlpy/experiments/results/"
landscapeConfiguration2["environment"] = "production"


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


# !!! This is the one that is imported !!! #
#clusterConfigurationDict = clusterConfiguration
clusterConfigurationDict = landscapeConfiguration
#clusterConfigurationDict = landscapeConfiguration
