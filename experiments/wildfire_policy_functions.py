"""
A set of policy functions used to evaluate visual fidelity.
"""

def wildfirePolicySeverityFactory(ercThreshold, timeUntilEndOfFireSeasonThreshold):
    """
    Gives a policy function defined on the two parameters.
    :param ercThreshold: The minimum intensity conditions at which we will suppress the fire.
    :param timeUntilEndOfFireSeasonThreshold: The maximum amount of time until the end of the fire season at which
      we will begin to allow all wildfires to burn.
    :return: A function mapping states to actions
    """

    def policy_severity(
            state,
            possibleActions,
            transitionTuple=None,
            ercThreshold=ercThreshold,
            timeUntilEndOfFireSeasonThreshold=timeUntilEndOfFireSeasonThreshold):
        assert transitionTuple is not None
        erc = transitionTuple.additionalState["ERC"]
        timeUntilEndOfFireSeason = 181 - transitionTuple.additionalState["startIndex"]
        assert erc >= 0, "ERC was {}".format(erc)
        assert erc <= 100, "ERC was {}".format(erc)
        assert timeUntilEndOfFireSeason <= 180, "timeUntilEndOfFireSeason was {}".format(timeUntilEndOfFireSeason)
        assert timeUntilEndOfFireSeason >= 0, "timeUntilEndOfFireSeason was {}".format(timeUntilEndOfFireSeason)
        if erc >= 95:
            return 0
        elif erc >= ercThreshold:
            return 1
        elif timeUntilEndOfFireSeason < timeUntilEndOfFireSeasonThreshold:
            return 1
        else:
            return 0
    return policy_severity

def wildfirePolicyLocationFactory(set1):
    """
    Gives a policy function defined on the two parameters. The default action is to suppress the fire.
    :param left: Boolean indicates whether this if for the first set of trajectories (True) or the second (False).

    See: https://github.com/smcgregor/FireWoman/blob/guide-for-logistic-policy/FireAp/Policy.cpp#L440
         https://github.com/smcgregor/FireWoman/blob/guide-for-logistic-policy/FireAp/simulator/ManagementZone.cpp#L684

    :return: A function mapping states to actions
    """
    def policy_location(state, possibleActions, transitionTuple=None, set1=set1):
        numCols = 1127
        pixelNumber = transitionTuple.additionalState["ignitionLocation"]
        xCoordinate = int(pixelNumber/numCols)
        assert pixelNumber >= 0, "pixelNumber was {}".format(pixelNumber)
        assert pixelNumber <= 1059380, "pixelNumber was {}".format(pixelNumber)
        assert xCoordinate >= 0, "yCoordinate was {}".format(xCoordinate)
        assert xCoordinate <= 940, "xCoordinate was {}".format(xCoordinate)
        if xCoordinate < 470:
            if set1:
                return 0
            else:
                return 1
        else:
            if set1:
                return 1
            else:
                return 0
    return policy_location

