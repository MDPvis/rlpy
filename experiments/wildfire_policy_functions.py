"""
A set of policy functions used to evaluate visual fidelity.
"""

databasePolicyParameters = {
    "ercThreshold": [50], # todo, change to real values
    "timeUntilEndOfFireSeasonThreshold": [25], # todo, change to real values
    "leftBoundary": [], # todo, change to real values
    "topBoundary": [] # todo, change to real values
}

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
        erc = transitionTuple.additionalState["erc"]
        timeUntilEndOfFireSeason = transitionTuple.additionalState["time until end"]
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

def wildfirePolicyLocationFactory(leftBoundary, topBoundary):
    """
    Gives a policy function defined on the two parameters. The default action is to suppress the fire.
    :param leftBoundary: All fires to the left are allowed to burn.
    :param topBoundary: All fires above the boundary are allowed to burn.
    :return: A function mapping states to actions
    """

    def policy_location(state, possibleActions, leftBoundary=leftBoundary, topBoundary=topBoundary):
        xCoordinate = state[0]
        yCoordinate = state[0]
        assert xCoordinate >= 0, "xCoordinate was {}".format(xCoordinate)
        assert xCoordinate <= 1127, "xCoordinate was {}".format(xCoordinate)
        assert yCoordinate >= 0, "yCoordinate was {}".format(yCoordinate)
        assert yCoordinate <= 940, "yCoordinate was {}".format(yCoordinate)
        if xCoordinate >= leftBoundary and yCoordinate <= topBoundary:
            return 1
        else:
            return 0
    return policy_location

