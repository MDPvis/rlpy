class TransitionTuple(tuple):
    """
    A Simple tuple class for storing state transitions in the Ball tree.
    The object holds the tuple for the pre-transition state that will be stitched to
    in the current post-transition state. The class contains properties not in the
    tuple:
    """
    def __new__(cls, preStateDistanceMetricVariables, postStateDistanceMetricVariables,
                isTerminal, isInitial, possibeActions, visualizationResultState=None):
        """
        :param cls: The _new_ constructor's version of `self`
        :param preStateDistanceMetricVariables: The state we might stitch to, this is also represented as a tuple.
          These include the action indicators.
        :param postStateDistanceMetricVariables: What state resulted from the pre-transition state.
          These do *not* include the action indicators.
        :param isTerminal: An indicator for whether the transitioned to state is terminal.
        :param isInitial: An indicator for whether the pre-transition state is an initial state.
        :param possibeActions: What actions can be taken in the resulting state.
        :param visualizationResultState: The variables the stitching domain are attempting to approximate.
          These can include actions if we are trying to approximate action distributions.
        :return: this extended tuple
        """
        t = tuple.__new__(cls, tuple(preStateDistanceMetricVariables))
        t.preStateDistanceMetricVariables = preStateDistanceMetricVariables
        t.postStateDistanceMetricVariables = postStateDistanceMetricVariables
        t.isTerminal = isTerminal
        t.isInitial = isInitial
        t.possibleActions = possibeActions
        t.visualizationResultState = visualizationResultState
        t.last_accessed_iteration = -1 # determines whether it is available for stitching
        return t