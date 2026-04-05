from enum import Enum


class ActionType(str, Enum):
    PLAN = "PLAN"
    #REPLAN = "REPLAN"
    SEARCH_WIKIPEDIA = "SEARCH_WIKIPEDIA"
    COMPUTE_SVT = "COMPUTE_SVT"
    ASK_USER = "ASK_USER"
    FINISH = "FINISH"
