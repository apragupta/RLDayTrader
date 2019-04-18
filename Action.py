from enum import Enum

Actions = Enum('Actions', 'buy sell hold')


class Action:

    def __init__(Action,action,amount):
        Action.action = action
        Action.amount = amount

    def __str__(self):
        return "action:" + str(self.action.name) + "\n" + "amount:" + str(self.amount)