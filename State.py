import numpy as np
import pandas as pd
import Action as act
import Weights as w
import matplotlib.pyplot as plt


data = pd.read_csv("data/GSPC.csv")
closing_prices = data.loc[:, "Close"]
v = closing_prices.values


data_eval = pd.read_csv("data/GSPC_2011.csv")
closing_prices_eval = data_eval.loc[:, "Close"]
v_eval = closing_prices_eval.values


def getLastNPrices(numDays, row, pricesData):
    lastNPrices = np.zeros(numDays)
    avg = 0
    # calculate avg only if we need it
    if numDays == 1:
        return [pricesData[row]]
    if row == 0:
        return [i + pricesData[0] for i in lastNPrices]
    if row <= numDays:
        for j in range(row):
            avg = pricesData[j] + avg
        avg = avg / row
    for i in range(numDays):
        if row - numDays + i + 1 >= 0:
            lastNPrices[i] = pricesData[row - numDays + i + 1]
        else:
            lastNPrices[i] = avg
    return lastNPrices.tolist()


# TODO change to include stuff other than just closing prices
class State:
    def __init__(State, account, index, numStocks, lastNClosingPrices, trend):
        # for normalization:
        #max_x = max(account,index,numStocks,max(lastNClosingPrices))
        #min_x = min(account,index,numStocks,max(lastNClosingPrices))
        #(2 * (account - min_x) / (max_x - min_x)) - 1
        State.account = account
        State.index = index
        State.numStocks = numStocks
        State.lastNClosingPrices = lastNClosingPrices
        State.trend = trend

    # Returns array of last N prices starting at (including) index row from array of prices v
    # if numDays is 0 then it just gives today's price
    def __str__(self):
        return "index:" + str(self.index) + "\n" + "account:" + str(self.account) + "\n" \
               + "numStocks:" + str(self.numStocks) + "\n" \
               + "lastNClosingPrices: \n" + str(self.lastNClosingPrices)

    # returns the next state given the action and the closing price of the next day
    # TODO change to include stuff other than just closing prices
    def getNextState(self, action, nextPrice):
        newClosingPrices = self.lastNClosingPrices[1:]
        newClosingPrices.append(nextPrice)
        newAccount = self.account
        newNumStocks = self.numStocks
        trend = (newClosingPrices[0] - newClosingPrices[-1]) / len(newClosingPrices)  # trend is slope between first and last day
        if action.action == act.Actions.buy:
            newAccount -= self.lastNClosingPrices[-1] * action.amount
            newNumStocks += action.amount
        elif action.action == act.Actions.sell:
            newAccount += self.lastNClosingPrices[-1] * action.amount
            newNumStocks -= action.amount
        return State(newAccount, self.index + 1, newNumStocks, newClosingPrices, trend)

    # the reward is the increase in the value of your portfolio
    # returns reward received if the given action is carried out on this state
    # TODO explore possible better reward functions?
    def calcReward(self, action, nextPrice, startingPortfolio):
        nextState = self.getNextState(action, nextPrice)
        holdingPenalty = 0
        if action.action == act.Actions.hold:
            holdingPenalty = 0
        return holdingPenalty + ((nextState.account + (nextState.numStocks * nextState.lastNClosingPrices[-1])) - startingPortfolio)

    def calcQvalue(self, weights):
        return (weights.accountWeight * self.account) + (weights.numStocksWeight * self.numStocks) \
                + np.dot(self.lastNClosingPrices, weights.pricesWeights) + (weights.trendWeight * self.trend)

    # returns the best action and maxQ value for this state (Q value if that action was applied to it),takes in list
    # possible actions given current weights
    def maxQAndAction(self, weights, actions, nextPrice):
        bestAction = None
        max = -2.2250738585072014e308
        for action in actions:
            nextState = self.getNextState(action, nextPrice)
            nextQ = nextState.calcQvalue(weights)
            if nextQ > max:
                max = nextQ
                bestAction = action
        return max, bestAction

    def calcDifference(self, nextPrice, action, actions, next_nextPrice, weights, startingPortfolio, gamma):
        oldQ = self.calcQvalue(weights)
        reward = self.calcReward(action, nextPrice, startingPortfolio)
        nextState = self.getNextState(action, nextPrice)
        maxQsprime, bestAction = nextState.maxQAndAction(weights, actions, next_nextPrice)
        return (reward + gamma * maxQsprime) - oldQ

    def updateWeights(self, oldWeights, alpha, difference):
        newAccountWeight = oldWeights.accountWeight + (alpha * difference * self.account)
        newNumStocksWeight = oldWeights.numStocksWeight + (alpha * difference * self.numStocks)
        newPricesWeights = [sum(x) for x in zip(oldWeights.pricesWeights, [alpha * difference * price for price in self.lastNClosingPrices])]
        newTrendWeight = oldWeights.trendWeight + (alpha * difference * self.trend)
        max_x = max(newAccountWeight, newNumStocksWeight, max(newPricesWeights), newTrendWeight)
        min_x = min(newAccountWeight, newNumStocksWeight, min(newPricesWeights), newTrendWeight)
        range = max_x - min_x
        if range != 0:
            norm_newpricesweight = [(2 * (priceweight - min_x) / range) - 1 for priceweight in newPricesWeights]
            return w.Weights((2 * (newAccountWeight - min_x) / range) - 1, (2 * (newNumStocksWeight - min_x)/range) - 1, norm_newpricesweight, (2 * (newTrendWeight - min_x)/range) - 1)
        elif min_x == 0 and max_x == 0:
            norm_newpricesweight = [0 for priceweight in newPricesWeights]
            return w.Weights(0, 0, norm_newpricesweight, 0)
        norm_newpricesweight = [1 for priceweight in newPricesWeights]
        return w.Weights(1, 1, norm_newpricesweight, 1)

    # returns action based on epsilon
    def EpsilonPolicy(self, epsilon, actions, weights, nextPrice):
        choices = ['best', 'random']
        choice = np.random.choice(choices, p=[(1 - epsilon), epsilon])
        if choice == 'best':
            max, bestA = self.maxQAndAction(weights, actions, nextPrice)
            return bestA
        else:
            withoutBest = actions[:]
            # withoutBest.remove(bestA)
            return np.random.choice(withoutBest)

    # Chooses the best action to take from a given state
    def chooseAction(self, epsilon, actions, weights, v, state, j):
        possibleActions = []
        for a in actions:
            if not ((a.action == act.Actions.sell) and (a.amount > state.numStocks)):
                possibleActions.append(a)
        action = state.EpsilonPolicy(epsilon, possibleActions, weights, v[j + 1])
        return action


def plotChoices(v, choices):
    plt.plot(v)
    count = 0
    for i in v:
        if count >= len(choices):
            break
        elif choices[count].action == act.Actions.buy:
            plt.scatter(count, i, c='green')
        elif choices[count].action == act.Actions.sell:
            plt.scatter(count, i, c='red')
        # elif choices[count].action == act.Actions.hold:
        #     plt.scatter(count, i, c='blue')
        count += 1
    plt.show()


def plotBalances(states):
    x = []
    y = []
    count = 1
    for s in states:
        y.append(s.account)
        x.append(count)
        count += 1
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel('balance')
    plt.title('Learning Curve')
    plt.show()


# def plotWeights(weights, episodes):
#     all_weights = []
#     x = []
#     # for w in weights:
#     #     tempWeights = []
#     #     tempWeights.append(w.accountWeight)
#     #     tempWeights.append(w.numStocksWeight)
#     #     for p in w.pricesWeights:
#     #         tempWeights.append(p)
#     #     tempWeights.append(w.trendWeight)
#     #     all_weights.append(tempWeights)
#     accountWeights = []
#     numStocksWeights = []
#     pricesWeightsWeights = []
#     trendWeights = []
#     for w in weights:
#         accountWeights.append(w.accountWeight)
#         numStocksWeights.append(w.numStocksWeight)
#         pricesWeightsWeights.append(w.pricesWeights)
#         trendWeights.append(w.trendWeight)
#     all_weights.append(accountWeights)
#     all_weights.append(numStocksWeights)
#     for p in pricesWeightsWeights:
#         all_weights.append(p)
#     all_weights.append(trendWeights)
#     # print(len(all_weights))
#     # print(len(all_weights[0]))
#     for i in range(episodes):
#         x.append(i)
#     # print(len(x))
#     for w in all_weights:
#         print(len(w))
#         print(w)
#     count = 0
#     for w in all_weights:
#         plt.plot(x, w, label=count)
#         count += 1
#     plt.legend(loc="lower left")
#     plt.xlabel('episode')
#     plt.ylabel('weighting')
#     plt.title('Weights')
#     plt.show()


def QLearn(episodes, epsilon, gamma, alpha, startingAccount, n, actions):
    np.random.seed(50)
    weights = w.Weights(0, 0, [0 for i in range(n)], 0)
    epsilon_change = epsilon / episodes
    choices = []
    states = []
    all_weights = []
    for i in range(episodes):
        lastNPrices = getLastNPrices(n, 0, v)
        trend = (lastNPrices[0] - lastNPrices[-1]) / len(lastNPrices)
        state = State(startingAccount, 0, 0, lastNPrices, trend)
        for j, price in enumerate(v):
            if j < len(v) - 2:
                action = state.chooseAction(epsilon, actions, weights, v, state, j)
                if i == episodes - 1:
                    choices.append(action)
                nextPrice = v[j + 1]
                nextState = state.getNextState(action, nextPrice)
                difference = state.calcDifference(nextPrice, action, actions, v[j+2], weights, startingAccount, gamma)
                weights = state.updateWeights(weights, alpha, difference)
                state = nextState
        epsilon -= epsilon_change
        if i % 10 == 0:
            print("Episode: " + str(i) + ", Account: " + str(state.account) + ", numStocks: " + str(state.numStocks))
        states.append(state)
        all_weights.append(weights)
    plotChoices(v, choices)
    plotBalances(states)
    # plotWeights(all_weights, episodes)
    return weights


def evaluate(weights, startingAmount, n, actions):
    lastNPrices = getLastNPrices(n, 0, v_eval)
    trend = (lastNPrices[0] - lastNPrices[-1]) / len(lastNPrices)
    state = State(startingAmount, 0, 0, lastNPrices, trend)
    choices = []
    cantSell = [x for x in actions if x.action != act.Actions.sell]
    for j, price in enumerate(v_eval):
        if j < len(v_eval) - 2:
            nextPrice = v_eval[j + 1]
            if state.numStocks > 0:
                max, action = state.maxQAndAction(weights, actions, nextPrice)
            else:
                max, action = state.maxQAndAction(weights, cantSell, nextPrice)
            choices.append(action)
            state = state.getNextState(action, nextPrice)
    print("account_eval:" + str(state.account))
    plotChoices(v_eval, choices)


buy = act.Action(act.Actions.buy, 1)
sell = act.Action(act.Actions.sell, 1)
hold = act.Action(act.Actions.hold, 1)
actions = [buy, sell, hold]
# for n in range(2, 3):
#     actions.append(act.Action(act.Actions.buy, n))
#     actions.append(act.Action(act.Actions.sell, n))
#     actions.append(act.Action(act.Actions.hold, n))
# weights = w.Weights(0.5, -1, [1, 3, 4, 6, 7])
# lastNPrices = getLastNPrices(5, 5, v)
# currentState = State(10000, 5, 0, lastNPrices)

evaluate(QLearn(7, 0.9, 0.1, 0.001, 10000, 5, actions), 10000, 5, actions)


# print(currentState)
# weights = w.Weights(0.5, -1, [1, 3, 4, 6, 7])
# print(weights)
# maxQ, bestA = currentState.maxQAndAction(weights, actions, v[6])
# print(str(maxQ), str(bestA))
#
# print(currentState.getNextState(buy, v[6]))
# print("buy:" + str(currentState.getNextState(buy, v[6]).calcQvalue(weights)))
# print("sell:" + str(currentState.getNextState(sell, v[6]).calcQvalue(weights)))
# print("hold:" + str(currentState.getNextState(buy, v[6]).calcQvalue(weights)))
#
# print(currentState.calcReward(buy, v[1], 10000))
# print(currentState.calcReward(sell, v[1], 10000))
#
# #print(currentState.calcQvalue(weights))
# dif = currentState.calcDifference(v[6], sell, actions, v[7], weights, 10000, 0.9)
# print("difference:" + str(dif))
# print(str(currentState.updateWeights(weights, 0.01, dif)))
