import numpy as np
import pandas as pd
import Action as act
from sklearn.preprocessing import PolynomialFeatures
import betterWeights as w
import matplotlib.pyplot as plt

data_SP = pd.read_csv("data/S&P500.csv", delimiter=";")
data_SP = data_SP.loc[:,:'UO']
data_SP = data_SP[29:1599]
# print(data_SP)
train_SP = data_SP[0:1300]
# print(train_SP)
test_SP = data_SP[1300:1599]

train_SP = train_SP.drop(columns=['Class','Date'])
test_SP = test_SP.drop(columns=['Class','Date'])
num_features = len(train_SP.columns)
closing = train_SP.loc[:,'Closing Price']
print(closing)
train_SP = train_SP.values
train_SP = np.array(train_SP)
train_SP = train_SP.astype(np.float)
#print(train_SP[0])


test_SP = test_SP.values


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
def normalize(list_to_normalize):
    max_x = max(list_to_normalize)
    min_x = min(list_to_normalize)
    range = max_x - min_x
    if range != 0:
        norm_list = [(2*(item-min_x)/range)-1 for item in list_to_normalize]
    elif min_x==0 or max_x ==0:
        norm_list = [0 for item in list_to_normalize]
    else:
        norm_list = [1 for item in list_to_normalize]
    return norm_list




class State:
    def __init__(State, account, index, features, numFeatures,degree):
        State.account = account
        State.index = index
        features = features.tolist()
        features.insert(0,account)
        State.features = features
        State.numFeatures = numFeatures
        State.degree = degree
        process_weights = [features]
        poly = PolynomialFeatures(degree)
        poly_features = poly.fit_transform(process_weights)
        State.poly_features = poly_features[0]

    def __str__(self):
        return "index:" + str(self.index) + "\n" + "account:" + str(self.account) + "\n" \
               + "Features: \n" + str(self.features)\
            +   "numFeatures:" + str(self.numFeatures)

    def getNextState(self, action, newFeatures):
        newAccount = self.account
        newOpening = newFeatures[0]
        newClosing = newFeatures[1]

        if action.action == act.Actions.buy:
            newAccount += (newClosing-newOpening)*action.amount
        return State(newAccount, self.index + 1,  newFeatures, self.numFeatures, self.degree)

    def calcReward(self, action, newFeatures):
        newOpening = newFeatures[0]
        newClosing = newFeatures[1]
        reward = 0
        if action.action == act.Actions.buy:
            reward = (newClosing-newOpening)*action.amount

        return reward
    def calcQvalue(self, weights):
        #print(len(weights.weights))
        return np.dot(self.poly_features,weights.weights)

    def maxQAndAction(self, weights, actions, newFeatures):
        bestAction = None
        max = -2.2250738585072014e308
        for action in actions:
            nextState = self.getNextState(action, newFeatures)
            nextQ = nextState.calcQvalue(weights)
            if nextQ > max:
                max = nextQ
                bestAction = action
        return max, bestAction

    def calcDifference(self, action, actions, weights, newFeatures, new_newFeatures, gamma):
        oldQ = self.calcQvalue(weights)
        #print("oldQ:" + str(oldQ))
        reward = self.calcReward(action,newFeatures)

        nextState = self.getNextState(action,newFeatures)

        maxQsprime, bestAction = nextState.maxQAndAction(weights,actions,new_newFeatures)
        #print("expQ:" + str((reward + gamma * maxQsprime)))
        return (reward + gamma * maxQsprime) - oldQ
    def updateWeights(self, oldWeights, alpha, difference):
        newWeights = [sum(x) for x in zip(oldWeights.weights, [alpha*difference*self.poly_features])]
        #print(len(newWeights[0]))
        return w.Weights(normalize(newWeights[0]),oldWeights.numFeatures,oldWeights.numWeights,self.degree)

    def EpsilonPolicy(self, epsilon, actions, weights, newFeatures):
        choices = ['best', 'random']
        max, bestA = self.maxQAndAction(weights, actions, newFeatures)
        choice = np.random.choice(choices, p=[(1 - epsilon), epsilon])
        if choice == 'best':
            return bestA
        else:
            withoutBest = actions[:]
            withoutBest.remove(bestA)
            return np.random.choice(withoutBest)

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

def plotChoices(closing, choices):
    plt.plot(closing)
    count = 0
    for i in closing:
        if count >= len(choices):
            break
        elif choices[count].action == act.Actions.buy:
            plt.scatter(count, i, c='green')
        # elif choices[count].action == act.Actions.sell:
        #     plt.scatter(count, i, c='red')
        elif choices[count].action == act.Actions.hold:
             plt.scatter(count, i, c='blue')
        count += 1
    plt.show()

def QLearn(episodes, epsilon, startingAccount, gamma, alpha, actions, degree, num_features):
    np.random.seed(50)
    #initializing weights to 0

    feature_weights = np.zeros(num_features+1)
    feature_weights = [feature_weights]
    poly = PolynomialFeatures(degree)
    weights = poly.fit_transform(feature_weights)
    weights = weights[0]
    weights[0] = 0
    num_weights = len(weights)
    #print(num_weights)
    weights = w.Weights(weights,num_features+1,num_weights,degree)
    epsilon_change = epsilon / episodes
    states = []
    choices = []

    for i in range(episodes):
        state = State(startingAccount,0,train_SP[0],num_features,degree)
        #print(str(state))
        for j in range(len(train_SP)):
            if j <  (len(train_SP)-2):
                nextFeatures = train_SP[j+1]
                next_nextFeatures = train_SP[j+2]
                action = state.EpsilonPolicy(epsilon,actions,weights,nextFeatures)
                if i == episodes - 1:
                    choices.append(action)
                nextState = state.getNextState(action,nextFeatures)
                #print(str(nextState))
                difference = state.calcDifference(action, actions, weights,nextFeatures, next_nextFeatures,gamma)
                weights = state.updateWeights(weights,alpha,difference)
                state = nextState
        epsilon -=epsilon_change
        print("Episode: " + str(i) + ", Account: " + str(state.account))
        states.append(state)
    plotBalances(states)
    plotChoices(closing,choices)








#print(normalize(train_SP[0]))
# degree = 2
# testState = State(1000,1,train_SP[0],num_features+1,degree)
buy = act.Action(act.Actions.buy, 1)
hold = act.Action(act.Actions.hold, 1)
actions = [buy, hold]

QLearn(200,0.6,10000,0.8,0.0001,actions,1,26)











# feature_weights = np.zeros(num_features+1)
# feature_weights = [feature_weights]
# poly = PolynomialFeatures(2)
# weights = poly.fit_transform(feature_weights)
# weights = weights[0]
# weights[0] = 0
# num_weights = len(weights)
# #
# #
# #
# testWeights = w.Weights(weights,num_features+1,num_weights,degree)
# print(str(testWeights))
# print(str(testState.calcQvalue(testWeights)))
# nextState_buy = testState.getNextState(buy,train_SP[1])
# nextState_hold = testState.getNextState(hold,train_SP[1])
# Q_buy = nextState_buy.calcQvalue(testWeights)
# Q_hold = nextState_hold.calcQvalue(testWeights)
# dif_buy = testState.calcDifference(buy,actions,testWeights,train_SP[1],train_SP[2],0.9)
# dif_hold = testState.calcDifference(hold,actions,testWeights,train_SP[1],train_SP[2],0.9)
# print(str(dif_buy))
# print(str(dif_hold))
# print("updted weigth:" + str(testState.updateWeights(testWeights,0.001,dif_buy)))
# # print(str(testState.updateWeights(testWeights,0.001,dif_hold)))
#
#
#
# print(str(testState.maxQAndAction(testWeights,actions,train_SP[1])))
#
#
# print(str(testState.calcReward(buy,train_SP[1])))
# print(str(testState.calcReward(hold,train_SP[1])))







#print(str(testState))

#print(str(testState.getNextState(buy,train_SP[1]).calcQvalue(testWeights,degree)))
# weights = w.Weights(np.ones(15),4,15,2)
#print(str(testState.getNextState(hold,train_SP[1]).calcQvalue(testWeights,degree)))
#
# print(str(testState.calcQvalue(weights,2)))


