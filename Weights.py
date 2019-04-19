class Weights:
    def __init__(Weights, accountWeight, numStocksWeight, pricesWeights, trendWeight):
        Weights.accountWeight = accountWeight
        Weights.numStocksWeight = numStocksWeight
        Weights.pricesWeights = pricesWeights
        Weights.trendWeight = trendWeight

    def __str__(self):
        return "accountWeight:" + str(self.accountWeight) + "\n" + "numstocksWeight:" + str(self.numStocksWeight) + \
               "\n" \
               + "pricesWeights:" + str(self.pricesWeights)




