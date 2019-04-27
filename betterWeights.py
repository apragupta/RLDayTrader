class Weights:
    def __init__(Weights, weights, numFeatures, numWeights, degree):
        Weights.weights = weights
        Weights.numFeatures = numFeatures
        Weights.numWeights = numWeights
        Weights.degree = degree

    def __str__(self):
        return "weights \n" + str(self.weights) + "\n" + "numFeatues:" + str(self.numFeatures) + \
               "\n" \
               + "numWeights:" + str(self.numWeights) + "\n"\
               + "degree:" + str(self.degree)