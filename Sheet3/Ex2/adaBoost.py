from decision_tree_utils import *
import random


class AdaBoost:

    def __init__(self, training_set, attributes, C):
        self.training_set = training_set
        self.N = len(training_set)
        self.attributes = attributes
        self.C = C

        self.weights = []
        # Assign equal weight the every training instance.
        for x in range(0, self.N):
            self.weights.append(1.0 / self.N)
        self.weightsSum = sum(self.weights)

        self.models = []
        self.errors = []
        self.input_data = training_set

        self.base_classifier_depth = 2 ** 31

    # a)
    def sample(self):
        # sample the input data from the training set, based on weight
        self.input_data = []
        for i in range(0, self.N):
            index = AdaBoost.weighted_choice(range(0, self.N), self.weights)
            self.input_data.append(self.training_set[index])

        return self.input_data

    # b)
    # input_data is taken from class variables, instead of as an argument for this function
    def model_generation(self, iterations):

        self.sample()

        # for each of t iterations
        for x in range(0, iterations):

            # apply the learning algorithm
            tree = trainModel(self.input_data, self.C, self.attributes, self.base_classifier_depth)

            # calculate the error
            ee = []
            for i in range(0, self.N):
                instance = self.training_set[i]
                index = AdaBoost.get_attribute_index(self.attributes, self.C)
                actual = instance[index]

                if actual == predict(tree, instance, self.attributes):
                    ee.append(0)
                else:
                    ee.append(1)

            e = sum([a*b for a,b in zip(ee,self.weights)])

            # if e = 0 or > 0.5, terminate model generation
            if e == 0 or e > 0.5:
                self.sample()
                continue

            # else save the generated model
            self.models.append(tree)
            self.errors.append(e)

            # for each instance in the data set
            for i in range(0, self.N):
                instance = self.training_set[i]

                index = AdaBoost.get_attribute_index(self.attributes, self.C)
                actual = self.training_set[i][index]

                # if instance was classified correctly by model
                if actual == predict(tree, instance, self.attributes):

                    # multiply weight of instance by e / (1 - e)
                    self.weights[i] *= e / (1 - e)

            # normalize the weights
            temp_sum = sum(self.weights)
            for y in range(0, self.N):
                self.weights[y] /= temp_sum
                self.weights[y] *= self.weightsSum

            # take new instances, based on weight
            self.sample()

    # c)
    def classification(self, instance):
        # assign weight of 0 to all classes
        class_weights = []

        # for each of the t (or less) models
        j = 0
        occurrences = []
        for tr in self.models:

            # add -log(e / (1 - e)) to weight of class predicted by model.
            predicted_attribute = predict(tr, instance, self.attributes)
            e = self.errors[j]
            if predicted_attribute not in occurrences:
                occurrences.append(predicted_attribute)
                class_weights.append(- math.log(e / (1 - e), 2))
            else:
                pred = occurrences.index(predicted_attribute)
                class_weights[pred] -= math.log(e / (1 - e), 2)
            j += 1

        # return class with highest weight.
        t = class_weights.index(max(class_weights))
        return occurrences[t]

    # set the base classifiers depth for task g)
    def set_base_classifier_depth(self, depth):
        self.base_classifier_depth = depth

    # auxiliary function
    @staticmethod
    def get_attribute_index(meta, C):
        j = 0
        for i in meta:
            if i == C:
                return j
            j += 1
        return -1

    # auxiliary function
    @staticmethod
    def weighted_choice(choices, weights):
        total = sum(weights)
        treshold = random.uniform(0, total)
        for k, weight in enumerate(weights):
            total -= weight
            if total < treshold:
                return choices[k]

    # d)
    @staticmethod
    def weather_test(iterations):
        print "Exercise 2 d) - testing on weather.nominal.arff with " + str(iterations) + " iterations(trees)..."
        dataset = parseARFF('res/weather.nominal.arff')
        target = 'play'
        attributes = dataset[0]  # extract attributes from the loaded dataset
        samples = dataset[1:]  # extract samples from the loaded dataset
        m = AdaBoost(samples, attributes, target)
        m.model_generation(iterations)

        test = ['rainy', 'cool', 'normal', 'FALSE', 'yes']
        print "instance :" + str(test)
        print "prediction: " + str(m.classification(test))
        # should be 'yes' (most of the time)

    # e)
    @staticmethod
    def car_test(iterations, max_depth=2**31):
        print "Testing on car.arff with " + str(iterations) + " iterations (trees)..."
        dataset = parseARFF('res/car.arff')
        target = 'class'

        attributes = dataset[0]
        train = dataset[1:]
        test = []

        # split the dataset into train and test sets
        t = int(len(train) / 3)
        j = 0
        for i in range(0, t):
            index = random.choice(range(0, len(train) - j))
            test.append(train[index])
            train.pop(index)
            j += 1

        # build the adaBoost classifier
        m = AdaBoost(train, attributes, target)
        m.set_base_classifier_depth(max_depth)
        m.model_generation(iterations)

        # calculate the accuracy on the test set
        index = AdaBoost.get_attribute_index(attributes, target)
        counter = 0
        for sample in test:
            pred = m.classification(sample)
            actual = sample[index]
            if pred == actual:
                counter += 1

        accuracy = float(counter) / len(test)
        print "accuracy: " + str(accuracy)
        return accuracy

    # f)
    @staticmethod
    def car_test_mean_deviation(iterations):
        print "Exercise 2 f)..."
        repetitions = 10
        s = 0
        accs = []
        for i in range(0, 10):
            acc = AdaBoost.car_test(iterations)
            accs.append(acc)
            s += acc

        mean = s / 10
        print "\nMean for " + str(repetitions) + " repetitions with " + str(iterations) + " trees: " + str(mean)

        difs = []
        for acc in accs:
            difs.append((mean - acc) ** 2)

        standard_deviation = math.sqrt(sum(difs) / 10)
        print "standard deviation: " + str(standard_deviation)

    # g)
    @staticmethod
    def classifier_test():
        print "Exercise 2 g)..."
        upto = 10
        s = 0
        accs = []
        for i in range(0, upto):
            acc = AdaBoost.car_test(10, i)
            accs.append(acc)

        print "\nAccuracies for base-classifier-depth from [1..10]: "
        print accs


if __name__ == '__main__':
    # d)
    AdaBoost.weather_test(10)

    print '-'*10 + "\n"

    # e)
    print "Exercise 2 e)"
    AdaBoost.car_test(10)

    print '-' * 10 + "\n"

    # f)
    # each with 3 trees, since it takes a while
    AdaBoost.car_test_mean_deviation(3)

    print '-' * 10 + "\n"

    # g)
    # this might take a bit
    AdaBoost.classifier_test()
    print "The results show that an increase in maximum base classifier depth, " \
          "only has a noticeable impact until about the fifth try."
    print "One possible reason for this is, that there are only few attributes and possible classes " \
          "in the car.arff data set."



