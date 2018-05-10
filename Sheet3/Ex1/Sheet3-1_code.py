import random
import argparse
import traceback
import weka.core.jvm as jvm

from weka.classifiers import Classifier
from weka.core.dataset import Instances
from weka.core.converters import Loader


def split_data(data, test_size): # split the data
    # create placeholder for train split
    data_train = Instances.copy_instances(data)
    # remove all instances from the placeholder
    for i in reversed(range(len(data_train))):
        data_train.delete(i)

    # create placeholder for test split
    data_test = Instances.copy_instances(data)
    # remove all instances from the placeholder
    for i in reversed(range(len(data_test))):
        data_test.delete(i)

    # create list of indices
    indices = list(range(len(data)))
    # shuffle indices
    random.shuffle(indices)
    # calculate number of indices in the test split
    num_test = int(round(len(indices) * test_size, 0))

    # get indices for the test split
    test_ids = indices[:num_test]
    # fill test split with instances
    for idx in test_ids:
        data_test.add_instance(data.get_instance(idx))

    # get indices for the train split
    train_ids = indices[num_test:]
    # fill train split with instances
    for idx in train_ids:
        data_train.add_instance(data.get_instance(idx))

    return data_train, data_test


def train(data_train, n_estimators): # train the model
    # create `Classifier` object
    rf = Classifier(
        classname="weka.classifiers.trees.RandomForest",
        options=[
            '-num-slots', '0',
            '-I', str(n_estimators)
        ]
    )

    # train classifier on the train split
    rf.build_classifier(data_train)

    return rf


def predict(rf, data_test): # test the model
    predicted = list()
    for inst in data_test:
        # classify each instance
        y = rf.classify_instance(inst)
        # put predicted class label into the list
        predicted.append(y)

    return predicted


def main(path, num_trees):
    loader = Loader(classname='weka.core.converters.ArffLoader') # load the data
    ds = loader.load_file(path)
    ds.class_is_last()

    accuracy = list()
    for i in range(10):
        random.seed(i)
        data_train, data_test = split_data(ds, 1.0 / 3) # split it

        labels_test = [inst.values[inst.class_index] for inst in data_test]

        rf = train(data_train, n_estimators=num_trees) # train the model
        predicted = predict(rf, data_test) # test the model

        # compute the accuracy of correctly classified instances
        num_correct = sum([1.0 for y, gt in zip(predicted, labels_test) if y == gt])
        accuracy.append(num_correct / len(labels_test))

    # compute the pecrcentage of correctly classified instances (averaged result)
    acc_mean = 1.0 * sum(accuracy) / len(accuracy)
    acc_std = (sum((x - acc_mean) ** 2 for x in accuracy) / len(accuracy)) ** 0.5

    print('Maximum number of trees: {}\n'.format(num_trees))
    print('Accuracy (mean={:.2%}, std={:.2%}):\n{}'.format(
        acc_mean,
        acc_std,
        ', '.join('{:.2%}'.format(acc) for acc in accuracy)
    ))
    print('-' * 120)


if __name__ == '__main__':
    try:
        jvm.start()

        def check_positive(value):
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
            return ivalue

        # user's input
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--path', type=str, help='Path to the dataset (*.arff)', required=True)
        parser.add_argument('-t', '--num_trees', type=check_positive, default=1, help='Number of trees in RandomForest')

        args = parser.parse_args()
        path = args.path
        num_trees = args.num_trees

        main(path, num_trees)
        #for num_trees in [1, 5, 10, 20, 30, 40, 50]:
        #    main(path, num_trees)
    except:
        traceback.print_exc()
    finally:
        jvm.stop()

