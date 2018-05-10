import math
import random

def entropyOnSubset(dataset, indices, C):
    s = [dataset[i+1] for i in indices]  # offset because of the header row
    c = dataset[0].index(C)  # column index of class attribute in the dataset
    v = [row[c] for row in s]  # values(C)
    freq = {k : float(v.count(k)) for k in v} # frequency table
    return sum([-f/len(v) * math.log(f/len(v), 2) for f in freq.values()]) # entropy


def informationGain(dataset, indices, C, A):
    s = [dataset[i+1] for i in indices]
    a = dataset[0].index(A)
    v = [row[a] for row in s]  # values(A)
    freq = {k : float(v.count(k)) for k in v}
    entropy = 0.0
    for f in freq:  # entropy for each subset
        subset_indices = [i for i, row in enumerate(s) if row[a] == f]
        entropy += (freq[f]/sum(freq.values())) * entropyOnSubset(dataset, subset_indices, C)
    return entropyOnSubset(dataset, indices, C) - entropy


def parseARFF(filepath):
    headers = []
    data = []
    for line in open(filepath):
        if line.startswith("%"):
            continue
        if line.startswith("@attribute"):
            headers.append(line.strip().split()[1])
        elif not line.startswith("@") and len(line.strip()) > 0:
            data.append(line.strip().split(','))
    return [headers] + data

class Node(object): #represents single node of a tree built by ID3 algorithm
    def __init__(self, label, value):
        """
        Node constructor
        args:
            - label: name of attribute associated with current node
            - value: value of the corresponding attribute
        """
        super(Node, self).__init__()
        self._label = label
        self._value = value
        self._children = list()

    @property
    def label(self): #get label
        return self._label
    
    @label.setter
    def label(self, value): #set it
        self._label = value
    
    @property
    def value(self): #get value
        return self._value
    
    @value.setter
    def value(self, value): #set it
        self._value = value
    
    @property
    def children(self): #return child nodes (Node) of the current node
        return self._children
    
    def add_child(self, child): #append child node (Node) to the current node and return current node
        self._children.append(child)
        return self
    
    def remove_child(self, child): #remove child node (Node) from the current node if it presents, return current node
        ids = [cid for cid, c in enumerate(self._children) if c == child]
        for child_id in reversed(ids):
            del self._children[child_id]
        return self

    def is_leaf(self): #if current node without the child one - return `True`
        return not self._children

    @staticmethod
    def inner_repr(instance, n_tabs=0):
        """
        Return string representation of the instance and its descendants
        
        args:
            - instance: root node
            - n_tabs: current instance's level
        """
        tabs = '\t' * n_tabs
        result = '{}"{}" -> "{}"'.format(tabs, instance._label, instance._value)
        if not instance.is_leaf():
            child_str = list()
            for c in instance.children:
                child_str.append(Node.inner_repr(c, n_tabs + 1))
            result += ': [\n{}\n{}]'.format('\n'.join(child_str), tabs)
        return result
    
    def __repr__(self): #override built-in method object.__repr__(self)
        return Node.inner_repr(self, 0)

def most_frequent(dataset, target_attr):
    """
    Return most frequent value of target attribute on given dataset
    
    args:
        - dataset: samples
        - target_attr: name of the target attribute (should be present in `dataset[0]`)
    """
    attributes = dataset[0]
    samples = dataset[1:]
    attr_values = [s[attributes.index(target_attr)] for s in samples]
    return max({attr: attr_values.count(attr) for attr in set(attr_values)}.items(), key=lambda x: x[1])[0]

def trainModel(samples, target, attributes, max_depth=2 ** 31, _depth=0):
    """
    Fit decision tree model
    
    args:
        - samples: samples
        - target: name of the target attribute (should be present in `attributes`)
        - attributes: attributes
        - max_depth: max depth of the tree
        - _depth: control depth of recursion
    """
    root = Node(target, None)  # create a root node
    target_id = attributes.index(target)  # get index for the target attribute
    if len(set(s[target_id] for s in samples)) == 1:  # if there is only one distinct value of target, return single-node tree containing this value
        root.value = samples[0][target_id]
        return root
    if all(a in {None, target} for a in attributes) or _depth >= max_depth:  # if no predictors left or we reached maximum recursion depth, return most frequent value of the target attribute
        root.value = most_frequent([attributes] + samples, target)
        return root

    IGs = list()  # calculate information gain for each attribute on given samples except the target
    for a in attributes:
        if a == target:
            continue
        ig = informationGain(
            [attributes] + samples,
            list(range(len(samples))),
            target,
            a
        )
        IGs.append(ig)
    max_ig = -1
    max_ig_idx = 0
    for i, ig in enumerate(IGs):  # find index of largest inf gain
        if ig > max_ig:
            max_ig = ig
            max_ig_idx = i
    A = attributes[max_ig_idx]  # get a name of attribute for the current node splitting
    root.label = A  # assign this attribute to the current node
    A_values = set(s[max_ig_idx] for s in samples)  # get distinct values of the attribute
    for v in A_values:  # loop over values of the found attribute
        subtree = Node(A, v)  # create subtree for each value
        samples_v = [s[:] for s in samples if s[max_ig_idx] == v]  # build subset filtering samples based on the attribute's value
        for i, _ in enumerate(samples_v):
            del samples_v[i][max_ig_idx]  # exclude found attribute from the subset
        if not samples_v:  # if we got empty subset - add to the subtree child node referring to the most frequent target
            subtree.add_child(Node(target, most_frequent([attributes] + samples, target)))
        else:
            child = trainModel(samples_v, target, [a for a in attributes if a != A], max_depth, _depth + 1)  # build tree for the subset
            if child.is_leaf():  # if we got a leaf node -  assign it to the subtree
                subtree.add_child(child)
            else:
                subtree.children.extend(child.children)  # assign its children to the subtree

        root.add_child(subtree)  # assign subtree to the root node
    return root

def trainModelOnSubset(samples, indices, target, attributes, max_depth=2 ** 31):
    """
    Fit decision tree model on the subset of given dataset
    args:
        - samples: samples
        - indices: indices to create subset of `samples`
        - target: name of the target attribute
        - attributes: attributes extracted from the dataset
        - max_depth: maximum depth of the decision tree
    """
    subset = [samples[idx] for idx in indices]
    return trainModel(subset, target, attributes, max_depth)

def predict(tree, sample, attributes):
    """
    Target class prediction
    
    args:
        - tree (Node): trained model (ID3)
        - sample: attributes associated with single sample
        - attributes: attributes from the dataset
    """
    if not tree.is_leaf():
        attr_id = attributes.index(list(set(c.label for c in tree.children))[0])

        for subtree in tree.children:
            if subtree.value == sample[attr_id]:
                return predict(subtree, sample, attributes)
    return tree.value

def evalModel(dataset, train_ids, test_ids, target, max_depth=2 ** 31):
    """
    Fit and evaluate model on given dataset
    args:
        - dataset: samples
        - train_ids: indices for training
        - test_ids: indices for evaluation
        - target: name of the target attribute (should be presented in `dataset[0]`)
        - max_depth: max depth of the decision tree
    """
    attributes = dataset[0]  # extract attributes
    target_id = attributes.index(target)  # get index of the target attribute
    samples = dataset[1:]  # extract samples
    samples_test = [dataset[1:][idx] for idx in test_ids]  # select samples for the evaluation
    model = trainModelOnSubset(samples, train_ids, target, attributes, max_depth)  # fit the model
    ground_truth = [s[target_id] for s in samples_test]  # extract true labels from the test split
    predicted = [predict(model, s, attributes) for s in samples_test]  # predict labels for the test split using trained model
    accuracy = sum(1.0 if y == gt else 0.0 for y, gt in zip(ground_truth, predicted)) / len(samples_test)  # calculate the accuracy
    return accuracy

def main():
    random.seed(42)  # for reproduce results
    dataset = parseARFF('data/weather.arff')  # load weather dataset
    target = 'play'  # target attribute's name
    attributes = dataset[0]  # extract attributes from the loaded dataset
    samples = dataset[1:]  # extract samples from the loaded dataset
    print('Dataset:\n')
    print('\n'.join(str(row) for row in dataset))
    print('-' * 120)
    tree = trainModel(samples, target, attributes)  # train decision tree on the whole dataset
    print('Tree on full weather dataset:\n')
    print(tree)
    print('-' * 120)
    print('Prediction of the target class "play" for samples (weather dataset, full tree):\n')
    for sample in samples:
        print(sample, '->', predict(tree, sample, attributes))  # predict target class for each row in dataset using decision tree trained on the whole dataset
    print('-' * 120)

    dataset = parseARFF('data/car.arff')
    target = 'class'
    test_size = 1.0 / 3  # split dataset into training and test parts in 2/1 proportion
    print('Evaluation on cars dataset (train/test split: 2/1):\n')
    for depth in [2 ** 31, 3, 5, 10, 20]:  # perform training of the decision tree using different limits for its depth
        print('Maximum depth of tree: {}\n'.format(depth))
        results = list()
        for i in range(10):
            indices = list(range(len(dataset) - 1))
            random.shuffle(indices)
            num_test = int(round(len(indices) * test_size, 0))
            test_ids = indices[:num_test]  # select random indices for the evaluation
            train_ids = indices[num_test:]  # select random indices for the training
            acc = evalModel(dataset, train_ids, test_ids, target, depth)  # accuracy on the test split
            results.append(acc)
        acc_mean = 1.0 * sum(results) / len(results)  # mean accuracy
        acc_std = (sum((x - acc_mean) ** 2 for x in results) / len(results)) ** 0.5  # std of the accuracy

        print('Accuracy (mean={:.2%}, std={:.2%}):\n{}'.format(
            acc_mean,
            acc_std,
            ', '.join('{:.2%}'.format(acc) for acc in results)
        ))

        # print('Accuracy (mean={:.2%}, std={:.2%}):\n{}'.format(
        #     acc_mean,
        #     acc_std,
        #     [float('{:.04f}'.format(acc)) for acc in results]
        # ))
        print('-' * 120)

# entry point
if __name__ == '__main__':
    main()
