import  pandas as pd
import numpy as np


# Создание искуственных данных
df = pd.DataFrame()
df['Внешность'] = [1, 1, 1, 0, 0, 0, 1] 
df['Алкоголь'] = [1, 1, 0, 0, 1, 1, 1]
df['Красноречие'] = [1, 0, 0, 0, 0, 1, 0]
df['Деньги'] = [1, 0, 1, 0, 1, 1, 1]
y = np.array([1, 0, 1, 0, 0, 1, 1])


# Функции отбора признаков
def entropy(a_list):
    a_list = np.array(a_list)
    enpropy = 0
    for i in np.unique(a_list):
        p = np.where(a_list == i)[0].size/a_list.size
        enpropy += -p*np.log2(p)
    return enpropy

def information_gain(root, left, right):
    n = len(left)/len(root)
    IG = entropy(root) - n*entropy(left) - (1-n)*entropy(right)
    return IG

def best_feature_to_split(X, y):
    IG = []
    for i in range(X.shape[1]):
        left = y[np.where(X[:, i] == 1)[0]]
        right = y[np.where(X[:, i] == 0)[0]]
        IG.append(information_gain(y, left, right))
    return np.array(IG).argmax(), entropy(y)


# Вершина дерева
class node:
    def __init__(self, feature_id=None, feature=None, left=None, right=None, entropy=None, answer=None):
        self.entropy = entropy
        self.feature = feature
        self.feature_id = feature_id
        self.left = left
        self.right = right
        self.answer = answer

    def __str__(self):
        return 'Node [' + self.feature + ', ent={:.2f}, ans='.format(self.entropy) + str(self.answer) + ']'

# Дерево
class Tree:
    def __init__(self, features):
        self.features = features
        self.root = None
        
    def fit(self, X, y):
        self.root = self.new_node(X, y) 
        
    def new_node(self, X, y):
        idx, entropy = best_feature_to_split(X, y)
        if entropy == 0:
            return node(feature_id=idx, feature='ЛИСТ', entropy=0, answer=y[0])
        else:
            indexs_l = np.where(X[:, idx] == 0)[0]
            indexs_r = np.where(X[:, idx] == 1)[0]
            curr = node(feature_id=idx, feature=self.features[idx], entropy=entropy)
            curr.left = self.new_node(X[indexs_l, :], y[indexs_l])
            curr.right = self.new_node(X[indexs_r, :], y[indexs_r])
            return curr;
    
    def print_tree(self):
        self.print_node(self.root, 0)
        
    def print_node(self, node, h):
        if node == None:
            return
        self.print_node(node.right, h + 1)
        print(' '*5*h, node)
        self.print_node(node.left, h + 1)
    
    def predict(self, X):
        return self.predict_node(self.root, X)
    
    def predict_node(self, node, X):
        if (node.entropy == 0):
            return node.answer;
        if (X[node.feature_id] == 1):
            return self.predict_node(node.right, X)
        else:
            return self.predict_node(node.left, X)


# Обучение дерева
my_tree = Tree(df.columns)
my_tree.fit(df.values, y)
my_tree.print_tree()
print('predict = {}'.format(my_tree.predict([1,1,1,1])))
print('predict = {}'.format(my_tree.predict([0,0,0,0])))
print('predict = {}'.format(my_tree.predict([1,0,0,1])))