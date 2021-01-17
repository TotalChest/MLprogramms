import  pandas as pd
import numpy as np


# Создание искуственных данных
df = pd.DataFrame()
df['Внешность']   = [5, 4, 4, 2, 1, 0, 3, 5, 3, 2, 3, 5] 
df['Алкоголь']    = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
df['Красноречие'] = [1, 2, 1, 2, 1, 0, 0, 1, 1, 2, 2, 0]
df['Деньги']      = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]
y        = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0])


# Функции отбора признаков
def entropy(a_list):
    a_list = np.array(a_list)
    enpropy = 0
    for i in np.unique(a_list):
        p = np.where(a_list == i)[0].size / a_list.size
        enpropy += -p*np.log2(p)
    return enpropy

def information_gain(root, left, right):
    n = len(left)/len(root)
    IG = entropy(root) - n*entropy(left) - (1-n)*entropy(right)
    return IG

def best_feature_to_split(X, y):
	IG = []
	for i in range(X.shape[1]):
		for treshold in np.unique(X[:, i]):
			left = y[np.where(X[:, i] >= treshold)[0]]
			right = y[np.where(X[:, i] < treshold)[0]]
			IG.append((information_gain(y, left, right), i, treshold))
	best_IG = 0;
	best_id = 0;
	best_treshold = 0;
	for tpl in IG:
		if tpl[0] >= best_IG:
			best_IG = tpl[0]
			best_id = tpl[1]
			best_treshold = tpl[2]
	return best_id, best_treshold, entropy(y)


# Вершина дерева
class node:
	def __init__(self, feature_id=None, feature=None, treshold=None,
    			 left=None, right=None, entropy=None, answer=None):
		self.entropy = entropy
		self.feature = feature
		self.feature_id = feature_id
		self.treshold = treshold
		self.left = left
		self.right = right
		self.answer = answer
		self.leaf = True if answer != None else False

	def __str__(self):
		if self.leaf:
			return f'Node [LEAF, ent={self.entropy}, ans={self.answer}]'
		else:
			return f'Node [if {self.feature} >= {self.treshold}, ent={self.entropy:.2f}]'
			
# Дерево
class Tree:
	def __init__(self, features, max_depth=5):
		self.features = features
		self.max_depth = max_depth
		self.root = None

	def fit(self, X, y):
		self.root = self.new_node(X, y, 0) 
        
	def new_node(self, X, y, h):
		idx, treshold, entropy = best_feature_to_split(X, y)
		if entropy == 0 or h == self.max_depth:
			return node(entropy=entropy, answer=np.bincount(y).argmax())
		else:
			indexs_r = np.where(X[:, idx] >= treshold)[0]
			indexs_l = np.where(X[:, idx] < treshold)[0]
			curr = node(feature_id=idx, feature=self.features[idx], treshold=treshold, entropy=entropy)
			curr.left = self.new_node(X[indexs_l, :], y[indexs_l], h + 1)
			curr.right = self.new_node(X[indexs_r, :], y[indexs_r], h + 1)
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
		if (node.leaf):
			return node.answer;
		if (X[node.feature_id] >= node.treshold):
			return self.predict_node(node.right, X)
		else:
			return self.predict_node(node.left, X)


# Обучение дерева
my_tree = Tree(df.columns, max_depth=3)
my_tree.fit(df.values, y)
my_tree.print_tree()
print('predict = {}'.format(my_tree.predict([5,1,2,1])))
print('predict = {}'.format(my_tree.predict([4,1,0,0])))
print('predict = {}'.format(my_tree.predict([2,1,1,0])))
print('predict = {}'.format(my_tree.predict([3,0,2,0])))
print('predict = {}'.format(my_tree.predict([1,1,2,1])))
