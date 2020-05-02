
# Assignment
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# read data
p1_train_input = pd.read_csv("p1_train_input.txt", names=['x1','x2'], sep='\s+')
p1_train_label = pd.read_csv("p1_train_target.txt", names=['y'], sep='\s+')
p1_test_input = pd.read_csv("p1_test_input.txt", names=['x1','x2'], sep='\s+')
p1_test_label = pd.read_csv("p1_test_target.txt", names=['y'], sep='\s+')
p1_train = pd.concat([p1_train_input, p1_train_label], axis=1)
p1_test = pd.concat([p1_test_input, p1_test_label], axis=1)


p2_train_input = pd.read_csv("p2_train_input.txt", names=['x1','x2'], sep='\s+')
p2_train_label = pd.read_csv("p2_train_target.txt", names=['y'], sep='\s+')
p2_test_input = pd.read_csv("p2_test_input.txt", names=['x1','x2'], sep='\s+')
p2_test_label = pd.read_csv("p2_test_target.txt", names=['y'], sep='\s+')
p2_train = pd.concat([p2_train_input, p2_train_label], axis=1)
p2_test = pd.concat([p2_test_input, p2_test_label], axis=1)

#plot the p1
sns.scatterplot(x='x1', y='x2', hue='y', s=100, data=p1_train)
plt.show()

p1_train_input[:,1]


def plot_gaussianmixture(n):
    model = GaussianMixture(n_components=2, init_params='random', random_state=0, tol=1e-9, max_iter=n)
    model.fit(p1_train_input)
    pi = model.predict_proba(p1_train_input)
    line = model.get_params()
    plt.scatter(p1_train_input['x1'], p1_train_input['x2'], s=50, linewidth=1, edgecolors="b", cmap=plt.cm.binary, c=pi[:, 0])
    plt.title("iteration: {}".format(n))

plot_gaussianmixture(500)

