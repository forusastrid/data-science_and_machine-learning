- day3/14.py

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url , sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print("데이터의 형태 :", data.shape)

- day3/15.py

X, y = mglearn.datasets.load_extended_boston()
print("X.shape : ", X.shape)

- day3/16.py

#K-최근접 이웃 분류

mglearn.plots.plot_knn_classification(n_neighbors=1) #별 기준으로 가까운거 1개씩 묶기

- day3/17.py

#K-최근접 이웃 분류

mglearn.plots.plot_knn_classification(n_neighbors=3) #별 기준으로 가까운거 3개씩 묶기

- day3/18.py

from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

- day3/19.py

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsclassifer(n_neighbors=3)

- day3/20.py

clf.fit=(X.train, y_train)
