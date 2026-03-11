-day4/21.py

print("테스트 세트 예측 : ", clf.predict(X_test))

-day4/22.py

print("테스트 세트 정확도 : {: .2f}" .format(clf.score(X_test, y_test))

-day4/23.py
      
fig, axes = plt.subplots(1,3,figsize=(10,3))

-day4/24.py
      
for n_neighbors, ax in zip([1,3,9], axes)
      clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X.y)
      mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
      mglearn.discrete_scatter(X[:, 0],X[:, 1], y, ax=ax)
      ax.set_title("{} 이웃".format(n_neighbors))
      ax.set_xlabel("특성 0")
      ax.set_ylabel("특성 1")
axes[0].legend(loc=3)
pit.show()

-day4/25.py
      
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
  cancer.data, cancer.target, stratify=cancer.target ,random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)
    
for n_neighbors in neighbors_settings:
      clf = KNeighborsClassifier(n_neighbors=n_neighbors)
      clf.fit(X_train, y_train)

      training_accuracy.append(clf.score(X_train, y_train)

      test_accuracy.append(clf.score(X_train, y_train)

plt.plot(neighbors_settings , training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings , test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
pit.show()                         
