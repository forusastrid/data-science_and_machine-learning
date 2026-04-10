#112

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

y_named = np.array(['blue', 'red'])[y]

X_train, X_test, y_train, y_test , y_train_named, y_test_named = \
    train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

#113

print("X_test.shape:" , X_test.shape)
print("결정 함수 결과 형태", gbrt.decision_function(X_test).shape)

#114

print("결정 함수:\n",  gbrt.decision_function(X_test)[:6])

#115

print("임계치와 결정 함수 결과 비교:\n",
      gbrt.decision_function(X_test) > 0)
print("예측 : \n", gbrt.predict(X_test))

#116

greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)

pred = gbrt.classes_[greater_zero]
print("pred는 예측 결과와 같다:",
      np.all(pred == gbrt.predict(X_test)))

#117

decision_function = gbrt.decision_function(X_test)
print("결정함수 최소값 : {:.2f}, 최대값 : {:.2f}".format(
    np.min(decision_function), np.max(decision_function)))
