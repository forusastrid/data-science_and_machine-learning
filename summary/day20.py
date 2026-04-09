#107

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train

X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifer(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도 : {:3f}". format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도 : {:3f}". format(mlp.score(X_test_scaled, y_test)))

#108

mlp = MLPClassifer(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도 : {:3f}". format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도 : {:3f}". format(mlp.score(X_test_scaled, y_test)))

#109

mlp = MLPClassifer(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도 : {:3f}". format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도 : {:3f}". format(mlp.score(X_test_scaled, y_test)))

#110

mlp.coefs_[0].std(axis=1), mlp.coefs_[0].var(axis=0)

#111

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("은닉 유닛")
plt.ylabel("입력 특성")
plt.colorbar()
plt.show()
