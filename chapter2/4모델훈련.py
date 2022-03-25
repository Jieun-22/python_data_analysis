new = ([25, 150]- mean)/ std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) /std

kn.score(test_scaled, test_target)
print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = '^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes,1],marker = 'D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_input)
train_scaled = scaler.transform(train_input)

test_scaled = scaler.transform(test_input)
kn.fit(train_scaled, train_target)
kn.score(test_scaled, test_target)
data = [[25,150]]
data_scaled = scaler.transform(data)
kn.predict(data)
