from gdfs import Model
y_test = [110, 144, 169, 196, 225]
y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_test = [11, 12, 13, 14, 15]
x_to_predict = [16, 17, 18, 19, 20]

model = Model(iterations=10000, learning_rate=0.01)


model.train(X, y)
model.evaluate(x_test, y_test)
val = model.predict(x_to_predict)
print(val)




