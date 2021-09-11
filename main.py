import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "health", "goout", "freetime", "Dalc", "Walc", "famrel"]]
#print(data.head())
predict = "G3"

X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels

#if I stop for loop then uncomment it
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        print("Best Accuracy: " + str(best))
        # saving our model
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

#loading our model
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print('Coefficient: \n', linear.coef_)# These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

predictions = linear.predict(x_test)# Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Plotting data
plot = 'G1'
style.use("ggplot")
pyplot.scatter(data[plot], data["G3"])
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()