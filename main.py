from sklearn.model_selection import train_test_split
import numpy


class BayesClassifier:

    def __init__(self, intervals_num=10, useLaPlace=False):
        self.intervals_num = intervals_num
        self.useLaPlace = useLaPlace

    def fit(self, Y, X):
        self.variables_num = len(X[0])
        self.categories = numpy.unique(Y).tolist()
        self.categories_num = len(self.categories)

        self.set_discrete_intervals(X)
        X = self.discretize_data(X)

        self.PY = []
        self.PXY = []
        for cat in self.categories:
            total_counter = 0
            for y in Y:
                if y == cat:
                    total_counter += 1

            self.PY.append(total_counter / len(X))

            self.PXY.append([])
            for i in range(self.variables_num):
                self.PXY[-1].append([])
                for j in range(self.intervals_num):
                    counter = 0
                    for k in range(len(X)):
                        y = Y[k]
                        x = X[k]
                        if y == cat and x[i] == j:
                            counter += 1

                    if self.useLaPlace:
                        p = ((counter + 1) / (total_counter + self.intervals_num))
                        self.PXY[-1][-1].append(p)
                    else:
                        self.PXY[-1][-1].append(counter / total_counter)

    def predict(self, x):
        P = self.predict_proba(x)
        max_i = 0
        for i in range(1, self.categories_num):
            if P[max_i] < P[i]:
                max_i = i

        return self.categories[max_i]

    def predict_proba(self, x):
        x = self.discretize_data([x])[0]

        P = []
        for cat_id in range(self.categories_num):
            p = self.PY[cat_id]
            for i in range(self.variables_num):
                p *= self.PXY[cat_id][i][x[i]]
            P.append(p)

        s = sum(P)
        if s > 0:
            for i in range(len(P)):
                P[i] *= 1 / s

        return P

    def discretize_data(self, X):
        X2 = []
        for x in X:
            x2 = []
            for i in range(self.variables_num):
                j = 0
                while j < self.intervals_num and x[i] > self.intervals[i][j][1]:
                    j += 1

                if j == self.intervals_num:
                    j -= 1

                x2.append(j)
            X2.append(x2)

        return X2

    def set_discrete_intervals(self, X):
        self.intervals = []
        for var_num in range(self.variables_num):

            all_values = [x[var_num] for x in X]
            min_value = min(all_values)
            max_value = max(all_values)

            value_span = max_value - min_value
            interval_length = value_span / self.intervals_num

            var_intervals = []
            for i in range(self.intervals_num):
                lower = min_value + i * interval_length
                upper = min_value + (i + 1) * interval_length
                var_intervals.append((lower, upper))

            self.intervals.append(var_intervals)


def test_classificator(classifier, X, Y):

    correct = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        result = classifier.predict(x)
        if y == result:
            correct += 1

    return correct


data = numpy.genfromtxt("dataset/wine.csv", delimiter=',')
data_Y, data_X = numpy.hsplit(data, [1])

train_Y, test_Y, train_X, test_X = train_test_split(
    data_Y, data_X, test_size=0.5)

nbc = BayesClassifier(intervals_num=10, useLaPlace=True)
nbc.fit(train_Y, train_X)

correct = test_classificator(nbc, test_X, test_Y)

print("====================")
print("tests number:", len(test_X))
print("correct:", correct, end='')
print(" (%.2f" % (100 * correct / len(test_X)), end='')
print("%)")
