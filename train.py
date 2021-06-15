from sklearn import svm
from sklearn import datasets
from bento_service import IrisClassifier
# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


# Create a iris classifier service with the newly trained model
iris_classifier_service = IrisClassifier()
iris_classifier_service.pack("model", clf)

import pandas as pd
test_input_df = pd.DataFrame(X).sample(n=5)
# test_input_df.to_csv("./test_input.csv", index=False)
print(iris_classifier_service.predict(test_input_df))

saved_path = iris_classifier_service.save()
print(saved_path)