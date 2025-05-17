# SnugML/JS
JavaScript Machine Learning ES6 Module

## CDN Usage

https://cdn.jsdelivr.net/gh/snugml/js@main/ml.mjs

### Example


```
// Import library
const MLJS = 'https://cdn.jsdelivr.net/gh/snugml/js@main/ml.mjs'
import { DecisionTreeClassifier } from MLJS;

// Create an instance of the DecisionTreeClassifier
const model = new DecisionTree();

// Train the model using feature matrix (features) and label array (label)
model.fit(features, label);

// Make predictions using the trained model with the feature matrix
const yPredict = model.predict(features);

// Optionally, you can return or log the predictions
console.log(yPredict);
```

## MLearnJS is continuously being developed

JSDelivr changes may be slow. To use the latest version of the model, use:

```
const MLJS = 'https://snugml.github.io/js/ml.mjs'
import { DecisionTreeClassifier} from MLJS;
```

## Available Exported Classes and Methods

| **#** | **Class/Method**          | **Location (File)**        | **Description**                                                  |
|-------|---------------------------|----------------------------|------------------------------------------------------------------|
| 1     | `LinearRegression`         | `/models/linear-model.mjs`        | Class for performing linear regression.                          |
| 2     | `PolynomialRegression`     | `/models/linear-model.mjs`        | Class for performing polynomial regression.                      |
| 3     | `GaussianNB`               | `/models/naive-bayes.mjs`         | Class implementing the Naive Bayes classifier for Gaussian data. |
| 4     | `DecisionTreeClassifier`   | `/models/tree.mjs`                | Class for building and using decision trees.                     |
| 5     | `MLPClassifier`   | `/models/neural-network.mjs`                | Class implementing the MLP classifier.                     |
| 6     | `KMeans`          | `/models/cluster.mjs`       | Class for implementing the KMeans Cluster Classifier.
| 7     | `KNearestNeighbors`          | `/models/neighbors.mjs`       | Class for implementing the KNearestNeighbors Classifier.
| 8     | `LabelEncoder`             | `/utils/preprocessing.mjs`       | Class for encoding labels into numeric form.                     |
| 9     | `trainTestSplit`           | `/utils/model-selection.mjs`     | Function for splitting datasets into training and testing sets.  |
| 10     | `joinArrays`               | `/utils/model-selection.mjs`     | Function for joining two or more arrays.                         |
| 11     | `zip`                      | `/utils/model-selection.mjs`     | Function for zipping two or more arrays element-wise.            |
| 12    | `accuracyScore`            | `/utils/metrics.mjs`             | Function to calculate the accuracy score of a model.             |
| 13    | `CSV`                     | `/utils/data-analysis.mjs`             | Class for read CSV and convert to arrays.             |

## Examples

[Linear Regression](test/01-linear-regression.html)

[Polynomial Regression](test/02-polynomial-regression.html)

[Gaussian Naive Bayes](test/03-gaussian-nb.html)

[Decision Tree](test/04-decision-tree.html)

[MLP Classifier Logic Gates ](test/05-mlp-classifier.html)

[MLP Classifier](test/06-mlp-classifier.html)

[KMeans Cluster Classifier](test/07-kmeans-classifier.html)

[KNearest Neighbors Classifier](test/08-neighbors-classifier.html)