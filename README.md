```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hyperframe import HyperFrame
from sklearn.model_selection import train_test_split
from demo.helpers import metrics, X, y
```


```python
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.33, random_state=42)
```


```python
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



# Initialisation


```python
dimension_labels = ["train_test", "species", "metric"]

index_labels = {"train_test": ["train", "test"],
                "species": ["setosa", "versicolor", "virginica"],
                "metric": ["precision", "recall", "f1"]}

scores = HyperFrame(dimension_labels, index_labels)
```

# Setting data


```python
yhat = clf.predict(X_train)
#iset alternative 1
scores.iset(metrics(y_train, yhat), "train", "", "")
```




    <hyperframe.HyperFrame at 0x7f400b457e10>




```python
yhat = clf.predict(X_test)
#iset alternative 2
scores.iset(metrics(y_test, yhat), train_test="test")
```




    <hyperframe.HyperFrame at 0x7f400b457e10>



# Getting data


```python
#iget alternative 1
scores.iget("train", "", "", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>0.906250</td>
      <td>0.935484</td>
      <td>0.920635</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.725000</td>
      <td>0.828571</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.857143</td>
      <td>0.705882</td>
      <td>0.774194</td>
    </tr>
  </tbody>
</table>
</div>




```python
#iget alternative 2
scores.iget(species="versicolor", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>0.725000</td>
      <td>0.828571</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.578947</td>
      <td>0.733333</td>
      <td>0.647059</td>
    </tr>
  </tbody>
</table>
</div>




```python
#iget alternative 3
scores.iget0("species", "train_test", return_type="pandas")
```

    {'metric': 'precision'}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setosa</th>
      <th>versicolor</th>
      <th>virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>0.906250</td>
      <td>0.725000</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.944444</td>
      <td>0.578947</td>
      <td>0.769231</td>
    </tr>
  </tbody>
</table>
</div>



#### Initialising a second HyperFrame


```python
scores_lr = HyperFrame(dimension_labels, index_labels)
clf = LogisticRegression(penalty="none", max_iter=1000)
clf.fit(X_train, y_train)

yhat = clf.predict(X_train)
scores_lr.iset(metrics(y_train, yhat), "train", "", "")

yhat = clf.predict(X_test)
scores_lr.iset(metrics(y_test, yhat), "test", "", "")
```




    <hyperframe.HyperFrame at 0x7f400b3e6128>



# Merging


```python
print("scores shape: {}".format(scores.shape))
print("scores_lr shape: {}".format(scores_lr.shape))
```

    scores shape: (2, 3, 3)
    scores_lr shape: (2, 3, 3)



```python
scores_models = scores.merge(scores_lr, "model", ["knn", "logistic regression"])
```


```python
scores_models.iget("test", "", "f1", "", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>knn</th>
      <th>logistic regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>0.918919</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.647059</td>
      <td>0.588235</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.689655</td>
      <td>0.645161</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores_models.iget("", "", "f1", "logistic regression", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setosa</th>
      <th>versicolor</th>
      <th>virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>0.952381</td>
      <td>0.732394</td>
      <td>0.757576</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.914286</td>
      <td>0.588235</td>
      <td>0.645161</td>
    </tr>
  </tbody>
</table>
</div>



#### Initialising a third HyperFrame


```python
scores_rf = HyperFrame(dimension_labels, index_labels)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

yhat = clf.predict(X_train)
scores_rf.iset(metrics(y_train, yhat), "train", "", "")

yhat = clf.predict(X_test)
scores_rf.iset(metrics(y_test, yhat), "test", "", "")
```




    <hyperframe.HyperFrame at 0x7f400b3e6940>




```python
scores_rf.iget("test", "", "", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>1.000000</td>
      <td>0.842105</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.590909</td>
      <td>0.866667</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.833333</td>
      <td>0.625000</td>
      <td>0.714286</td>
    </tr>
  </tbody>
</table>
</div>



# Expanding A DataFrame


```python
print("scores_models shape: {}".format(scores_models.shape))
print("scores_rf shape: {}".format(scores_rf.shape))
```

    scores_models shape: (2, 3, 3, 2)
    scores_rf shape: (2, 3, 3)



```python
scores_models = scores_models.expand(scores_rf, "model", "random forest")
```


```python
scores_models.iget("test", "", "f1", "", return_type="pandas")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>knn</th>
      <th>logistic regression</th>
      <th>random forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>0.918919</td>
      <td>0.914286</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.647059</td>
      <td>0.588235</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.689655</td>
      <td>0.645161</td>
      <td>0.714286</td>
    </tr>
  </tbody>
</table>
</div>



# Writing to file


```python
scores_models.write_file("./demo/scores_models")
```

# Reading from file


```python
scores_models = scores_models.read_file("./demo/scores_models")
```
