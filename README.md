# HyperFrame

The aim of this project is to provide a high-dimensional analogue to the two-dimensional pandas DataFrame.

This allows its user to organise information where the interaction of several factors is of interest.

The HyperFrame allows for the easy setting and saving of data for storage, and the fast, interactive creation of two-dimensional pandas DataFrames of any combination of two factors for data exploration.


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




    <hyperframe.HyperFrame at 0x7ff4d4241320>




```python
yhat = clf.predict(X_test)
#iset alternative 2
scores.iset(metrics(y_test, yhat), train_test="test")
```




    <hyperframe.HyperFrame at 0x7ff4d4241320>



# Getting data


```python
#iget alternative 1
scores.iget("train", "", "", return_type="pandas").round(2)
```




<div>
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
      <td>0.89</td>
      <td>1.00</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.80</td>
      <td>0.71</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
#iget alternative 2
scores.iget(species="versicolor", return_type="pandas").round(2)
```




<div>
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
      <td>0.71</td>
      <td>0.71</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.70</td>
      <td>0.47</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>




```python
#iget alternative 3
scores.iget0("species", "train_test", return_type="pandas").round(2)
```

    {'metric': 'precision'}





<div>

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
      <td>0.89</td>
      <td>0.71</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.95</td>
      <td>0.70</td>
      <td>0.65</td>
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




    <hyperframe.HyperFrame at 0x7ff4d4231588>



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
scores_models.iget("test", "", "f1", "", return_type="pandas").round(2)
```




<div>

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
      <td>0.97</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.56</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.72</td>
      <td>0.72</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores_models.iget("", "", "f1", "logistic regression", return_type="pandas").round(2)
```




<div>

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
      <td>0.92</td>
      <td>0.74</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0.95</td>
      <td>0.58</td>
      <td>0.72</td>
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




    <hyperframe.HyperFrame at 0x7ff4d41cd978>




```python
scores_rf.iget("test", "", "", return_type="pandas").round(2)
```




<div>

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
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.75</td>
      <td>0.40</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.61</td>
      <td>0.88</td>
      <td>0.72</td>
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
scores_models.iget("test", "", "f1", "", return_type="pandas").round(2)
```




<div>

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
      <td>0.97</td>
      <td>0.95</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.56</td>
      <td>0.58</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.72</td>
    </tr>
  </tbody>
</table>
</div>



# Simple Mathematical Operations


```python
scores.max("train_test").iget("", "", return_type="pandas")
```




<div>

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
      <td>0.950000</td>
      <td>1.000000</td>
      <td>0.974359</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0.714286</td>
      <td>0.714286</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0.800000</td>
      <td>0.812500</td>
      <td>0.750000</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores.min("train_test", "metric").iget("", return_type="pandas")
```




    setosa        0.885714
    versicolor    0.466667
    virginica     0.650000
    dtype: float64




```python
scores.mean("train_test", "species", "metric")
```




    0.7810886435641339




```python
scores.sum()
```




    14.059595584154408



# Writing to file


```python
scores_models.write_file("./demo/scores_models")
```

# Reading from file


```python
scores_models = scores_models.read_file("./demo/scores_models")
```
