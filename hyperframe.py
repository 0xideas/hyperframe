import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import json
import shutil
import subprocess
import warnings


def nest(l, depth=1, reps=1):
    """create a nested list of depth 'depth' and with 'reps' repititions"""
    if depth == 0:
        return(None)
    elif depth == 1:
        return(l)
    else:
        return([nest(l, depth-1, reps)] * reps)


def constant_array(constant, *args):
    """create an array filled with the 'constant' of shape *args"""
    if len(args) == 0:
        return(constant)
    else:
        return(np.array([constant_array(constant, *args[1:])]*args[0]))


def zeros(*args):
    """create a constant array of shape *args"""
    return(constant_array(0.0, *args))


def ones(*args):
    """create a constant array of shape *args"""
    return(constant_array(1.0, *args))


def unnest(l):
    """unnest a nested list l"""
    return([x for y in l for x in mlist(y)])


def mlist(x):
    """make sure x is a list"""
    if isinstance(x, list):
        return(x)
    else:
        return([x])


def idict(list_):
    """create an ordered dict of i -> 'list_'[i]"""
    return(OrderedDict(zip(range(len(list_)), list_)))


def ridict(list_):
    """create an ordered dict of 'list_'[i] -> i"""
    return(OrderedDict({v:k for k, v in idict(list_).items()}))


def ilist(dict_):
    return(list(dict_.values()))


def rilist(dict_):
    return(list(dict_.keys()))


def sort_dict(dict_):
    return([(k, v) for k, v in sorted(list(dict_.items()), key= lambda x: x[0])])


class HyperFrame:
    """
    A numpy array with dimension labels and named indices of each dimension for
    storage and access to high-dimensional data.

    Attributes:
        dim_labels (OrderedDict[int, string]): index -> dimension label
        rdim_labels (OrderedDict[string, int]): dimension label -> index
        val_labels (OrderedDict[string, OrderedDict[int, string]]): dimension label -> index -> index label
        rval_labels (OrderedDict[string, OrderedDict[string, int]]): dimension label -> index label -> index
        data (np.array): data
    """
    def __init__(self, dimension_labels, index_labels, data=None):
        """
        The constructor of the HyperFrame class.

        Parameters:
            dimension_labels (list[string]): dimension labels
            index_labels (dict[string, list[string]]): dimension_label -> index labels
            data (np.array): data
        """

        assert isinstance(dimension_labels, list)

        for k, v in index_labels.items():
            assert( isinstance(v, list))

        index_labels = OrderedDict(index_labels)

        self.dim_labels = idict(dimension_labels)
        self.rdim_labels = ridict(dimension_labels)

        self.val_labels = OrderedDict({k: idict(v) for k, v in index_labels.items()})        
        self.rval_labels = OrderedDict({k: ridict(v) for k, v in index_labels.items()})
        

        if data is None:
            data = zeros(*[len(self.val_labels[dim_label]) for _, dim_label in self.dim_labels.items()])
        
        HyperFrame.validate_data(data, self.dim_labels, self.val_labels)

        self.data = data


    def copy(self):
        """
        copy this HyperFrame

        Returns:
            HyperFrame: a new HyperFrame with the same data
        """
        index_labels = OrderedDict({k: rilist(v) for k, v in self.rval_labels.items()})
        return(HyperFrame( ilist(self.dim_labels), index_labels , self.data.copy()))

    def iget(self, *args, **kwargs):
        return(self._iget(False, *args, **kwargs))

    def icopy(self, *args, **kwargs):
        return(self._iget(True, *args, **kwargs))
    
    def _iget(self, copy=False, *args, **kwargs):
        """
        subset the dataframe
        """
        return_type = kwargs.get("return_type", "hyperframe")
        _ = kwargs.pop("return_type", None)

        assert len(args) == 0 or (len(args) == len(self.dim_labels) and len(kwargs)==0)

        kwargs = OrderedDict([(k, mlist(v)) for k, v in kwargs.items()] +
                             [(self.dim_labels[i], mlist(v))  if v in ilist(self.val_labels[self.dim_labels[i]])
                                                              else (self.dim_labels[i], ilist(self.val_labels[self.dim_labels[i]]))
                              for i, v in enumerate(args)])


        self.validate_kwargs(kwargs)
                        
        ndim_labels = [(i, v)
                       for i, v in self.dim_labels.items()
                       if v not in kwargs.keys() or len(kwargs[v]) > 1]

        ndim_labels = [x[1] for x in sorted(ndim_labels, key=lambda y: y[0])]
        
        nval_labels = {k: kwargs.get(k, v) for k, v in self.val_labels.items() if len(kwargs.get(k, v)) > 1}
        nval_labels = {k: (mlist(v) if not isinstance(v, dict) else list(v.values())) for k, v in nval_labels.items()}

        if copy:
            ndata = self.data.copy()
        else:
            ndata = self.data


        indices = self.construct_indices(kwargs, ndata.shape)
        ndata = ndata[np.ix_(*indices)]
        ndata = ndata.reshape(*[x for x in ndata.shape if x > 1]) 

        subset = HyperFrame(ndim_labels, nval_labels, ndata)


        if return_type == "pandas" and len(subset.data.shape) > 2:
            warnings.warn("Returning HyperFrame as subset dimenionality is above 2")


        if len(subset.data.shape) <= 2 and return_type == "pandas":
            return(self._get_pandas_object(subset))
        elif return_type == "hyperframe":
            return(subset)
        elif return_type == "numpy":
            return(subset.data)
        else:
            warnings.warn("return_type must be in ['hyperframe', 'pandas', 'numpy']")

   


    @staticmethod
    def _get_pandas_object(subset):
        indices = [[v2 for k2, v2 in sort_dict(subset.val_labels[v])] 
                   for k, v in sort_dict(subset.dim_labels)]

        if len(indices) == 1:
            return(pd.Series(subset.data, index=indices[0]))
        else:
            return(pd.DataFrame(subset.data, index=indices[0], columns = indices[1]))

    def iget0(self, *args, return_type=None):
        assert len(args) > 0 and len(args) < 3

        kwargs = {v: self.val_labels[v][0] for i, v in self.dim_labels.items() if v not in args}
        print(kwargs)
        subset = self.iget(**kwargs)
        
        if return_type is None:
            return( subset)
        elif return_type == "numpy":
            return(subset.data)
        elif return_type == "pandas": 
            return(self._get_pandas_object(subset))           



    
    def iset(self, new_data, copy=False,  **kwargs):
        kwargs = OrderedDict([(k, mlist(v)) for k, v in kwargs.items()])

        self.validate_kwargs(kwargs)

        test1 = np.issubdtype(type(new_data), np.number)
        test2 = np.issubdtype(type(self.iget(**kwargs).data), np.number)

        test3 = type(new_data) == type(np.array([0]))

        assert test3 or (test1 and test2)
        
        if copy:
            hframe = self.copy()
        else:
            hframe = self

        indices = self.construct_indices(kwargs, hframe.data.shape)
        hframe.data[np.ix_(*indices)] = new_data.reshape([len(x) for x in indices])
            
        return(hframe)
    


    def construct_indices(self, kwargs, shape):
        numpy_indices = [list(range(x)) for x in shape]
        for dim_label, target_labels in kwargs.items():
            dim = self.rdim_labels[dim_label]
            target_indices = {dim:[self.rval_labels[dim_label][target] for target in target_labels]}

            for k, v in target_indices.items():
                v2 = v if len(v) > 1 else v[0]
                numpy_indices[k] = v

        return(numpy_indices)

    
    def validate_kwargs(self, kwargs):
        for key, value in kwargs.items():
            try:
                assert key in self.dim_labels.values()
                
                assert(len(value) > 0)
                for v in value:
                    assert(v in self.val_labels[key].values())
            except:
                print("{}\n{}\n\n{}\n{}\n".format(key, self.dim_labels.values(), value, self.val_labels[key].values()))
                raise Exception("illegal argument provided")
        
    
    @staticmethod
    def validate_data(data, dim_labels, val_labels):

        assert isinstance(data, type(np.array([0]))) or np.issubdtype(type(data), np.number)
        
        assert len(dim_labels) == len(val_labels)
        
        HyperFrame.validate_dict(dim_labels, len(data.shape))
        
        for dim, dim_label in dim_labels.items():
            assert dim_label in val_labels.keys()
            
            HyperFrame.validate_dict(val_labels[dim_label], data.shape[dim])
    
    @staticmethod
    def validate_dict(dict_, dims):
        assert len(dict_) == dims
        assert HyperFrame.dense_keys(dict_)
        
    @staticmethod
    def dense_keys(dict_):
        dict_keys = dict_.keys()
        return np.all([x in dict_keys for x in range(len(dict_))])

    @staticmethod
    def strip_path(path):
        filename = path.split("/")[-1]
        if "." in filename and filename.split(".")[:-1] in ["csv", "txt", "hyperframe"]:
            path = ".".join(path.split(".")[:-1]) 

            warnings.warn("path changed to: {}".format(path))
        return(path)

    def write_file(self, path):

        path = self.strip_path(path)

        dir_ = path+"/"
        os.mkdir(dir_)

        with open(os.path.join(dir_, "labels.json"), "w") as f:
            f.write(json.dumps({"dim_labels": self.dim_labels,
                                "rdim_labels": self.rdim_labels,
                                "val_labels": self.val_labels,
                                "rval_labels": self.rval_labels}))

        np.save(os.path.join(dir_, "data"), self.data)

        
        subprocess.run(["zip", "-r", path + ".zip", "-j", dir_])
        subprocess.run(["rm", "-r", dir_])
        subprocess.run(["mv", path + ".zip", path + ".hyperframe" ])

    @staticmethod
    def read_file(path):

        path = HyperFrame.strip_path(path)

        dir_ = path+"/"

        subprocess.run(["mv", path + ".hyperframe", path + ".zip" ])
        subprocess.run(["unzip", path + ".zip", "-d", dir_])
        subprocess.run(["mv", path + ".zip", path + ".hyperframe" ])

        data = np.load(os.path.join(dir_, "data.npy"))

        with open(os.path.join(dir_, "labels.json"), "r") as f:
            labels = json.loads(f.read())

        subprocess.run(["rm", "-r", dir_ ])

        return(HyperFrame(ilist(labels["dim_labels"]), {k: ilist(v) for k, v in labels["val_labels"].items()}, data))