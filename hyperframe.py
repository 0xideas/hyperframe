import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import json
import shutil
import warnings
import subprocess


def nest(l, depth=1, reps=1):
    """create a nested list of depth 'depth' and with 'reps' repititions"""
    if depth == 0:
        return(None)
    elif depth == 1:
        return(l)
    else:
        return([nest(l, depth-1, reps)] * reps)

def unnest(l):
    """unnest a nested list l"""
    return([x for y in l for x in mlist(y)])



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
        dimension_labels (list[string]): dimension labels
        index_labels (OrderedDict[string, list[string]]): dimension label -> index labels
        dim_labels (OrderedDict[int, string]): index -> dimension label
        rdim_labels (OrderedDict[string, int]): dimension label -> index
        val_labels (OrderedDict[string, OrderedDict[int, string]]): dimension label -> index -> index label
        rval_labels (OrderedDict[string, OrderedDict[string, int]]): dimension label -> index label -> index
        data (np.array): data
        shape (set): shape of the data
    """
    def __init__(self, dimension_labels, index_labels, data=None):
        """
        The constructor of the HyperFrame class.

        Parameters:
            dimension_labels (list[string]): dimension labels
            index_labels (dict[string, list[int | string]]): dimension_label -> index labels
            data (np.array): data
        """

        index_labels = OrderedDict(index_labels)

        self.dimension_labels = dimension_labels
        self.dim_labels = idict(dimension_labels)
        self.rdim_labels = ridict(dimension_labels)

        self.index_labels = index_labels
        self.val_labels = OrderedDict({k: idict(v) for k, v in index_labels.items()})        
        self.rval_labels = OrderedDict({k: ridict(v) for k, v in index_labels.items()})
        

        if data is None:
            data = zeros(*[len(self.val_labels[dim_label]) for _, dim_label in self.dim_labels.items()])

        self.data = data

        self.shape = data.shape

        HyperFrame._validate_hyperframe(self)

    def len(self, *args):
        return([self.shape[self.rdim_labels[a]] for a in args])

    def sum(self, *args):
        return(self.apply_f1(np.sum, *args))

    def mean(self, *args):
        return(self.apply_f1(np.mean, *args))

    def min(self, *args):
        return(self.apply_f1(np.min, *args))

    def max(self, *args):
        return(self.apply_f1(np.max, *args))

    def apply_f1(self, f, *args):
        if len(args) == 0:
            args = self.dimension_labels
            
        assert np.all([a in self.dimension_labels for a in args])
        dims = sorted([self.rdim_labels[a] for a in args])

        ndata = self.data
        for i, dim in enumerate(dims):
            ndata = f(ndata, dim-i)

        if isinstance(ndata, type(np.array([0]))):
            new_dimension_labels = [d for d in self.dimension_labels if d not in args]
            new_index_labels = OrderedDict([(k, v) for k, v in self.index_labels.items() if k not in args])

            return(HyperFrame(new_dimension_labels, new_index_labels, ndata))
        else:
            assert np.issubdtype(type(ndata), np.number)

            return(ndata)

    def copy(self):
        """
        copy this HyperFrame

        Returns:
            HyperFrame: a new HyperFrame with the same data
        """
        return(HyperFrame( self.dimension_labels, OrderedDict(self.index_labels) , self.data.copy()))


    def iget(self, *args, **kwargs):
        """
        Get a subset of the dataframe by EITHER args or kwargs

        Parameters:
            *args (list[string]): values on each dimension by which the data should be subset, dimensions that should
                                  not be subset should have a value not in the dimension index
            **kwargs (dict[string, int | string | list[int] | list[string]]): dimension labels -> value or values 
                                                                              to subset by
            return_type (string) (in kwargs): in ["hyperframe", "pandas", "numpy"]

        Returns:
            HyperFrame | pd.DataFrame | pd.Series | np.array: subset of original data
        """

        return_type = kwargs.pop("return_type", "hyperframe")

        kwargs = self._build_kwargs(args, kwargs)
                        
        ndim_labels = [(i, v)
                       for i, v in self.dim_labels.items()
                       if v not in kwargs.keys() or len(kwargs[v]) > 1]

        ndim_labels = [x[1] for x in sorted(ndim_labels, key=lambda y: y[0])]
        
        nval_labels = {k: kwargs.get(k, v) for k, v in self.val_labels.items() if len(kwargs.get(k, v)) > 1}
        nval_labels = {k: (mlist(v) if not isinstance(v, dict) else list(v.values())) for k, v in nval_labels.items()}


        indices = self._construct_indices(kwargs, self.data.shape)
        ndata = self.data[np.ix_(*indices)]
        ndata = ndata.reshape(*[x for x in ndata.shape if x > 1]) 

        subset = HyperFrame(ndim_labels, nval_labels, ndata)

        return(HyperFrame._cast_return_value(subset, return_type))


    @staticmethod
    def _cast_return_value(hyperframe, return_type):
        """'cast' a hyperframe to a pandas or numpy object, or return unaltered"""
        if return_type == "pandas":
            return(HyperFrame._get_pandas_object(hyperframe))
        elif return_type == "numpy":
            return(hyperframe.data)
        elif return_type == "hyperframe":
            return(hyperframe)
        else:
            warnings.warn("return_type must be in ['hyperframe', 'pandas', 'numpy']")


    @staticmethod
    def _get_pandas_object(hyperframe):
        """
        Turn a HyperFrame of dimensionality <= into a pandas object

        Paramters:
            hyperframe (HyperFrame)

        Returns:
            pd.DataFrame | pd.Series
        """
        indices = [[v2 for k2, v2 in sort_dict(hyperframe.val_labels[v])] 
                   for k, v in sort_dict(hyperframe.dim_labels)]

        assert len(indices) > 0, "pandas objects must have at least one dimension"
        assert len(indices) <= 2, "pandas objects cannot have {} dimensions".format(len(indices))

        if len(indices) == 1:
            return(pd.Series(hyperframe.data, index=indices[0]))
        else:
            return(pd.DataFrame(hyperframe.data, index=indices[0], columns = indices[1]))


    def iget0(self, *args, return_type=None):
        """
        Return data for the dimensions in *args by subsetting the other dimensions to the first index label

        Parameters:
            *args (list[string]): the dimensions to be preserved in full
            return_type (string) (in kwargs): in ["hyperframe", "pandas", "numpy"]

        Returns:
            HyperFrame | pd.DataFrame | pd.Series | np.array: subset of original data
        """
        assert len(args) > 0 and len(args) <= 2
        assert np.all([a in self.dimension_labels for a  in args])

        kwargs = {v: self.val_labels[v][0] for i, v in self.dim_labels.items() if v not in args}
        print(kwargs)
        subset = self.iget(**kwargs)

        return(HyperFrame._cast_return_value(subset, return_type))


    def iset(self, new_data, *args, **kwargs):
        """
        Replace a subset of the data with 'new_data'

        Parameters:
            new_data (np.array): new data
            *args (list[string]): values on each dimension by which the data should be subset, dimensions that should
                                  not be subset should have a value not in the dimension index
            **kwargs (dict[string, int | string | list[int] | list[string]]): dimension labels -> value or values
                                                                              to subset by

        Returns:
            HyperFrame: HyperFrame with changed data
        """
        kwargs = self._build_kwargs(args, kwargs)

        assert np.issubdtype(type(new_data), np.number) or ( isinstance(new_data, type(np.array([0]))) and  new_data.shape)
        
        indices = self._construct_indices(kwargs, self.data.shape)
        self.data[np.ix_(*indices)] = new_data.reshape([len(x) for x in indices])
            
        return(self)

    def _validate_other(self, other, expand):

        self_dimension_labels_subset = [label for label in self.dimension_labels if label in other.dimension_labels]

        assert (len(self_dimension_labels_subset) + int(expand)) == len(self.dimension_labels)

        for i, label in enumerate(self_dimension_labels_subset):
            assert label == other.dim_labels[i], "the dimension labels of self and other must be identical" 

        dims_identical = [np.all(np.array(self.rval_labels[label].keys()) == 
                                          np.array(other.rval_labels[label].keys()))
                         for label in self_dimension_labels_subset]

        assert np.all(dims_identical)


    def expand(self, other, on_dimension, new_index_label):
        """
        Expand this HyperFrame along one dimension with another HyperFrame

        Parameters:
            other (HyperFrame): another HyperFrame with identical 'dimension_labels' and identical 'index_labels'
                                except for on one dimension: On that dimension, the 'index_labels' of other do
                                not overlap with those of self
            on_dimension (string): dimension label on which other is "tacked on"
            new_index_label (string): index label for other within the new HyperFrame

        Returns:
            HyperFrame: A new HyperFrame with additional index labels and data on one dimension

        """

        HyperFrame._validate_hyperframe(other)
        self._validate_other(other, True)

        dim_to_expand_on = [(i, label) for i, label in self.dim_labels.items() if label not in other.dimension_labels]

        assert len(dim_to_expand_on) == 1

        dim_different, dim_label_different = dim_to_expand_on[0]

        assert dim_label_different == on_dimension

        assert new_index_label not in self.index_labels[dim_label_different], \
                "the dimension with different val_labels cannot have overlapping val_labels"

        new_index_labels = OrderedDict(self.index_labels)
        new_index_labels[dim_label_different] += [new_index_label]

        new_data = np.concatenate([self.data, np.expand_dims(other.data, dim_different)], axis=dim_different)

        return(HyperFrame(self.dimension_labels, new_index_labels, new_data))


    def merge(self, other, new_dimension, new_dimension_index_labels):
        """
        Merge self and other to create a new HyperFrame with one additional dimension

        Paramters:
            other (HyperFrame | list[HyperFrame]): other HyperFrame(s) with the same dimension_labels and
                                                   index_labels as self
            new_dimension (string): name of the new dimension
            new_dimension_index_labels (list[string]): labels on the new dimension

        Returns:
            HyperFrame
        """

        other = mlist(other)

        assert new_dimension not in self.dimension_labels
        assert len(new_dimension_index_labels) == (len(mlist(other)) + 1), \
               "there must be as many new_dimension_index_labels as there are objects to merge"

        assert len(new_dimension_index_labels) == len(set(new_dimension_index_labels)), \
               "new_dimension_index_labels must be unique"

        for other_ in other:
            self._validate_other(other_, False)

        new_dimension_labels = self.dimension_labels + [new_dimension]

        new_index_labels = OrderedDict(self.index_labels)
        new_index_labels[new_dimension] = new_dimension_index_labels

        reshaped_data = [np.expand_dims(hf.data, -1) for hf in [self] + other]
        new_data = np.concatenate(reshaped_data, axis= len(self.data.shape))

        return(HyperFrame(new_dimension_labels, new_index_labels, new_data))


    def _construct_indices(self, kwargs, shape):

        numpy_indices = [list(range(x)) for x in shape]

        for dim_label, target_labels in kwargs.items():
            dim = self.rdim_labels[dim_label]
            target_indices = {dim:[self.rval_labels[dim_label][target] for target in target_labels]}

            for k, v in target_indices.items():
                v2 = v if len(v) > 1 else v[0]
                numpy_indices[k] = v

        return(numpy_indices)


    def _build_kwargs(self, args, kwargs):

        assert (len(args) == 0 and len(kwargs) > 0) or (len(args) == len(self.dim_labels) and len(kwargs)==0)

        for i, arg in enumerate(args):
            if arg not in self.index_labels[self.dim_labels[i]] and arg != "":
                raise ValueError("{} is not a valid index_label for {}".format(arg, self.dim_labels[i]))

        assert len(kwargs) == 0 or np.all([k in self.dimension_labels and
                                           np.all([v_ in self.index_labels[k] for v_ in mlist(v)])
                                           for k, v in kwargs.items()])

        kwargs = [(k, mlist(v)) for k, v in kwargs.items()]

        args_to_kwargs = [(self.dim_labels[i], mlist(v) if v in self.index_labels[self.dim_labels[i]]
                                                        else self.index_labels[self.dim_labels[i]])
                         for i, v in enumerate(args)]

        kwargs = OrderedDict(kwargs + args_to_kwargs)
        self._validate_kwargs(kwargs)

        return(kwargs)   
    

    def _validate_kwargs(self, kwargs):

        for key, value in kwargs.items():
            try:
                assert key in self.dim_labels.values()
                
                assert(len(value) > 0)
                for v in value:
                    assert(v in self.val_labels[key].values())
            except:
                print("{}\n{}\n\n{}\n{}\n".format(key,
                                                  self.dim_labels.values(),
                                                  value,
                                                  self.val_labels[key].values()))

                raise Exception("illegal argument provided")
        
    
    @staticmethod
    def _validate_hyperframe(hyperframe):

        assert isinstance(hyperframe.dimension_labels, list)
        assert len(hyperframe.dimension_labels) > 0

        assert isinstance(hyperframe.index_labels, OrderedDict)
        assert len(hyperframe.index_labels) > 0
        assert np.all([isinstance(v, list) for k, v in hyperframe.index_labels.items()])

        assert len(hyperframe.dimension_labels) == len(hyperframe.index_labels)

        HyperFrame._validate_dict(hyperframe.dim_labels, len(hyperframe.data.shape))
        
        for dim, dim_label in hyperframe.dim_labels.items():
            assert dim_label in hyperframe.val_labels.keys()
            HyperFrame._validate_dict(hyperframe.val_labels[dim_label], hyperframe.data.shape[dim])

        assert isinstance(hyperframe.data, type(np.array([0])))

      
    @staticmethod
    def _validate_dict(dict_, dims):

        assert len(dict_) == dims
        assert HyperFrame._dense_keys(dict_)
        
    @staticmethod
    def _dense_keys(dict_):
        return np.all([x in dict_.keys() for x in range(len(dict_))])

    @staticmethod
    def _strip_path(path):

        filename = path.split("/")[-1]
        if "." in filename and filename.split(".")[:-1] in ["csv", "txt", "hyperframe"]:
            path = ".".join(path.split(".")[:-1]) 
            warnings.warn("path changed to: {}".format(path))

        return(path)


    def write_file(self, path):
        """write file to path"""

        path = self._strip_path(path)

        dir_ = os.path.join(path + str(np.random.uniform())[2:], "")

        assert not os.path.exists(dir_)
        assert not os.path.exists(path + ".zip")

        os.mkdir(dir_)

        try:
            with open(os.path.join(dir_, "labels.json"), "w") as f:
                f.write(json.dumps({"dim_labels": self.dim_labels,
                                    "rdim_labels": self.rdim_labels,
                                    "val_labels": self.val_labels,
                                    "rval_labels": self.rval_labels}))

            np.save(os.path.join(dir_, "data"), self.data)

            shutil.make_archive(path, 'zip', dir_)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir_)

    @staticmethod
    def read_file(path):
        """read file from path"""

        path = HyperFrame._strip_path(path)

        dir_ = os.path.join(path + str(np.random.uniform())[2:], "")

        shutil.unpack_archive(path+".zip", dir_)

        try:
            data = np.load(os.path.join(dir_, "data.npy"))

            with open(os.path.join(dir_, "labels.json"), "r") as f:
                labels = json.loads(f.read())
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir_)

        return(HyperFrame(ilist(labels["dim_labels"]), {k: ilist(v) for k, v in labels["val_labels"].items()}, data))


    @staticmethod
    def old_read_file(path):

        path = HyperFrame._strip_path(path)

        dir_ = path+"/"

        subprocess.run(["mv", path + ".hyperframe", path + ".zip" ])
        subprocess.run(["unzip", path + ".zip", "-d", dir_])
        subprocess.run(["mv", path + ".zip", path + ".hyperframe" ])

        data = np.load(os.path.join(dir_, "data.npy"))

        with open(os.path.join(dir_, "labels.json"), "r") as f:
            labels = json.loads(f.read())

        subprocess.run(["rm", "-r", dir_ ])

        return(HyperFrame(ilist(labels["dim_labels"]), {k: ilist(v) for k, v in labels["val_labels"].items()}, data))