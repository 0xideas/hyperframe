import numpy as np
import pandas as pd
from collections import OrderedDict


#for testing
alphabet = "abcdefghijklmnopqrstuvwxyz"

def nest(l, n=1):
    if n == 0:
        return(None)
    elif n == 1:
        return(l)
    else:
        return([nest(l, n-1)]*len(l))

def build_dim_labels(m):
    return(list(alphabet[:m]))

def build_val_labels(m, n):
    return(dict(zip(alphabet, [list(alphabet.upper()[:n]) for _ in range(m)])))


#initialisation
def constant_array(constant, *args):
    if len(args) == 0:
        return(constant)
    else:
        return(np.array([constant_array(constant, *args[1:])]*args[0]))

def zeros(*args):
    return(constant_array(0.0, *args))

def ones(*args):
    return(constant_array(1.0, *args))

#helpers
def unnest(l):
    '''unnest a nested list l'''
    return([x for y in l for x in mlist(y)])

def mlist(x):
    '''make sure x is a list'''
    if isinstance(x, list):
        return(x)
    else:
        return([x])

def idict(list_):
    return(OrderedDict(zip(range(len(list_)), list_)))

#reverse k, v for idict
def ridict(list_):
    return(OrderedDict({v:k for k, v in idict(list_).items()}))


def ilist(dict_):
    return(list(dict_.values()))

def rilist(dict_):
    return(list(dict_.keys()))

class HyperFrame:
    def __init__(self, dimension_labels, index_labels, data=None):

        assert isinstance(dimension_labels, list)

        for k, v in index_labels.items():
            assert( isinstance(v, list))

        index_labels = OrderedDict(index_labels)

        #refactor to make this reversal unnessecary
        self.dim_labels = idict(dimension_labels)
        self.val_labels = OrderedDict({k: idict(v) for k, v in index_labels.items()})
        
        self.rdim_labels = ridict(dimension_labels)
        self.rval_labels = OrderedDict({k: ridict(v) for k, v in index_labels.items()})
        

        if data is None:
            data = zeros(*[len(self.val_labels[dim_label]) for _, dim_label in self.dim_labels.items()])
        
        self.validate_data(data, self.dim_labels, self.val_labels)

        self.data = data



    def copy(self):
        index_labels = OrderedDict({k: rilist(v) for k, v in self.rval_labels.items()})
        return(HyperFrame( ilist(self.dim_labels), index_labels , self.data.copy()))
    
    def iget(self, **kwargs):
        """
        get the HyperFrame with the 
        """
        self.validate_kwargs(kwargs)
                        
        ndim_labels = [(i, v)
                       for i, v in self.dim_labels.items()
                       if v not in kwargs.keys() or len(kwargs[v]) > 1]

        ndim_labels = [x[1] for x in sorted(ndim_labels, key=lambda y: y[0])]
        
        nval_labels = {k: kwargs.get(k, v) for k, v in self.val_labels.items() if len(kwargs.get(k, v)) > 1}
        nval_labels = {k: (mlist(v) if not isinstance(v, dict) else list(v.values())) for k, v in nval_labels.items()}

        ndata = self.data.copy()

        indices = self.construct_indices(kwargs, ndata.shape)
        ndata = eval("ndata[{}]".format(indices))
            
        hdata = HyperFrame(ndim_labels, nval_labels, ndata)
        return(hdata)
    
    def sort_dict(self, dict_):
        return([(k, v) for k, v in sorted(list(dict_.items()), key= lambda x: x[0])])
    
    def iget0(self, *args, return_type=None):
        assert len(args) > 0 and len(args) < 3
        kwargs = {v: self.val_labels[v][0] for i, v in self.dim_labels.items() if v not in args}
        subset = self.iget(**kwargs)
        
        if return_type is None:
            return(subset)
        elif return_type == "numpy":
            return(subset.data)
        elif return_type == "pandas":            
            indices = [[v2 for k2, v2 in self.sort_dict(v)] 
                       for k, v in self.sort_dict(subset.val_labels)]
            
            assert len(indices) == len(args)
            
            index = indices[0]
            columns = indices[1]
            
            return(pd.DataFrame(subset.data, columns = columns, index=index))
    
    
    def iset(self, new_data,  **kwargs):
        
        self.validate_kwargs(kwargs)
        
        assert self.iget(**kwargs).data.shape == new_data.shape 
        
        hframe = self.copy()
            
        indices = self.construct_indices(kwargs, hframe.data.shape)
        exec("hframe.data[{}] = new_data".format(indices))
            
        return(hframe)
    
    def construct_indices(self, kwargs, shape):
        numpy_indices = [":"]*len(shape)
        for dim_label, target_labels in kwargs.items():
            dim = self.rdim_labels[dim_label]
            target_indices = {dim:[self.rval_labels[dim_label][target] for target in target_labels]}

            for k, v in target_indices.items():
                v2 = v if len(v) > 1 else v[0]
                numpy_indices[k] = str(v2)
                
        return(",".join(numpy_indices))

    
    def validate_kwargs(self, kwargs):
        assert(len(kwargs) > 0)
        for key, value in kwargs.items():
            assert key in self.dim_labels.values()
            
            assert(len(value) > 0)
            for v in value:
                assert(v in self.val_labels[key].values())
        
    
    def validate_data(self, data, dim_labels, val_labels):

        assert isinstance(data, type(np.array([0])))
        
        assert len(dim_labels) == len(val_labels)
        
        self.validate_dict(dim_labels, len(data.shape))
        
        for dim, dim_label in dim_labels.items():
            assert dim_label in val_labels.keys()
            
            self.validate_dict(val_labels[dim_label], data.shape[dim])
            
    def validate_dict(self, dict_, dims):
        assert len(dict_) == dims
        assert self.dense_keys(dict_)
        
        
    def dense_keys(self, dict_):
        dict_keys = dict_.keys()
        return np.all([x in dict_keys for x in range(len(dict_))])
    