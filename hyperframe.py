import numpy as np
import pandas as pd


#for testing
alphabet = "abcdefghijklmnopqrstuvwxyz"

def nest(l, n=1):
    if n == 1:
        return(l)
    else:
        return([nest(l, n-1), nest(l, n-1)])

def build_dim_labels(m):
    return(idict(list(alphabet[:m])))

def build_val_labels(m, n):
    return(dict(zip(alphabet, [idict(list(alphabet.upper()[:n])) for _ in range(m)])))




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
    return(dict(zip(range(len(list_)), list_)))

class HyperFrame:
    def __init__(self, data, dim_labels, val_labels):
        
        self.validate_data(data, dim_labels, val_labels)
        self.data = data
        self.dim_labels = dim_labels
        self.val_labels = val_labels
        
        self.rdim_labels = {v: k for k, v in dim_labels.items()}
        self.rval_labels = {k: {v2: k2 for k2, v2 in v.items()} for k, v in val_labels.items()}
        
    def copy(self):
        return(HyperFrame(self.data.copy(), dict(self.dim_labels), dict(self.val_labels)))
    
    def iget(self, **kwargs):
        """
        get the HyperFrame with the 
        """
        self.validate_kwargs(kwargs)
                        
        ndim_labels = [(i, v)
                       for i, v in self.dim_labels.items()
                       if v not in kwargs.keys() or len(kwargs[v]) > 1]

        ndim_labels = idict([x[1] for x in sorted(ndim_labels, key=lambda y: y[0])])
        
        nval_labels = {k: kwargs.get(k, v) for k, v in self.val_labels.items() if len(kwargs.get(k, v)) > 1}
        nval_labels = {k: (v if isinstance(v, dict) else idict(mlist(v))) for k, v in nval_labels.items()}

        ndata = self.data.copy()

        indices = self.construct_indices(kwargs, ndata.shape)
        ndata = eval("ndata[{}]".format(indices))
            
        hdata = HyperFrame(ndata, ndim_labels, nval_labels)
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
        
        assert self.hget(**kwargs).data.shape == new_data.shape 
        
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
    