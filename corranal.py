from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp 
import pandas as pd
from numpy.linalg import svd

class CA(object):
    """Correspondence analysis (CA).
    
    ### Usage

    ```python
    corranal = CA(aggregate_cnt)
    corranal.fit()
    coord_df = corranal.get_coordinates()
    inertia_prop = corranal.score_inertia()
    ```

    @@__init__
    
    @@fit
    @@get_coordinates
    @@score_inertia
    """

    def __init__(self, df):
        """Create a new Correspondence Analysis.
        
        Args:
          df: pandas DataFrame, with row and column names.
          
        Raises:
          TypeError: The input is not a pandas DataFrame
          ValueError: Numpy array contains missing values.
          TypeError: Numpy array contains data types other than numeric.
        """
        if isinstance(df, pd.DataFrame) is not True:
            raise TypeError('The input is not a Pandas DataFrame.')  
        self._rows = np.array(df.index)
        self._cols = np.array(df.columns)
        self._np_data = np.array(df.values)      
        if np.isnan(self._np_data).any():
            raise ValueError('Numpy array contains missing values.')
        if np.issubdtype(self._np_data.dtype, np.number) is not True:
            raise TypeError('Numpy array contains data types other than numeric.')

    def fit(self):
        """Compute Correspondence Analysis.
        
        This method performs generalized singular value decomposition (SVD) for
        correspondence matrix and computes principal and standard coordinates for
        rows and columns.
        """     
        p_corrmat = self._np_data / self._np_data.sum()
        r_profile = p_corrmat.sum(axis=1).reshape(p_corrmat.shape[0], 1)
        c_profile = p_corrmat.sum(axis=0).reshape(p_corrmat.shape[1], 1)
        Dr_invsqrt = np.diag(np.power(r_profile, -1/2).T[0])
        Dc_invsqrt = np.diag(np.power(c_profile, -1/2).T[0])
        ker_mat = np.subtract(p_corrmat, np.dot(r_profile, c_profile.T))
        left_mat = Dr_invsqrt
        right_mat = Dc_invsqrt
        weighted_lse = left_mat.dot(ker_mat).dot(right_mat)
        U, sv, Vt = svd(weighted_lse, full_matrices=False)
        self._Dr_invsqrt = Dr_invsqrt
        self._Dc_invsqrt = Dc_invsqrt
        self._U = U
        self._V = Vt.T
        self._SV = np.diag(sv)
        self._inertia = np.power(sv, 2)
        # Pricipal coordinates for rows and columns.
        self._F = self._Dr_invsqrt.dot(self._U).dot(self._SV)
        self._G = self._Dc_invsqrt.dot(self._V).dot(self._SV)
        # Standard coordinates for rows and columns.
        self._Phi = self._Dr_invsqrt.dot(self._U)
        self._Gam = self._Dc_invsqrt.dot(self._V)
        return self
    
    def _coordinates_df(self, array_x1, array_x2):
        """Create pandas DataFrame with coordinates in rows and columns.
        
        Args:
          array_x1: numpy array, coordinates in rows.
          array_x2: numpy array, coordinates in columns.
        
        Returns:
          coord_df: A Pandas DataFrame with columns 
            {'x_1',..., 'x_K', 'point', 'coord'}:
            - x_k, k=1,...,K: K-dimensional coordinates.
            - point: row and column points for labeling.
            - coord: {'row', 'col'}, indicates row point or column point.
        """
        row_df = pd.DataFrame(
            array_x1, columns=['x' +  str(i) for i in (np.arange(array_x1.shape[1]) + 1)])
        row_df['point'] = self._rows
        row_df['coord'] = 'row'
        col_df = pd.DataFrame(
            array_x2, columns=['x' +  str(i) for i in (np.arange(array_x2.shape[1]) + 1)])
        col_df['point'] = self._cols
        col_df['coord'] = 'col'         
        coord_df = pd.concat([row_df, col_df], ignore_index=True)
        return coord_df
    
    def get_coordinates(self, option='symmetric'):
        """Take coordinates in rows and columns for symmetric or assymetric map.
        
        - For symmetric map, we can catch row-to-row and column-to-column 
        association (maybe not row-to-column association); 
        - For asymmetric map, we can further catch row-to-column association.
        
        Args:
          option: string, in one of the following three:
            'symmetric': symmetric map with 
              - rows and columns in principal coordinates.
            'rowprincipal': asymmetric map with 
              - rows in principal coordinates and 
              - columns in standard coordinates.  
            'colprincipal': asymmetric map with 
              - rows in standard coordinates and 
              - columns in principal coordinates.
        
        Returns:
          Pandas DataFrame, contains coordinates, row and column points.
        
        Raises:
          ValueError: Option only includes {"symmetric", "rowprincipal", "colprincipal"}.
        """     
        if option == 'symmetric':
            # Symmetric map
            return self._coordinates_df(self._F, self._G)
        elif option == 'rowprincipal':
            # Row principal asymmetric map
            return self._coordinates_df(self._F, self._Gam)
        elif option == 'colprincipal':
            # Column principal asymmetric map
            return self._coordinates_df(self._Phi, self._G)
        else:
            raise ValueError(
                'Option only includes {"symmetric", "rowprincipal", "colprincipal"}.')

    @property
    def score_inertia(self):
        """Score inertia.
        
        Returns:
          numpy array, contains proportions of total inertia explained 
            in coordinate dimensions.
        """
        inertia = self._inertia
        self._inertia_prop = (inertia / inertia.sum()).cumsum()
        return self._inertia_prop
