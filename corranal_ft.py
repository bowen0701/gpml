# corranal.py: Python Correspondence Analysis
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp 
import pandas as pd
from numpy.linalg import svd

def _coordinates_df(array_x1, array_x2, rows, cols):
    """Create pandas DataFrame with coordinates in rows and columns.
        
    Args:
      array_x1: numpy array, coordinates in rows.
      array_x2: numpy array, coordinates in columns.
      rows: numpy array, row group name.
      cols: numpy array, column group name.
        
    Returns:
      coord_df: pandas DataFrame with columns {'x_1',..., 'x_K', 'point', 'coord'}:
        x_k, k=1,...,K: K-dimensional coordinates.
        point: row and column points for labeling.
        coord: {'row', 'col'}, indicates row point or column point.
    """
    row_df = pd.DataFrame(
        array_x1, columns=['x' +  str(i) for i in (np.arange(array_x1.shape[1]) + 1)])
    row_df['point'] = rows
    row_df['coord'] = 'row'
    col_df = pd.DataFrame(
        array_x2, columns=['x' +  str(i) for i in (np.arange(array_x2.shape[1]) + 1)])
    col_df['point'] = cols
    col_df['coord'] = 'col'         
    coord_df = pd.concat([row_df, col_df], ignore_index=True)
    return coord_df

def ca(df, option='symmetric'):
    """Correspondence Analysis.
    
    This method performs generalized singular value decomposition (SVD) for
    correspondence matrix and computes principal and standard coordinates for
    rows and columns.
    
    ### Usage
    
    ```
    ca_output = ca(author_data, option='symmetric')
    coord_df = ca_output['coordinates']
    inertia_prop = ca_output['inertia_proportion']
    ```
    
    Args:
      df: pandas DataFrame, with row and column names.
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
      d : dict, information dict.
        * d['coordinates']: pandas DataFrame, contains coordinates, row and column points. 
        * d['inertia_proportion']: numpy array, contains proportions of total inertia explained 
            in coordinate dimensions.
    
    Raises:
      TypeError: The input is not a pandas DataFrame
      ValueError: Numpy array contains missing values.
      TypeError: Numpy array contains data types other than numeric.
      ValueError: Option only includes {"symmetric", "rowprincipal", "colprincipal"}.
    """

    if isinstance(df, pd.DataFrame) is not True:
        raise TypeError('The input is not a pandas DataFrame.')  
    rows = np.array(df.index)
    cols = np.array(df.columns)
    np_data = np.array(df.values)      
    if np.isnan(np_data).any():
        raise ValueError('Numpy array contains missing values.')
    if np.issubdtype(np_data.dtype, np.number) is not True:
        raise TypeError('Numpy array contains data types other than numeric.')

    p_corrmat = np_data / np_data.sum()
    r_profile = p_corrmat.sum(axis=1).reshape(p_corrmat.shape[0], 1)
    c_profile = p_corrmat.sum(axis=0).reshape(p_corrmat.shape[1], 1)
    Dr_invsqrt = np.diag(np.power(r_profile, -1/2).T[0])
    Dc_invsqrt = np.diag(np.power(c_profile, -1/2).T[0])
    ker_mat = np.subtract(p_corrmat, np.dot(r_profile, c_profile.T))
    left_mat = Dr_invsqrt
    right_mat = Dc_invsqrt
    weighted_lse = left_mat.dot(ker_mat).dot(right_mat)
    U, sv, Vt = svd(weighted_lse, full_matrices=False)
    V = Vt.T
    SV = np.diag(sv)
    inertia = np.power(sv, 2)
    
    if option == 'symmetric':
        # Symmetric map with pricipal coordinates for rows and columns.
        F = Dr_invsqrt.dot(U).dot(SV)
        G = Dc_invsqrt.dot(V).dot(SV)
        coordinates = _coordinates_df(F, G, rows, cols)
    elif option == 'rowprincipal':
        # Asymmetric map with principal coordinates for rows and standard ones for columns.
        F = Dr_invsqrt.dot(U).dot(SV)
        Gam = Dc_invsqrt.dot(V)
        coordinates = _coordinates_df(F, Gam, rows, cols)
    elif option == 'colprincipal':
        # Asymmetric map with standard coordinates for rows and principal ones for columns.
        Phi = Dr_invsqrt.dot(U)
        G = Dc_invsqrt.dot(V).dot(SV)
        coordinates = _coordinates_df(Phi, G, rows, cols)
    else:
        raise ValueError(
            'Option only includes {"symmetric", "rowprincipal", "colprincipal"}.')
    
    inertia_proportion = (inertia / inertia.sum())

    d = {'coordinates': coordinates, 
         'inertia_proportion': inertia_proportion}
    
    return d
