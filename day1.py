--- 1.py

import os
import sys
if 'google.colab' in sys.modules:
    if not os.path.isdir('mglearn'):
        !wget -q -O mglearn.tar.gz https://bit.ly/mglearn-tar-gz
        !tar -xzf mglearn.tar.gz

--- 2.py

import sklearn
from preamble import *

--- 3.py

import numpy as np

x =np.array([[1,2,3], [4,5,6]])
print("x:\n", x)

--- 4.py

from scipy import sparse

eye = np.eye(4)
print("NumPy 배열 :\n", eye)

--- 5.py

sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy의 CSR 행렬 :\n", sparse_matrix)

---
