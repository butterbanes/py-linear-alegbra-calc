# Python Linear Algebra Calculator
This program acts as a wrapper program for various NumPy and SciPy functions relating to the solving of matrices.
 - By "wrapper program", I simply mean that it implements the use of mostly predefined methods given by the NumPy and SciPy python modules. The exceptions to this are the functions regarding Gaussian Elimination/Row Echelon Form function, as well as the Reduced Row Echelon Form function and a few other helper functions I defined.

## Installation
For this repository, there exists a text document called "requirements.txt" which withhold the dependencies needed for this program to properly function. Make sure that your virtual environment is active before running the pip-install command
 - Using Python Virtual Environments: [Python Virtual Environment Tutorial](https://gist.github.com/ryumada/c22133988fd1c22a66e4ed1b23eca233#python-virtual-environment-tutorial) -- by @ryumada

```bash
# example using pip-install
pip install -r requirements.txt
```
## Usage
As of now (to be updated in the future), this program only takes in .txt files that hold **2D** matrices in one of two representations that will work. One approach uses semicolons (;) to separate the matrix rows, another approach is simply separating each row by a newline. This program takes in two matrices to start, which can be changed in-program whenever the menu is presented.

### Template 
```bash
# EXAMPLE (2x5 one-line using semicolons):
1 2 3 4 5; 6 7 8 9 10

# EXAMPLE (2x5 visually represented)
1 2 3 4 5
6 7 8 9 10
```

### Using the Program
```bash
python3 la_calc_py.py some_matrix_1.txt some_matrix_2.txt
```
**\*\*You have to provide two matrices as arguments\*\***

## Functions/Features
```python
# adds one matrix with another (both must be the same size)
mat_add(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# subtracts one matrix from another (both must be the same size)
mat_sub(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# multiplies one matrix by another
mat_mult(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# flips rows and cols of a specified matrix
mat_transpose(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns determinant of a specified matrix
mat_determinant(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns inverse of a specified square non-zero determinated matrix
mat_inverse(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns number of linearly independent rows or cols in a specified matrix
mat_rank(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# converts specified matrix into Row Echelon Form OR Reduced Row Echelon Form
mat_rr_ge(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns Eigenvalues and Eigenvectors of a specified matrix
mat_eigval_eigvec(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns permuation matrix, lower triangular matrix, and upper triangular matrix of a decomposed specified matrix
mat_lu_decomp(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns orthogonal matrix and upper triangular of decomposed specified matrix
mat_qr_decomp(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])

# returns U → m*m orthogonal matrix, ∑ → m*n matrix, and Vᵗ → n*n orthogonal matrix
mat_svd(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64])
```
## Prerequisites
- Python 3.10+
- NumPy
- SciPy

##### Author: Ethan Smith ([butterbanes](https://www.github.com/butterbanes))
##### Contact: bbanes.dev@gmail.com
