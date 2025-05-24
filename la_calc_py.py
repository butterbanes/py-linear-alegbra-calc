import numpy as np
import numpy.typing as npt
import pathlib as pl
import sys
from typing import Literal

def menu() -> int:
    
    print("---------------")
    print("CALCULATOR MENU")
    print("---------------")

    print("0:  Matrix Addition",
          "\n1:  Matrix Subtraction",
          "\n2:  Matrix Multiplication",
          "\n3:  Matrix Transposition",
          "\n4:  Determinant",
          "\n5:  Inverse Matrix",
          "\n6:  Matrix Ranking",
          "\n7:  Row Reduction/Gaussian Elimination",
          "\n8:  Eigenvalues and Eigenvectors",
          "\n9:  Lower and Upper Triangle Decomposition",
          "\n10: Orthogonal Matrix and Upper Triangle Decomposition",
          "\n11: Singular Value Decomposition",
          "\n12: Refresher On Matrix File Layout",
          "\n13: Change Matrices",
          "\n-1: EXIT")
    print("---------------")
    
    choice:int = int(input("Enter your choice: "))
    while choice < -1 or choice > 13:
        print("Invalid Choice | Try Again")
        choice:int = int(input("Enter your choice: "))
   
    if choice == -1:
        sys.exit(0)

    if choice == 12:
        mat_templ_example()
        choice = menu()

    return choice
#-----------------------------------------------#

def matrix_ops(choice:int, mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]):
    try:
        match choice:
            case 0:
                print(f"Matrix Addition Result:\n{mat_add(mat1, mat2)}")
            case 1:
                print(f"Matrix Subtraction Result:\n{mat_sub(mat1, mat2)}")
            case 2:
                print(f"Matrix Multiplication Result:\n{mat_mult(mat1, mat2)}")
            case 3:
                print(f"Matrix Transposition Result:\n{mat_transpose(mat1, mat2)}")
            case 4:
                print(f"Matrix Determinant Result:\n{mat_determinant(mat1, mat2)}")
            case 5:
                print(f"Matrix Inverse Result:\n{mat_inverse(mat1, mat2)}")
            case 6:
                print(f"Matrix Ranking Result:\n{mat_rank(mat1, mat2)}")
            case 7:
                mat_rr_ge(mat1, mat2)
            case 8:
                mat_eigval_eigvec(mat1, mat2)
            case 9:
                mat_lu_decomp(mat1, mat2)
            case 10:
                mat_qr_decomp(mat1, mat2)
            case 11:
                mat_svd(mat1, mat2)
            case 13:
                mat1, mat2 = change_matrices(mat1, mat2)
            case _:
                print("ACHIEVEMENT: How did we get here?")
    except np.linalg.LinAlgError as lae:
        print(f"LinAlgError Raised: {str(lae)}")
    return mat1, mat2

#-----------------------------------------------#

#--- SINGLE MATRIX OPERATIONS ---#
def mat_rank(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]) -> int:
    selected_mat:npt.NDArray[np.float64] = np.array([], dtype=np.float64) 
    match parse_which(mat1, mat2):
        case 1:
            selected_mat = mat1
        case 2:
            selected_mat = mat2
    return np.linalg.matrix_rank(selected_mat)

def mat_transpose(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    selected_mat:npt.NDArray[np.float64] = np.array([], dtype=np.float64) 
    match parse_which(mat1, mat2):
        case 1:
            selected_mat = mat1
        case 2:
            selected_mat = mat2
    return selected_mat.transpose()


def mat_determinant(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    selected_mat:npt.NDArray[np.float64] = np.array([], dtype=np.float64) 
    match parse_which(mat1, mat2):
        case 1:
            selected_mat = mat1
        case 2:
            selected_mat = mat2
    sign, logdet = np.linalg.slogdet(selected_mat)
    res = sign * np.exp(logdet)
    return res

def mat_inverse(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    use_both:bool = False
    print(mat1)
    print(mat2)
    usr_inp = input("Use an already existing matrix for the b-var?").strip().lower() 
    if usr_inp in ("y", "yes"):
        use_both = True
    elif usr_inp in ("n", "no"):
        use_both = False
    else:
        print("ERR: Invalid choice | try again")
        mat_inverse(mat1, mat2)
    A_mat:npt.NDArray[np.float64] = np.array([], dtype=np.float64)
    b_mat:npt.NDArray[np.float64] = np.array([], dtype=np.float64)
    if use_both is True:
        print(mat1, mat2)
        print("Choose which matrix you would like to use as your A-var\n")
        print("Whatever matrix is chosen for A, the other will be assigned for b [mat1 | mat2]: ")
        match parse_which(mat1, mat2):
            case 1:
                A_mat = mat1
                b_mat = mat2
            case 2:
                A_mat = mat2
                b_mat = mat1
        return np.array((np.linalg.solve(A_mat, b_mat)), dtype=np.float64)
    
    #Jumps here on False
    match parse_which(mat1, mat2):
        case 1:
            A_mat = mat1
            in_mat_str: str = input("Enter a file or an array to use for your b variable: ")
            if pl.Path(in_mat_str).is_file():
                with open(in_mat_str) as b_vf:
                    temp_mat_str = b_vf.read()
                    if ";" in temp_mat_str:
                        b_mat = parse_matrix_str(temp_mat_str, "ol")
                    elif ";" not in temp_mat_str:
                        b_mat = parse_matrix_str(temp_mat_str, "vr")
        case 2:
            A_mat = mat2
            in_mat_str: str = input("Enter a file or an array to use for your b variable: ")
            if pl.Path(in_mat_str).is_file():
                with open(in_mat_str) as b_vf:
                    temp_mat_str = b_vf.read()
                    if ";" in temp_mat_str:
                        b_mat = parse_matrix_str(temp_mat_str, "ol")
                    elif ";" not in temp_mat_str:
                        b_mat = parse_matrix_str(temp_mat_str, "vr")
    
    return np.array((np.linalg.solve(A_mat, b_mat)), dtype=np.float64)


#--- END SINGLE MATRIX OPERATIONS ---#

#--- BASIC MATRIX ARITHMETIC ---#
def mat_add(mat1: npt.NDArray[np.float64], mat2: npt.NDArray[np.float64]) -> npt.NDArray[np.float128]:
    if mat1.size != mat2.size:
        print("ERR: to perform addition  on two matrices, they must be the same size")
        sys.exit(1)

    mat_add_res:npt.NDArray[np.float128] = np.add(mat1, mat2)
    return mat_add_res

def mat_sub(mat1: npt.NDArray[np.float64], mat2: npt.NDArray[np.float64]) -> npt.NDArray[np.float128]:
    if mat1.size != mat2.size:
        print("ERR: to perform subtraction on two matrices, they must be the same size")
        sys.exit(1)

    mat_sub_res:npt.NDArray[np.float128] = np.subtract(mat1, mat2)
    return mat_sub_res

def mat_mult(mat1: npt.NDArray[np.float64], mat2: npt.NDArray[np.float64]) -> npt.NDArray[np.float128]:
    mat_mult_res:npt.NDArray[np.float128] = np.multiply(mat1, mat2)
    return mat_mult_res

#--- END BASIC MATRIX ARITHMETIC ---#

def mat_templ_example():
    with open("mat_template.txt") as mat_tp:
        print(mat_tp.read())

def parse_matrix_str(in_mat_str: str, mode: Literal["ol", "vr"]) -> npt.NDArray[np.float64]:
    if mode == "ol":
        return np.array(
            [list(map(np.float64, row.strip().split())) for row in in_mat_str.strip().split(';')],
            dtype=np.float64)
    elif mode == "vr":
        return np.array(
            [list(map(np.float64, line.strip().split())) for line in in_mat_str.strip().split('\n')],
            dtype=np.float64)

# This function is for those matrix operation functions that only require one
#   matrix to work (i.e. transposition, determinant)
def parse_which(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]) -> int:
    print(f"Matrix #1\n{mat1}")
    print(f"Matrix #2:\n{mat2}")
    choice = input("Which matrix would you like to use as your main matrix? [m1 | m2]: ").strip().lower()
    ret_ans: int = -1 
    if choice in ("1", "m1", "mat1", "matrix1"):
        ret_ans = 1
    elif choice in ("2", "m2", "mat2", "matrix2"):
        ret_ans = 2
    else:
        print("ERR: Invalid choice | try again")
        return parse_which(mat1, mat2)
    
    return ret_ans 

def change_matrices(mat1:npt.NDArray[np.float64], mat2:npt.NDArray[np.float64]):
    ch_mat1:str = input("Change Matrix #1? [y/n]: ").strip().lower()
    ch_mat2:str = input("Change Matrix #2? [y/n]: ").strip().lower()
    if ch_mat1 in ("y", "yes"):
        mat1_fn:str = input("New Matrix #1 File Name: ")
        with open(mat1_fn) as m1_f:
            temp_mat_str = m1_f.read()
            if ";" in temp_mat_str:
                mat1 = parse_matrix_str(temp_mat_str, "ol")
            elif ";" not in temp_mat_str:
                mat1 = parse_matrix_str(temp_mat_str, "vr")
    if ch_mat2 in ("y", "yes"):
        mat2_fn:str = input("New Matrix #2 File Name: ")
        with open(mat2_fn) as m2_f:
            temp_mat_str = m2_f.read()
            if ";" in temp_mat_str:
                mat2 = parse_matrix_str(temp_mat_str, "ol")
            elif ";" not in temp_mat_str:
                mat2 = parse_matrix_str(temp_mat_str, "vr")

    return mat1, mat2

def main():
    mat_templ_example()
    print("Please enter your two matrix file names that are IN ROW MAJOR")
    mat1_fn:str = input("Matrix #1 File Name: ")
    mat2_fn:str = input("Matrix #2 File Name: ")
    
    #populate the actual matrices via file contents
    mat1:npt.NDArray[np.float64] = npt.NDArray[np.float64](np.array([], dtype=object))
    mat2:npt.NDArray[np.float64] = npt.NDArray[np.float64](np.array([], dtype=object))
    with open(mat1_fn) as m1_f:
        temp_mat_str = m1_f.read()
        if ";" in temp_mat_str:
            mat1 = parse_matrix_str(temp_mat_str, "ol")
        elif ";" not in temp_mat_str:
            mat1 = parse_matrix_str(temp_mat_str, "vr")
    with open(mat2_fn) as m2_f:
        temp_mat_str = m2_f.read()
        if ";" in temp_mat_str:
            mat2 = parse_matrix_str(temp_mat_str, "ol")
        elif ";" not in temp_mat_str:
            mat2 = parse_matrix_str(temp_mat_str, "vr")
    choice = 99
    while choice != -1: 
        choice = menu()
        mat1, mat2 = matrix_ops(choice, mat1, mat2)
        

        

if __name__ == '__main__':
    main()
