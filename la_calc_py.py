import numpy as np
import numpy.typing as npt
import pandas as pd
import re as regex
import sklearn as skl
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
          "\n-1: EXIT")
    print("---------------")
    
    choice:int = int(input("Enter your choice: "))
    while choice < -1 or choice > 12:
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
    match choice:
        case 0:
            print(f"Mat_Add_Result:\n{mat_add(mat1, mat2)}")
        case 1:
            print(f"Matrix Subtraction Result:\n{mat_sub(mat1, mat2)}")
        case 2:
            print(f"Matrix Multiplication Result:\n{mat_mult(mat1, mat2)}")
        case 3:
            mat_transpose(mat1, mat2)
        case 4:
            #this function will ask which matrix to find the determinant of
            mat_determinanti(mat1, mat2)
        case 5:
            mat_inverse(mat1, mat2)
        case 6:
            mat_rank(mat1, mat2)
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
        case _:
            print("ACHIEVEMENT: How did we get here?")

#-----------------------------------------------#

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

def mat_templ_example():
    with open("mat_template.txt") as mat_tp:
        print(mat_tp.read())

def parse_matrix(in_mat_str: str, mode: Literal["ol", "vr"]) -> npt.NDArray[np.float64]:
    if mode == "ol":
        return np.array(
            [list(map(np.float64, row.strip().split())) for row in in_mat_str.strip().split(';')],
            dtype=np.float64)
    elif mode == "vr":
        return np.array(
            [list(map(np.float64, line.strip().split())) for line in in_mat_str.strip().split('\n')],
            dtype=np.float64)

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
            mat1 = parse_matrix(temp_mat_str, "ol")
        elif ";" not in temp_mat_str:
            mat1 = parse_matrix(temp_mat_str, "vr")
    with open(mat2_fn) as m2_f:
        temp_mat_str = m2_f.read()
        if ";" in temp_mat_str:
            mat2 = parse_matrix(temp_mat_str, "ol")
        elif ";" not in temp_mat_str:
            mat2 = parse_matrix(temp_mat_str, "vr")
    choice:int = menu()
    answer:str = ""
    while choice != -1: 
        matrix_ops(choice, mat1, mat2)
        answer = input("Continue with the same original matrices? [y/n]")
        normalized:str = answer.strip().lower()
        
        while normalized not in ("y", "yes","n", "no"):           
            print("ERR: Invalid input | try again")
            answer = input("Continue with the same original matrices? [y/n]")
            normalized = answer.strip().lower()

        if normalized in ("n", "no"):
            print("Please enter your two matrix file names that are IN ROW MAJOR")
            mat1_fn:str = input("Matrix #1 File Name: ")
            mat2_fn:str = input("Matrix #2 File Name: ")
            with open(mat1_fn) as m1_f:
                temp_mat_str = m1_f.read()
                if ";" in temp_mat_str:
                    mat1 = parse_matrix(temp_mat_str, "ol")
                elif ";" not in temp_mat_str:
                    mat1 = parse_matrix(temp_mat_str, "vr")
            with open(mat2_fn) as m2_f:
                temp_mat_str = m2_f.read()
                if ";" in temp_mat_str:
                    mat2 = parse_matrix(temp_mat_str, "ol")
                elif ";" not in temp_mat_str:
                    mat2 = parse_matrix(temp_mat_str, "vr")
            choice = menu()
        elif normalized in ("y", "yes"):
            choice = menu()
        

        

if __name__ == '__main__':
    main()
