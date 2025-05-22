import numpy as np
import pandas as pd
import sklearn as skl
import sys

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
          "\n12: View Matrix File Template",
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

    return choice
#-----------------------------------------------#

def matrix_ops(choice:int, mat1:np.ndarray, mat2:np.ndarray):
    match choice:
        case 0:
            mat_add(mat1, mat2)
        case 1:
            mat_sub(mat1, mat2)
        case 2:
            mat_mult(mat1, mat2)
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

def mat_templ_example():
    with open("mat_template.txt") as mat_tp:
        print(mat_tp.read())

def parse_matrix(in_mat_str:str, mode:str):
    ret_arr = None
    if mode == "ol":
        ret_arr = np.array([list(map(int, row.strip().split())) 
                            for row in in_mat_str.strip().split(';')])
    elif mode == "vr":
        ret_arr = np.array([list(map(int, line.split())) 
                            for line in in_mat_str.strip().split('\n')])
    return ret_arr


def main():
    choice:int = menu()
    print("Please enter your two matrix file names that are IN ROW MAJOR")
    mat1_fn:str = input("Matrix #1 File Name: ")
    mat2_fn:str = input("Matrix #2 File Name: ")
    
    #populate the actual matrices via file contents
    mat1 = np.array([], dtype=object)
    mat2 = np.array([], dtype=object)
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

    print(mat1)
    print(mat2)

    matrix_ops(choice, mat1, mat2)

if __name__ == '__main__':
    main()
