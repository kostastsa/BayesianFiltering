import numpy as np

def collapse(mean_mat, covariance_tens, weight_vec):
        M,dx = np.shape(mean_mat)
        mean_out = np.matmul(weight_vec, mean_mat)
        cov_out = np.zeros([dx,dx])
        for m in range(M):
            diff = mean_mat[m] - mean_out
            cov_out = cov_out + weight_vec[m]*(covariance_tens[m] + 
                                               np.tensordot(diff, diff, axes =0))
        return mean_out, cov_out

def dec_to_base(num,base):  #Maximum base - 36
    base_num = ""
    while num>0:
        dig = int(num%base)
        if dig<10:
            base_num += str(dig)
        else:
            base_num += chr(ord('A')+dig-10)  #Using uppercase letters
        num //= base
    base_num = base_num[::-1]  #To reverse the string
    return base_num
            
        