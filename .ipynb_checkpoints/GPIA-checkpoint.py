'''
Gaussian Process Regressor with Interval Arithmetic 

Implements GP Regression via algorithm 2.1 from "Gaussian Processes for Machine Learning"

Rigorously tracks errors of function input via interval arithmetic

'''
# function imports
import numpy as np # the classics
from scipy.linalg import cholesky, cho_solve # for matrix inversion
from scipy.spatial.distance import cdist # for defining K(X,X)
from interval import interval, imath, fpu # for interval arithmetic


# Regressor Implementation implementation
# Interval Arithmetic implementation

class GPIA_Regressor():
    '''
    Description
    '''

def Z_matrix(X,Y,tau = 5,sig_sq = 1e-10):
    """
    Input: 
        X: an m by n matrix (training input)
        Y: an m-vector (training output)
        tau: parameter in the squared exponential kernal
        sig_sq: noise level of errors in GP model.  Corresponds to alpha in sklearn
            Note: even if zero noise is assumed, sig_sq should be set positive to avoid numerical errors in the matrix inversion.    
    Output:
        Z: an m-vector, Z = (K(X,X)+sig_sq*I)^(-1)Y
    """
    # finding m
    m = X.shape[0]
    
    # defining the covariance matrix
    dists = cdist(X / tau, X / tau, metric="sqeuclidean")
    K = np.exp(-0.5 * dists)
    
    # Defining matrix that must be inverted
    A = K + sig_sq*np.eye(m)
    
    # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
    L = cholesky(A, lower=True, check_finite=False)
    
    # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ Y)
    Z = cho_solve((L, True),Y,check_finite=False)
    
    return Z

# interval squared exponential kernal
def sq_exp_intvl(a,b,tau):
    '''
    Input:
        a: length q vector
        b: length q vector
        tau: parameter in square exponential kernal
    Output: 
        k: interval result of square exponential kernal on a,b with parameter tau
    '''
    # notice that squaring the norm cancels out the square root operation
    norm_sq_list = [(interval(a[i]) - interval(b[i]))**2 for i in range(len(a))]
    norm_sq = sum(norm_sq_list)
    denom = -2*interval(tau)**2
    return imath.exp(norm_sq/denom)

# Define interval GP Regression
def GP_reg_intvl(X_train, x_star, Z, tau):
    """
    This function combines with GP_input_setup in order to do GP regression
    Evaluates at points in R^n--single points, not collection
    This version does interval arithmetic
    
    Input:
        X_train: an m by q matrix (training input)
        x_star: a 1 by q matrix (where we evaluate the regression)
        Z: m-vector, output of GP_input_setup
        tau: parameter for squared exponential kernal
        
    Output:
        Y_star: a m by q matrix, output of Gaussian regression of x_star
    """
    # finding dimensions
    m = X_train.shape[0]
    
    # defining the covariance matrix K(X*,X)
    KX_starX = []
    for j in range(m):
        xj = X_train[j,:]
        KX_starX.append(sq_exp_intvl(x_star,xj,tau))
    
    # getting result of regression, Y_star
    # have to basically redefine linear algebra here by hand to implement interval arithmetic
    Y_star = []
    for j in range(Z.shape[1]): # for 2D case (like Leslie map) Z.shape[1] = 2
        yj_list = [0]*m
        for i in range(m):
            yj_list[i] = KX_starX[i]*Z[i,j]
        Y_star.append(sum(yj_list))
                    
    return Y_star



# def Z_matrix(X,Y,tau,sig_sq = 1e-10):
#     """
#     Input: 
#         X: an m by n matrix (training input)
#         Y: an m-vector (training output)
#         tau: parameter in the squared exponential kernal
#         sig_sq: noise level of errors in GP model.  Corresponds to alpha in sklearn
#             Note: even if zero noise is assumed, sig_sq should be set positive to avoid numerical errors in the matrix inversion.    
#     Output:
#         Z: an m-vector, Z = (K(X,X)+sig_sq*I)^(-1)Y
#     """
#     # finding m
#     m = X.shape[0]
    
#     # defining the covariance matrix
#     dists = cdist(X / tau, X / tau, metric="sqeuclidean")
#     K = np.exp(-0.5 * dists)
    
#     # Defining matrix that must be inverted
#     A = K + sig_sq*np.eye(m)
    
#     # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
#     L = cholesky(A, lower=True, check_finite=False)
    
#     # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ Y)
#     Z = cho_solve((L, True),Y,check_finite=False)
    
#     return Z

# # interval squared exponential kernal
# def sq_exp_intvl(a,b,tau):
#     '''
#     Input:
#         a: length q vector
#         b: length q vector
#         tau: parameter in square exponential kernal
#     Output: 
#         k: interval result of square exponential kernal on a,b with parameter tau
#     '''
#     # notice that squaring the norm cancels out the square root operation
#     norm_sq_list = [(interval(a[i]) - interval(b[i]))**2 for i in range(len(a))]
#     norm_sq = sum(norm_sq_list)
#     denom = -2*interval(tau)**2
#     return imath.exp(norm_sq/denom)

# # Define interval GP Regression
# def GP_reg_intvl(X_train, x_star, Z, tau):
#     """
#     This function combines with GP_input_setup in order to do GP regression
#     Evaluates at points in R^n--single points, not collection
#     This version does interval arithmetic
    
#     Input:
#         X_train: an m by q matrix (training input)
#         x_star: a 1 by q matrix (where we evaluate the regression)
#         Z: m-vector, output of GP_input_setup
#         tau: parameter for squared exponential kernal
        
#     Output:
#         Y_star: a m by q matrix, output of Gaussian regression of x_star
#     """
#     # finding dimensions
#     m = X_train.shape[0]
    
#     # defining the covariance matrix K(X*,X)
#     KX_starX = []
#     for j in range(m):
#         xj = X_train[j,:]
#         KX_starX.append(sq_exp_intvl(x_star,xj,tau))
    
#     # getting result of regression, Y_star
#     # have to basically redefine linear algebra here by hand to implement interval arithmetic
#     Y_star = []
#     for j in range(Z.shape[1]): # for 2D case (like Leslie map) Z.shape[1] = 2
#         yj_list = [0]*m
#         for i in range(m):
#             yj_list[i] = KX_starX[i]*Z[i,j]
#         Y_star.append(sum(yj_list))
                    
#     return Y_star

