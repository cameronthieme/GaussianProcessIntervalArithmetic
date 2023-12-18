# GaussianProcessIntervalArithmetic
Implementation of Gaussian process regression with interval arithmetic.  We use algorithm 2.1 from "Gaussian Processes for Machine Learning" by Rasmussen and Williams.  Interval arithmetic relies on the PyInterval package.  

## What is a Gaussian process?

A Gaussian process (GP) is a type of machine learning model.  A good reference is "Gaussian Processes for Machine Learning", which can be found here:

http://gaussianprocess.org/gpml/

## What is Interval Arithmetic?

When computers perform a computation, they frequently return inexact results due to rounding.  In most cases these errors are very small and unimportant, but sometimes they can compound in order to give highly inaccurate results.  Interval arithmetic gives rigorous bounds on these errors by outputting an interval which is guaranteed to contain the true output of the function.  

For instance, suppose we have a real-valued function $f$. When we code it into our computer and ask for the value of $f(x)$, the computer will return some value $y$.  However, $y$ is not the true value of $f(x)$, it is an approximation after some number of rounding operations.  If we use interval arithmetic and ask for the value of $f(x)$, our computer will instead return an interval $[a,b]$, with a rigorous guarantee that $a \leq f(x) \leq b$. 

For more information, check out 
http://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

# GPIA for CMGDB

CMGDB is a software tool that uses Conley-Morse theory in order to analyze dynamical systems.  It gives rigorous information about the long-run (asymptotic) behavior of the system.  There's a lot to unpack there, and for more details check out
https://www.math.fau.edu/files/kalies_papers/dsagdms.pdf

From a practical perspective, CMGDB takes as input a function $f$ which represents some discrete dynamical system.  It outputs rigorous information about the behavior of the system.  

In order to guarantee that the output of the software is a rigorous description of the dynamics of $f$, we need to use interval arithmetic.  The CMGDB group is currently interested in analyzing the accuracy of using machine learning surrogate models--especially Gaussian processes--to learn the behavior of a dynamical system.  Hence we are interested in having an implementation of Gaussian process regression which performs interval arithmetic in order to track errors.  

## Methodology

Assuming a prior mean function of zero, the Gaussian Process regression function (the mean function of the posterior process) is an explicit (and relatively simple) formula: $$ \mu^* = K(X_*,X)(K(X,X)+\sigma^2 I)^{-1}\hat{y} $$
Letting $m$ be the number of training points and $n$ be the dimension of the input vectors, we have that $X \in R^{m\times n}$ is the training matrix, $K(X,X) \in R^{m \times m}$ is the covariance matrix, and $K(X,X)_{i,j} = k(x^{(i)},x^{(j)})$ where $k$ is the kernal.  Also $\hat{y} \in R^{m \times q}$ is the training output and $\sigma^2 \in R$ is the variance of the noise in the process. We remark that the GP treats each of the $q$ dimensions of $y$ as independent.  

Note that, for fixed input data, 

$$Z:= (K(X,X)+\sigma^2 I)^{-1}\hat{y}$$ 

is just an $m\times q$ matrix (real-valued).  Then the output of the GP (the mean) is actually a pretty simple function where, taking the view that $X_* = x_*$ (so we just write the output of each point separately, so that we can view this as a function), we get that
$$ \mu(x_*) = K(x_*,X)Z$$
where
$$K(x_*,X) = [k(x_*,x^{(1)}),\cdots,k(x_*,x^{(m)})].$$

Assuming that we are using the common squared exponential kernal, the entries of the covariance matrix $K$ involve only exponentials, powers, division, and a Euclidean norm (so subtraction, addition, square root).  All of these functions are implemented in PyInterval.

Our interest is in performing interval arithmetic with respect to the test data $X_*$, so we we basically need to implement two steps. 

First, we compute $Z$ for a given input data.  We will do this using Cholesky factorization, following Algorithm 2.1 from "Gaussian Processes for Machine Learning".  Note that this algorithm is what the sklearn function uses as well.  Since this part of the equation does not depend on the test data $X_*$, we do not need to follow interval arithmetic here (which is lucky, as the matrix inversion would probably be ridiculously slow).  

The second step (and where we break from the sklearn code) is to solve for the vector $K(x_*,X)$ using interval arithmetic, and multiply that vector by $Z$. Note that this also follows Algorithm 2.1.

## Extra Remarks

I also want to note that this code is a somewhat limited product; it is not be nearly as robust as the sklearn function; in particular, it is limited to only the use of the squared exponential kernal. It also only tracks the mean of the process, and not the variance. 

But it does perform interval arithmetic, and give us rigorously bounded output :)
