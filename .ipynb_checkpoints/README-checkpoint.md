# GaussianProcessIntervalArithmetic
Implementation of the Gaussian process regression algorithm with interval arithmetic

## What is Interval Arithmetic?

When computers perform a computation, they frequently return inexact results due to rounding.  In most cases these errors are very small and unimportant, but sometimes they can compound in order to give highly inaccurate results.  Interval arithmetic gives rigorous bounds on these errors by outputting an interval which is guaranteed to contain the true output of the function.  

For instance, if we have a real-valued function $f$, when we code it into our computer and ask for the value of $f(x)$, the computer will return some value $y$.  However, $y$ is not the true value of $f(x)$, it is an approximation after some number of rounding operations.  If we use interval arithmetic and ask for the value of $f(x)$, our computer will instead return an interval $[a,b]$, with a rigorous guarantee that the true value of $f(x)$ satisfies $a \leq f(x) \leq b$. 

For more information, check out 
http://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

# Gaussian Process Regression with Interval Arithmetic

One current goal of the CMGDB group is to take in input data, use a Gaussian Process (GP) to define a map with that data (the map is the mean of the process), use that map to define a multivalued map which tracks rigorous bounds on where the image of the map lands in a grid, and then run that multivalued map through the CMGDB program in order to determine the Morse graph.  Hopefully that Morse graph is consistant with the true dynamics with high probability; that is a theoretical question that will take some work.

However, at the moment, we are having a much more practical problem; the software used in order to get the rigorous multivalued map that we desire (PyInterval) does not play well with the software used in order to get the GP generated map (sklearn GaussianProcessRegressor).  This code is a first attempt at solving that problem.  This should be doable, because the output of the Gaussian Process regression (the mean function of the process) is an explicit (and relatively simple) formula: $$ \mu^* = K(X_*,X)(K(X,X)+\sigma^2 I)^{-1}\hat{y} $$
Letting $m$ be the number of training points and $n$ be the dimension of the input vectors, we have that $X \in R^{m\times n}$, $K(X,X), I \in R^{m \times m}$, and $K(X,X)_{i,j} = k(x^{(i)},x^{(j)})$ where $k$ is the kernal.  Also $\hat{y} \in R^{m \times q}$ is the training output and $\sigma^2 \in R$ is the variance of the noise in the process. Note that the GP treats each of the $q$ dimensions of $y$ as independent.  

Note that, for fixed input data, 
$$ Z:= (K(X,X)+\sigma^2 I)^{-1}\hat{y}$$ is just an $m\times q$ matrix (real-valued).  Then the output of the GP (the mean) is actually a pretty simple function where, taking the view that $X_* = x_*$ (so we just write the output of each point separately, so that we can view this as a function), we get that
$$ \mu(x_*) = K(x_*,X)Z$$
where
$$K(x_*,X) = [k(x_*,x^{(1)}),\cdots,k(x_*,x^{(m)})].$$

Assuming that we are using the common squared exponential kernal, the entries of the covariance matrix $K$ involve only exponentials, powers, division, and a Euclidean norm (so subtraction, addition, square root).  All of these functions are implemented in PyInterval.

Our interest is in performing interval arithmetic with respect to the test data $X_*$, so we we basically need to implement two steps. 

First, we compute $Z$ for a given input data.  We will do this using Cholesky factorization, following algorithm 2.1 from "Gaussian Processes for Machine Learning".  Note that this algorithm is what the sklearn function uses as well.  Since this part of the equation does not depend on the test data $X_*$, we do not need to follow interval arithmetic here (which is lucky, as the matrix inversion would probably be ridiculously slow).  

The second step (and where we break from the sklearn code) is to solve for the vector $K(x_*,X)$ using interval arithmetic, and multiply that vector by $Z$.

I also want to note that this code is a somewhat limited product; it is not be nearly as robust as the sklearn function; in particular, it is limited to only the use of the squared exponential kernal. It also only tracks the mean of the process, and not the variance.
