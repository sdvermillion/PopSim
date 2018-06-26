import numpy as np
from scipy.stats import norm
from scipy.stats import invwishart
from scipy.stats import bernoulli

class Toolbox:
	def dim_check(self, X):
		"""
		Expands one dimensional arrays to two dimensional
		
		Parameters
		----------
		X : array-like, ndims <= 2
			Array
			
		Returns
		-------
		X : array-like, ndims = 2
			Two dimensional array
		"""
		if X.ndim <= 1:
			return np.expand_dims(X, axis = 0)
		else:
			return X
	
	def generate_seed(self, n_features, size):
		"""
		Generates uncorrelated dataset from standard multivariate Gaussian distribution
		
		Parameters
		----------
		size : int
			The number of samples to create
			
		Returns
		-------
		self : object
			Returns self.
		"""
		# Initialize standard multivariate Gaussian distribution parameters
		mu = np.zeros(n_features)	# All means are zero
		sigma = np.eye(n_features)	# All variances are one and covariances are zero
		
		# Generate Sample of n Normal Variables w/ 0 Mean and 1 std
		seed = np.random.multivariate_normal(mu, sigma, size)
		
		# Output
		return self._removeCorr(seed)
	
	def time_fit(self, X, rho):
		"""
		Transforms data set X by a single time step using supplied autocorrelations
		
		Parameters
		----------
		X : {array-like}, shape = [n_samples, n_features]
			Uncorrelated dataset generated from standard multivariate Gaussian distribution
		rho : {array-like}, shape = [n_features]
			Array of autocorrelations such that rho[i] is the autocorrelation between
			feature i at time step t and t + 1.
			
		Returns
		-------
		X_step : {array-like}, shape = [n_samples, n_features]
			Returns uncorrelated data array.
		"""
		# Transform each individual's detector profile one detector at a time
		X_step = np.zeros(X.shape)
		n = X.shape[0]
		for i in range(X.shape[1]):
			sigma = 1 - rho[i]**2
			#X_step[:,i] = rho[i]*X[:,i] + np.random.normal(scale = sigma, size = n)
			X_step[:,i] = np.random.multivariate_normal(rho[i]*X[:,i], sigma*np.eye(X.shape[0]))
				
		# Remove residual correlation using Cholesky transform
		return self._removeCorr(X_step)
	
	def corr_fit(self, X, sigma):
		# Ensure correlation matrix is positive definite
		sigma = self._nearestPD(sigma)
		
		# Cholesky transformation of correlation matrix
		L = np.linalg.cholesky(sigma)

		# Apply correlation to uncorrelated data
		X_corr = np.matmul(L, X.T)
		return X_corr.T
	
	def copula_fit(self, X):
		u = np.zeros(X.shape)
		for i in range(X.shape[1]):
			# Generate Percentile
			u[:,i] = np.squeeze(norm.cdf(X[:,i]))
		return u
	
	def random_corr_matrix(self, df, scale):
		q = invwishart.rvs(df, scale)
		p = np.diag(1/np.sqrt(np.diag(q)))
		return np.matmul(np.matmul(p,q),p)
	
	def bernoulli_fit(self, filename):
		data = np.genfromtxt(filename, skip_header = 1, delimiter = ",", missing_values = '--', filling_values = 0.0)
		p = data[1,2]/np.sum(data[:,2])
		return bernoulli(p)
	
	def emperical_fit(self, filename):
			data = np.genfromtxt(filename, skip_header = 1, delimiter = ",", missing_values = '--', filling_values = 0.0)
			p = data[1,2]/np.sum(data[:,2])
			return bernoulli(p)
	
	def _removeCorr(self, A):
		corr = np.cov(A, rowvar = False)
		L = np.linalg.cholesky(self._nearestPD(corr))
		Li = np.linalg.inv(L)
		X_uncorr = np.matmul(Li, A.T)
		
		# Output
		return X_uncorr.T
	
	def _nearestPD(self, A):
		"""
		Find the nearest positive-definite matrix to input
	
		A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
		credits [2].
	
		[1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
	
		[2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
		matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
		"""
	
		B = (A + A.T) / 2
		_, s, V = np.linalg.svd(B)
	
		H = np.dot(V.T, np.dot(np.diag(s), V))
	
		A2 = (B + H) / 2
	
		A3 = (A2 + A2.T) / 2
	
		if self._isPD(A3):
			return A3
	
		spacing = np.spacing(np.linalg.norm(A))
		# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
		# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
		# Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
		# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
		# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
		# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
		# `spacing` will, for Gaussian random matrixes of small dimension, be on
		# othe order of 1e-16. In practice, both ways converge, as the unit test
		# below suggests.
		I = np.eye(A.shape[0])
		k = 1
		while not self._isPD(A3):
			mineig = np.min(np.real(np.linalg.eigvals(A3)))
			A3 += I * (-mineig * k**2 + spacing)
			k += 1
	
		return A3
	
	def _isPD(self, B):
		"""Returns true when input is positive-definite, via Cholesky"""
		try:
			_ = np.linalg.cholesky(B)
			return True
		except np.linalg.LinAlgError:
			return False