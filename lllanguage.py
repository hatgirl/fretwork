import numpy as np
import math

# LLL algo

# Gram-Schmidt orthogonalization is a subroutine
def gram_schmidt( basis ):
	
   if (len(basis) < len(basis[0])):
      print "Small basis set, vector space is underspecified."
	
   dim = len(basis)
   ortho_basis = []
     
   for i in range(dim):
     	
      new_vector = basis[i]	
     	
      for j in range(i):
         # project the current vector along the next vector in
         # the basis
         proj = np.dot(basis[i], ortho_basis[j])
         mag_sq =  np.dot(ortho_basis[j], ortho_basis[j])
         mu = proj / mag_sq
         # substract out the component of the basis vector in 
         # the current vector 
         new_vector = new_vector - mu*ortho_basis[j]
     	
      # new_vector now has all basis vector components "removed"
      # so will be orthogonal (so long as it was not already a 
      # linear combination of basis vectors)
      # Do not normalize, add to new basis if not the 0 vector
      norm_factor = np.dot(new_vector, new_vector)
      if (norm_factor > 0):
         ortho_basis.append(new_vector)
   return ortho_basis

#x = np.array([2.0, 1.0, 0.0])
#y = np.array([0.0, 2.0, 1.0])
#z = np.array([0.0, 1.0, 0.25])
#print gram_schmidt([x, y, z])
#
#x = np.array([1.0, 0.0, 0.0])
#y = np.array([1.0, 1.0, 0.0])
#z = np.array([1.0, 1.0, 1.0])
#a = np.array([2.0, 5.2, 6.0])
#print gram_schmidt([x, y, z, a])			

def lll_reduction( basis ):
   
   delta = 0.75 
   ortho_basis = gram_schmidt(basis)
   dim = len(ortho_basis)

   i = 1
   while(i < dim):
      for j in range(i-1, -1, -1):
	 print "i = " + str(i)
         print "j = " + str(j)
         mu_ij = np.dot(basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) 
	 print mu_ij
      	 if (mu_ij < -0.5 or mu_ij > 0.5):
            basis[i] -= round(mu_ij)*basis[j]
            ortho_basis = gram_schmidt(basis)

      mu_ii_1 = np.dot(basis[i], ortho_basis[i-1]) / np.dot(ortho_basis[i-1], ortho_basis[i-1])
      if (np.dot(ortho_basis[i], ortho_basis[i]) >= (delta - mu_ii_1**2)*(np.dot(ortho_basis[i-1], ortho_basis[i-1]))):
         i = i+1
      else:
         t = basis[i]
         basis[i] = basis[i-1]
         basis[i-1] = t
         ortho_basis = grad_schmidt(basis)
         i = max(i-1, 1)

   return basis

x = np.array([1.0, 0.0, 0.0])
y = np.array([1.0, 1.0, 0.0])
z = np.array([1.0, 1.0, 1.0])
print lll_reduction([x, y, z])


