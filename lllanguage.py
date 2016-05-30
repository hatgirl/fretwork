import spacy
import numpy as np
import math
import codecs

nlp = spacy.load('en')

# Gram-Schmidt orthogonalization is a subroutine
def gram_schmidt( basis ):

   # remove all 0 vectors from the set
   basis = [vec for vec in basis if not np.array_equal(vec, nlp(u' ').vector)] 	
   
   # if underspecified, don't bother
   if (len(basis) < len(basis[0])):
      print "Small basis set, vector space is underspecified."
   else:
 	
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
   
   # we want enough vectors to span the word2vec vectorspace
   target_dim = len(nlp(u' ').vector)
   
   # remove all 0 vectors from basis set
   basis = [vec for vec in basis if not np.array_equal(vec, nlp(u' ').vector)] 	
   
   if (len(basis) < target_dim):
      print "Need additional text to yield the 300 linearly independent vectors \
      needed to span the word2vec vectorspace"
   
   else:
      # pick the first n vectors from the basis, where n = len(word2vec vectors)
      lin_ind_basis = [basis[i] for i in range(target_dim)]

      # add additional vectors to the linearly independent basis until we have a 
      # spanning set
      i = target_dim
      num_vectors = len(basis)
      num_lin_ind_vectors = len(gram_schmidt(lin_ind_basis))
      while (num_lin_ind_vectors < target_dim and i < num_vectors):
         
         new_span = len(gram_schmidt(lin_ind_basis+[basis[i]]))
         if ( new_span > num_lin_ind_vectors ):
            num_lin_ind_vectors = new_span
            lin_ind_basis.append(basis[i])
         i+=1

      # did we get enough or did we run out of the provided vectors first?
      if (num_lin_ind_vectors < target_dim):
         print "Need additional text to yield the 300 linearly independent vectors \
         needed to span the word2vec vectorspace"
      else:      
         delta = 0.75 
         ortho_basis = gram_schmidt(basis)
         dim = len(ortho_basis)

         i = 1
         while(i < dim):
            for j in range(i-1, -1, -1):
               mu_ij = np.dot(basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) 
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
               ortho_basis = gram_schmidt(basis)
               i = max(i-1, 1)

   return basis

x = np.array([1.0, 0.0, 0.0])
y = np.array([1.0, 1.0, 0.0])
z = np.array([1.0, 1.0, 1.0])
a = np.array([2.0, 2.0, 1.0])
print lll_reduction([x, y, z, a])

x = np.array([1, 1, 1])
y = np.array([-1, 0, 2])
z = np.array([3, 5, 6])

print lll_reduction([x, y, z])

text = codecs.open('test.txt', encoding='utf-8').read()
doc = nlp(text)
#
vectors = [word.repvec for word in doc]
print len(vectors)
print "vectors[0] = " + str(vectors[0])
ortho = gram_schmidt(vectors)
print len(ortho)
reduction = lll_reduction(vectors)
print len(reduction)

