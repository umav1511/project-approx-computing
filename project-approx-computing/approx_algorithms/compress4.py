import numpy as np
import math
from numba import jit

@jit(nopython=True)
def decToBinary2(n, bits): 
      
    # Size of an integer is 
    # assumed to be 32 bits
    b = [0] 
    for i in range(bits-1, -1, -1):  
        k = n >> i; 
        b+=[k & 1]
        #b.append(k & 1)
    b.pop(0)
    return b

@jit(nopython=True)
def approx_3_2(a, b, c):
  return (int(a or b),int(c));

@jit(nopython=True)
def approx_4_2_T2(a, b, c, d):
  return [int(a or b), int(c or d )];

@jit(nopython=True)
def approx_4_2_T1(a, b, c, d):
  xor3=( not (int (a) ) and not(int(b)) and int(c)) or (not(int(a)) and int(b) and not(int(c))) or (int(a) and not(int(b)) and not(int(c))) or (int(a) and int(b) and int(c))
  return (int(int(((int)(xor3) and not(int(d))) or (not(int(xor3)) and int(d)) ) + (a and b and c and d)), int(((a or b) and (c or d)) or (c and d) or (a and b)) );

@jit(nopython=True)
def approx_reduce(pp, n, approx_col):
  max_col_height = 2
  new =[[0 for _ in range(2*n-1)] for _ in range(max_col_height)]

  new[0][6] = pp[0][6]
  new[0][5] = (int)(pp[0][5] or pp[1][5])
  new[0][4], new[1][4] = approx_3_2(pp[0][4], pp[1][4], pp[2][4])
  new[0][3], new[1][2] = approx_4_2_T1(pp[0][3], pp[1][3], pp[2][3], pp[3][3])

  sum0 = 0
  for i in range(2*n - 1):
     sum0+= (new[0][i] * (1 << (6 - i)) + new[1][i] * (1 << (6 - i))) 
  return sum0;    
   
@jit(nopython=True)     
def exact_reduce(a, n, approx_level):
   exact_sum = 0
   for row in range(n):
     for i in range(row):
       exact_sum += a[row][2 - i] * (1 << (4 + i))
   return exact_sum 
    
@jit(nopython=True)
def compress_top4(a_int,b_int, n):
    approx = 3
    #pp = np.zeros((n,2 * n - 1)).astype(int)
    pp = [[0 for _ in range(2*n-1)] for _ in range(n)]
    #a = bin(a_int)[2:].zfill(n)
    #b = bin(b_int)[2:].zfill(n)
    a = decToBinary2(a_int, n)
    b = decToBinary2(b_int, n)
    for j in range(n):
      if a[n - 1 - j] == 1:  
        for i in range(n):
          pp[j][n - 1 - j + i] = b[i]
    #first = pp[:, 0 : n-1]
    #second = pp[:, n - 1 : 2 * n]
    #first_flip = np.flip(first, 0)
    #pp = np.concatenate((first_flip,second), axis = 1)
    #pp = pp.astype(int)
    approx_sum = approx_reduce(pp, n, approx)
    exact_sum = exact_reduce(pp, n, approx)
    return approx_sum + exact_sum

#print(compress_top4(11,7,4))
n = 4

twopnby2 = (int)(((2 ** n)/2) + 1)
#twopnby2 = 2**n
total_error = 0
r_total_error =0
square_error=0
r_square_error=0
err_distances = np.ndarray(shape = ((twopnby2) ** 2, 1), dtype = np.float32)
r_err_distances = np.ndarray(shape = ((twopnby2) ** 2, 1), dtype = np.float32)
for a in range(twopnby2):
  for b in range(twopnby2):
    #print(a, b)
    err_distances[a* twopnby2 + b] = abs(a*b - compress_top4(a, b, n))
    if float(a*b)!=0:
        r_err_distances[a* (twopnby2) + b] = (float)(float(err_distances[a* (twopnby2) + b])/float(a*b))
    else:
        r_err_distances[a* (twopnby2) + b] = 0 
    total_error += err_distances[a * (twopnby2) + b]
    r_total_error += r_err_distances[a * (twopnby2) + b]
    square_error += ( err_distances[a * (twopnby2) + b] ** 2)
    r_square_error += ( r_err_distances[a * (twopnby2) + b] ** 2)

med = total_error/((twopnby2) ** 2)
rmed=  r_total_error/((twopnby2) ** 2)
print("med")

#print(ned)
print("rmed")
print(rmed)


rms_aed = math.sqrt(square_error/((twopnby2) ** 2))
var_aed = np.var(err_distances)

rms_red = math.sqrt(r_square_error/((twopnby2) ** 2))
var_red = np.var(r_err_distances)

print("\nrms_aed")
print(rms_aed)
print("\nvar_aed")
print(var_aed)
print("\nrms_red")
print(rms_red)
print("\nvar_red")
print(var_red)

print(med[0], rmed[0], rms_aed, rms_red, var_aed, var_red)



