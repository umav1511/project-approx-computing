import numpy as np
import math
from numba import jit
approx = 5


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
def approx_reduce(a, n, approx_col):
  stop = 0
  level = 0
  max_col_height = n
  #new = np.zeros((max_col_height, 2*n - 1)).astype(int)
  #row_counter1 = np.zeros((2*n - 1, 1)).astype(int)
  #row_counter2 = np.zeros((2*n - 1, 1)).astype(int)
  
  new =  [[0 for _ in range(2*n-1)] for _ in range(4)]

  #new[0][14] = a[0][14]
  #new[0][13] = a[0][13] or a[1][13]
  #new[0][12], new[1][12] = approx_3_2(a[0][12], a[1][12], a[2][12])
  #new[0][11], new[1][11] = approx_4_2_T2(a[0][11], a[1][11], a[2][11], a[3][11])
  new[0][10], new[1][10] = approx_4_2_T2(a[0][10], a[1][10], a[2][10], a[3][10])
  new[2][10] = a[4][10]
  
  #9
  new[0][9], new[1][9] = approx_4_2_T2(a[0][9], a[1][9], a[2][9], a[3][9])
  new[2][9] = a[4][9] or a[5][9]

  #8
  new[0][8], new[1][8] = approx_4_2_T2(a[0][8], a[1][8], a[2][8], a[3][8])
  new[2][8], new[3][8] = approx_3_2(a[4][8], a[5][8], a[6][8])

  #7
  new[0][7], new[1][7] = approx_4_2_T2(a[0][7], a[1][7], a[2][7], a[3][7])
  new[2][7], new[3][7] = approx_4_2_T2(a[4][7], a[5][7], a[6][7], a[7][7])

  #6
  new[0][6], new[1][6] = approx_4_2_T2(a[1][6], a[2][6], a[3][6], a[4][6])
  new[2][6], new[3][6] = approx_3_2(a[5][6], a[6][6], a[7][6])

  #5
  new[0][5], new[1][5] = approx_4_2_T2(a[2][5], a[3][5], a[4][5], a[4][5])
  new[2][5] = a[6][5] or a[7][5]
 
  #print(new)
  new2 = [[0 for _ in range(2*n-1)] for _ in range(2)]  
  new2[0][14] = a[0][14]
  new2[0][13] = a[0][13] or a[1][13]
  new2[0][12], new2[1][12] = approx_3_2(a[0][12], a[1][12], a[2][12])
  new2[0][11], new2[1][11] = approx_4_2_T2(a[0][11], a[1][11], a[2][11], a[3][11])

  new2[0][10], new2[1][10] = approx_3_2(new[0][10], new[1][10], new[2][10])
  new2[0][9], new2[1][9] = approx_3_2(new[0][9], new[1][9], new[2][9])
  new2[0][8], new2[1][7] = approx_4_2_T1(new[0][8], new[1][8], new[2][8], new[3][8])
  new2[0][7], new2[1][6] = approx_4_2_T1(new[0][7], new[1][7], new[2][7], new[3][7])
  new2[0][6], new2[1][5] = approx_4_2_T1(new[0][6], new[1][6], new[2][6], new[3][6])
  new2[0][5], new2[1][4] = approx_4_2_T1(new[0][5], new[1][5], new[2][5], new[3][5])

  sum0=0
  for i in range(2*n-1):
     sum0 += ( new2[0][i] * (1 << (14 - i))) + (new2[1][i] * (1 << (14 - i))) 
  #print(new2)
  return sum0;    
   
@jit(nopython=True)
def exact_reduce(a, n, approx_level):
   exact_sum = 0
   for row in range(3, n):
     for i in range(row - 2):
       exact_sum += a[row][4 - i] * (1 << (10 + i))
   return exact_sum 
    
    
#change parameters here        
#n = 8




@jit(nopython=True)
def compress_top(a,b,n): 
    #pp = np.zeros((n,2 * n - 1)).astype(int)
    pp = [[0 for _ in range(2*n-1)] for _ in range(n)]
    a = decToBinary2(a, n)
    b = decToBinary2(b, n)
    for j in range(n):
      if a[n - 1 - j] == 1:   
        for i in range(n):
          
          pp[j][n - 1 - j + i] = b[i]
        
    #first = pp[:, 0 : n-1]
    #second = pp[:, n - 1 : 2 * n]
    #print("first")
    #print(first)
    #first_flip = np.flip(first, 0)
    #print("first flip")
    #print(first_flip)
    #pp = np.concatenate((first_flip,second), axis = 1)
    #pp = pp.astype(int)


    #psum1 = np.zeros((2 * n - 1,1)).astype(int)
    #psum2 = np.zeros((2 * n - 1,1)).astype(int)

    #print("approx_reduce")
    #print(approx_reduce)
    approx_sum = approx_reduce(pp, n, approx)
    #print(approx_sum)
    #print(approx_sum)
    #sum1 = 0
    #sum2 = 0
    #for i in range(2*n - 1):
    #   sum1 += approx_sum[0][i] * 2**(2 * n - 2 - i)
    #   sum2 += approx_sum[1][i] * 2**(2 * n - 2 - i)

    exact_sum = exact_reduce(pp, n, approx)
    approx_prod = approx_sum + exact_sum
    return approx_prod

n = 8
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
    err_distances[a* twopnby2 + b] = abs(a*b - compress_top(a, b, n))
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

