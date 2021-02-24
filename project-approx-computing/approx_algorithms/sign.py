import numpy as np
import math
import numba
from numba import jit
#n = 4
#nby2 = (int)(n/2)
#p1      = np.ndarray(shape = (int(n / 2), n ), dtype = bool)
#g1      = np.ndarray(shape = (int(n / 2), n ), dtype = bool)
#sums1   = np.ndarray(shape = (int(n / 2), n ), dtype = bool)
#carrys1 = np.ndarray(shape = (int(n / 2), n ), dtype = bool)
#cin1    = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#e1      = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#s1      = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#c1      = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#z1      = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#lsb1    = np.ndarray(shape = (int(n / 2), 1), dtype = bool)
#am1     = np.ndarray(shape = (int(n / 2), 1), dtype = bool)

#prod1   = np.ndarray(shape = (2 * n, 1), dtype = bool)



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
    #print(b)

@jit
def ndecToBinary2(n, bits):
    n2 = (1 << (bits - 1)) + n
    b = [1]
    b += decToBinary2(n, bits - 1)
    return b
    #print(b)

def twos_complement(a, n):
      z     = 2 ** (n - 1) + a

      bin_a = list(map(int, list(bin(z)[2:].zfill(n - 1))))
      bin_a.insert(0, 1) 
      return bin_a  

@jit(nopython = True)
def booth_encoding(a, b, c):
   #print(numba.typeof(a))
   #print(numba.typeof(b))
   #print(numba.typeof(c))
   if (a,b,c) == (0, 0, 0):
       return (0, 0, 1)
   elif (a, b, c) == (0, 0, 1):
       return (0, 0, 0)
   elif (a, b, c) == (0, 1, 0):
       return (0, 0, 0)
   elif (a, b, c) == (0, 1, 1):
       return (1, 0, 0)
   elif (a, b, c) == (1, 0, 0):
       return (1, 1, 0)
   elif (a, b, c) == (1, 0, 1):
       return (0, 1, 0)
   elif (a, b, c) == (1, 1, 0):
       return (0, 1, 0)
   elif (a, b, c) == (1, 1, 1):
       return (0, 0, 1)
   #switcher = {
   #    (0, 0, 0) : (0, 0, 1),
   #    (0, 0, 1) : (0, 0, 0),
   #    (0, 1, 0) : (0, 0, 0),
   #    (0, 1, 1) : (1, 0, 0),
   #    (1, 0, 0) : (1, 1, 0),
   #    (1, 0, 1) : (0, 1, 0),
   #    (1, 1, 0) : (0, 1, 0),
   #    (1, 1, 1) : (0, 0, 1)


   #}
   #return switcher.get((a, b, c), "nothing")

@jit(nopython=True)
def be(am, am0, s, c, z):
   #if (not s and (not c) and (not z))
   
    if z==1:
       return 0;
    if s == 1:
       ret = am0
    else:
       ret = am
    if c== 1:
       ret = not ret

    return ret;

@jit
def lut_a1(prev, am, s, c, z):
    return (c and be(am, 0, s, c, z)) or (c and prev) or (be(am, 0, s, c, z) and prev) 

@jit
def lut_a2(prev, encoding, s, c):
    if s == 1:
        ret = not prev and encoding, not prev ^ encoding
    elif c == 1:
        ret = not encoding and (not prev), not prev ^ (not encoding)
    else: 
        ret = not prev and encoding, not prev ^ encoding
    return ret

@jit
def lut_a(prev, encoding):
    return prev and encoding, prev ^ encoding;
   
@jit
def lut_b(prev, msba, complement, zero):
    if msba and (not zero) and (not complement):
       ret = True
    elif complement and (not msba):
       ret = True
    else:
       ret = False
    return not ret and prev, not ret ^ prev

@jit
def lut_am(prev, a_lsb_1, a_lsb, s, c, z):
    carry = (c and (not s)) or s
    ret = be(a_lsb_1, a_lsb, s, c, z) ^ prev ^ carry
    return ret
@jit(nopython=True)
def approx_signed(a1, b1, n):
   nby2 = (int)(n/2)
   p      = [[0 for _ in range(n)] for _ in range(nby2)]
   g      = [[True for _ in range(n)] for _ in range(nby2)]
   sums   =  [[True for _ in range(n)] for _ in range(nby2)]
   carrys =  [[True for _ in range(n)] for _ in range(nby2)]
   cin    =  [True for _ in range(nby2)] 
   e      = [True for _ in range(nby2)] 
   s      = [0 for _ in range(nby2)]
   c      = [0 for _ in range(nby2)]
   z      = [0 for _ in range(nby2)]
   lsb    = [True for _ in range(nby2)]
   am     = [True for _ in range(nby2)] 

   prod   = [True for _ in range(n*2)]
   #print("inside approx signed")
   if a1 < 0:
       a= ndecToBinary2(a1, n)
      #a = twos_complement(a, n)
   else:
      a = decToBinary2(a1, n)
      #a = list(map(int, list(bin(a)[2:].zfill(n))))


   if b1 < 0:
       b = ndecToBinary2(b1, n)
          #b = twos_complement(b, n)
   else:
       b = decToBinary2(b1, n)
          #b = list(map(int, list(bin(b)[2:].zfill(n))))

   if n == 4:
      s[0], c[0], z[0] = booth_encoding(b[2], b[3], 0)
      s[1], c[1], z[1] = booth_encoding(b[0], b[1], b[2])
   if n == 8:
      (s[0], c[0], z[0]) = booth_encoding(b[6], b[7], 0)
      (s[1], c[1], z[1]) = booth_encoding(b[4], b[5], b[6])
      (s[2], c[2], z[2]) = booth_encoding(b[2], b[3], b[4])
      (s[3], c[3], z[3]) = booth_encoding(b[0], b[1], b[2])
   e[0] = b[2]
   e[1] = b[0]

   for i in range(nby2):
      if i == 0:
         for j in range(2, n):
             #print(a[j - 2], a[j - 1], s[i], c[i], z[i])
             g[i][j], p[i][j] = lut_a(0, be(a[j - 2], a[j - 1], s[i], c[i], z[i])) 

         lsb[i] = 0 
         cin[i] = 1
         am[i]  = lut_am(0, a[n - 2], a[n - 1], s[i], c[i], z[i])       
         g[i][0], p[i][0] = lut_b(1, a[0], c[0], z[0] ) 
         g[i][1], p[i][1] = lut_a2(True, be(p[i][2], a[0], s[0], c[0], z[0]), s[0], c[0])  
         if s[i] == 0:
             p[i][1] = p[i][2]
      
      elif (i != ((int)(n/2) - 1)):
         for j in range(2, n):
             g[i][j], p[i][j] = lut_a(sums[i - 1][j - 2], be(a[j - 2], a[j - 1], s[i], c[i], z[i]))  
         lsb[i] = 0
         cin[i] = 1
         am[i] = lut_am(sums[i - 1][n - 2], a[n - 2], a[n - 1], s[i], c[i], z[i])  
         g[i][0], p[i][0] = lut_b(carrys[i - 1][0], a[0], c[i], z[i])
         g[i][1], p[i][1] = lut_a(not(carrys[i - 1][0]), be(a[0], a[0], s[i], c[i], z[i]))

      else:
         for j in range(2, n):
             g[i][j], p[i][j] = lut_a(sums[i - 1][j - 2], be(a[j - 2], a[j - 1], s[i], c[i], z[i]))  
         lsb[i] = 0
         am[i] = lut_am(sums[i - 1][n - 2], a[n - 2], a[n - 1], s[i], c[i], z[i])  
         g[i][0], p[i][0] = lut_b(carrys[i - 1][0], a[0], c[i], z[i])
         g[i][1], p[i][1] = lut_a(not(carrys[i - 1][0]), be(a[0], a[0], s[i], c[i], z[i]))     
         cin[i]   = lut_a1(sums[i - 1][n - 1], a[n - 1], s[i], c[i], z[i])    

      for k in (range(n-1, -1, -1)):
         if k == n - 1:
           sums[i][k]   = (p[i][k] and (not cin[i])) or (not(p[i][k]) and cin[i])
           carrys[i][k] = g[i][k] or (cin[i] and p[i][k])
      
         else:
           sums[i][k]   = (p[i][k] and  not( carrys[i][k + 1])) or  (not(p[i][k]) and carrys[i][k + 1])
           carrys[i][k] = g[i][k] or (carrys[i][k + 1] and p[i][k])   
      #else:
      #     carrys[i][k] = g[i][k] or (carrys[i][k + 1] and p[i][k]) 
      #     if carrys[i - 1][0] != True:
      #          if carrys[i][k + 1] == True:
      #               carrys[i][k + 1] = False
      #     sums[i][k] = p[i][k] ^ carrys[i][k + 1]
          
   

   for i in range(nby2):
       prod[2*n - 1  - (2 * i)] = lsb[i]
       prod[2*n - 2  - (2 * i)] = am[i]

   for i in range(n):
       prod[i] = sums[nby2 - 1][i]

   total = 0
   for k in range(1, 2*n):
         total += (prod[k] * (2 ** (2*n - 1 - k)))
   total -= prod[0]* (2 ** (2 * n - 1))
#print(prod)
   return (total)

@jit(nopython = True)
def noting():
   a=[0,0,0,0,0]
   return not(a[0])

print(approx_signed(2,4,8))
n=4
err_distances = np.ndarray(shape = (((2 ** n)) ** 2, 1), dtype = np.float32)
r_err_distances = np.ndarray(shape = (((2 ** n)) ** 2, 1), dtype = np.float32)
total_error = 0
r_total_error = 0
square_error = 0
r_square_error = 0

for a_int in range(-(2**(int)(n/2)), (2**((int)(n/2)))):
  for b_int in range(-(2**(int)(n/2)), (2**((int)(n/2)))):
       print(a_int, b_int)
       #if a_int < 0:
       #  a= ndecToBinary2(a_int, n)
         #a = twos_complement(a_int, n)
       #else:
       #  a = decToBinary2(a_int, n)
         #a = list(map(int, list(bin(a_int)[2:].zfill(n))))

       #if b_int < 0:
       #  b = ndecToBinary2(b_int, n)
         #b = twos_complement(b_int, n)
       #else:
       #  b = decToBinary2(b, n)
         #b = list(map(int, list(bin(b_int)[2:].zfill(n))))

       approx_prod = approx_signed(a_int, b_int, n)
       exact_prod  = a_int*b_int
       print(a_int, b_int, approx_prod, exact_prod)
       err_distances[a_int * (2 ** n) + b_int] = abs(approx_prod - exact_prod)
       if(exact_prod != 0):
          r_err_distances[a_int * (2 ** n) + b_int] = (float)(float(err_distances[a_int * (2 ** n) + b_int] )/float(exact_prod))
       else:
          r_err_distances[a_int * (2 ** n) + b_int] = 0 
       total_error += err_distances[a_int * (2 ** n) + b_int]
       r_total_error += r_err_distances[a_int * (2 ** n) + b_int]
       square_error += (err_distances[a_int * (2 ** n) + b_int] ** 2)
       r_square_error += (r_err_distances[a_int * (2 ** n) + b_int] ** 2)
      
med = total_error/((2 ** n) ** 2)
ned = med/(2 ** (2 * n ))
mred = r_total_error/((2 ** n - 1) ** 2)

print("\nmean absolute error distance")
print(med)
#print(ned)
print("\nmean relative error distance")
print(mred)
#print(np.std(err_distances))

rms_aed = math.sqrt(square_error/((2 ** n) ** 2))
var_aed = np.var(err_distances)

rms_red = math.sqrt(r_square_error/((2 ** n) ** 2))
var_red = np.var(r_err_distances)

print("\nrms_aed")
print(rms_aed)
print("\nvar_aed")
print(var_aed)
print("\nrms_red")
print(rms_red)
print("\nvar_red")
print(var_red)


