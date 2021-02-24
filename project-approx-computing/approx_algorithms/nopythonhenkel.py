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
def top(a, x, n):
  x1 = (int)(x % (2 ** (int)(n / 2)))
  a1 = (int)(a % (2 ** (int)(n / 2)))
  x2 = (int)(x / (2 ** (int)(n / 2)))
  a2 = (int)(a / (2 ** (int)(n / 2))) 
  #print(x1, a1, x2, a2)
  LL = henkel(x1, a1, (int)(n / 2))
  HL = henkel(x2, a1, (int)(n / 2)) * (2 ** (int)(n / 2))
  LH = henkel(x1, a2, (int)(n / 2)) * (2 ** (int)(n / 2))
  HH = (x2 * a2) * (2 ** n)
  #print(HH, LL, HL, LH)
  return(HH + LL + HL + LH)


@jit(nopython=True)
def henkel(a, x, n):
  b = n
  c = 0
  q = 2 * b - 1
  r = 2 * b - 1
  k = int(b / 2)
  d = b - k

  def fa(a, b, c):
    return ( not (int (a) ) and not(int(b)) and int(c)) or (not(int(a)) and int(b) and not(int(c))) or (int(a) and not(int(b)) and not(int(c))) or (int(a) and int(b) and int(c)), a and b or b and c or c and a;
  #print(a, x)
  #a1 = list(map(int, list(bin((int)(a))[2:].zfill(n))))
  #x1 = list(map(int, list(bin((int)(x))[2:].zfill(n))))
  #print(a1)
  #print(x1)
  a = decToBinary2(a, n)
  x = decToBinary2(x, n)
  #print(a)
  #print(x)

  #p = np.zeros((2 * b), dtype = int)
  p = [0] * (2 * b)

  #for i in reversed(range( b - k, b - 1 + 1)):
  #   for j in reversed(range(d, b - 1 + 1)):
  #      p[q], c = fa(p[q], a[j] and x[i], c)
  #      q = q - 1
  #   q = r - 1
  #   r = r - 1
  #   d = d + 1
  #   c = 0

  for i in range( b - 1, b - k - 1, -1):
     for j in range(b - 1, d - 1, -1):
        p[q], c = fa(p[q], a[j] and x[i], c)
        q = q - 1
     q = r - 1
     r = r - 1
     d = d + 1
     c = 0

  #for r in reversed(range(2 * b - b, (2 * b - (int)(b / 2) - 1) + 1 )):
  #   p[r] = 1

  for r in range((2 * b - (int)(b / 2) - 1), 2*b - b -1, -1):
      p[r] = 1
  #c = 1
  #p[r] = c
  #q = 2*b - k - 1
  #r = q
  #d = b - k - 1
  #c = 0
  
  #carry predictor here
  for i in range(0, b):
     c = c or (a[i] and x[b - 1 - i])
     if c:
        break  


  q = 2 * b - b - 1
  d = 2 * b - b - 1

  for i in range(0, 2 * b - b - 1):
     #for j in reversed(range(0, d)):
     for j in (range(d-1, -1, -1)):
         p[q], c = fa(p[q], a[i] and x[j], c)
         if j == 0 and c == 1:
             while q!=0 and c==1:  
                p[q - 1], c = fa(p[q - 1], c , 0)
                q = q - 1
         q = q - 1
     q = 2 * b - b - 1
     d = d - 1
     c = 0
  
  sum1 = 0
  #print(p)
  for i in range(2 * b):
     sum1 += p[i] * 2 ** (2 * b - 1 - i)
  #print(sum1)
  return sum1



n = 8
twopnby2 = (int)(((2 ** n)/2) + 1)
#twopnby2 = 2**n
total_error = 0
r_total_error =0
square_error=0
r_square_error=0
err_distances = np.ndarray(shape = ((twopnby2) ** 2, 1), dtype = np.float32)
r_err_distances = np.ndarray(shape = ((twopnby2) ** 2, 1), dtype = np.float32)
print(top(1,3,4))
for a in range(twopnby2):
  for b in range(twopnby2):
    #print(a, b)
    err_distances[a* twopnby2 + b] = abs(a*b - top(a, b, n))
    print(a,b,a*b, top(a,b,n))
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

