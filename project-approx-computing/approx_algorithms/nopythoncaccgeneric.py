import numpy as np;
import math;
from numba import jit

@jit
def fa(a, b, c):
  #return int(a) ^ int(b) ^ int(c);
  return ( not (int (a) ) and not(int(b)) and int(c)) or (not(int(a)) and int(b) and not(int(c))) or (int(a) and not(int(b)) and not(int(c))) or (int(a) and int(b) and int(c)) 

@jit
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

@jit
def approx_4_4(a, b):
  if (a == 5 and b == 15) or (a == 6 and b == 7) or (a == 6 and b == 15) or (a == 7 and b == 15) or (a == 13 and b == 13) or (a == 15 and b == 5):
    return a*b - 8;
  else:
    return a*b;

@jit(nopython=True)    
def approx_acc(a, b, n):
 
  n2 = 2 ** (n/2)
  a2 = a%n2
  a1 = int(a/n2)
  b2 = b%n2
  b1 = int(b/n2)
  

  if n!=4:
     LL = int(approx_acc(a2, b2, n/2))
     HL = int(approx_acc(a1, b2, n/2) * n2)
     LH = int(approx_acc(a2, b1, n/2) * n2)
     HH = int(approx_acc(a1, b1, n/2) * (n2 ** 2))
  
  else:
     if (a == 5 and b == 15) or (a == 6 and b == 7) or (a == 6 and b == 15) or (a == 7 and b == 15) or (a == 13 and b == 13) or (a == 15 and b == 5):
        return a*b - 8;
     else:
        return a*b;
  
  return(LL + HL + LH + HH);
  
@jit(nopython = True)
def approx_app(a, b, n):

  n2 = 2 ** (int)(n/2)
  a2 = a%n2
  a1 = int(a/n2)
  b2 = b%n2
  b1 = int(b/n2)
  
  nby2 = int(n/2)


  if n!=4:
     #print((int(approx_app(a2, b2, nby2))))
     #print(bin(int(approx_app(a2, b2, nby2))))
     #print(bin(int(approx_app(a2, b2, nby2)))[2:])
     #print(bin(int(approx_app(a2, b2, nby2)))[2:].zfill(n))

     #LL1 = bin(int(approx_app(a2, b2, nby2)))[2:].zfill(n)
     #HL1 = bin(int(approx_app(a1, b2, nby2) * n2))[2:].zfill(n + nby2)
     #LH1 = bin(int(approx_app(a2, b1, nby2) * n2))[2:].zfill(n + nby2)
     #HH1 = bin(int(approx_app(a1, b1, nby2) * (n2 ** 2)))[2:].zfill(2 * n)
   
     LL = decToBinary2(approx_app(a2, b2, nby2), n)
     HL = decToBinary2((approx_app(a1, b2, nby2) * n2), (n + nby2))
     LH = decToBinary2((approx_app(a2, b1, nby2) * n2), (n + nby2))
     HH = decToBinary2((approx_app(a1, b1, nby2) * (n2 ** 2)), (2*n))  
     #result = np.zeros((2 * n, 1)).astype(int)
     result = [0] * (2*n)
     sum1 = 0
  
     for i in range(nby2):
        result[2*n - 1 - i] = LL[n - 1 - i]
        sum1 += result[2 * n - 1 - i] * 2**(i)
     
     for i in range(nby2, n):

        result[2*n - 1 - i] = fa(LL[n - 1 - i], HL[n + (nby2) - 1 - i], LH[n + (nby2) - 1 - i])
        sum1 += result[2 * n - 1 - i] * 2**( i)
     
     for i in range(n, n + nby2):
   
        result[2 * n  - 1 - i] = fa(HL[n + nby2 - 1 - i], LH[n + nby2 - 1 - i], HH[2*n - 1 - i])
        sum1 += result[2*n - 1 - i] * 2**(i)
     
     for i in range(n + nby2, 2 * n):
        result[2*n - 1 - i] = HH[2*n - 1 - i]
        sum1 += result[2*n - 1 - i] * 2**(i)
  

  else:
     if (a == 5 and b == 15) or (a == 6 and b == 7) or (a == 6 and b == 15) or (a == 7 and b == 15) or (a == 13 and b == 13) or (a == 15 and b == 5):
        return int(a*b - 8);
     else:
        return int(a*b);
  
  return sum1;
  
print(fa(1,1,0))

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
    err_distances[a* twopnby2 + b] = abs(a*b - approx_app(a, b, n))
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


