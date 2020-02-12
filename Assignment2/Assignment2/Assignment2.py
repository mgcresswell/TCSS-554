import numpy as np


#Preform dot product of A and r
#until convergence or n = 1000
def Converge(A, r):
    i = 1
    n = 1000
    while i <= n:    
        newR = A.dot(r)   
        newR = np.round(newR, decimals=5)    
        print("r =", newR) 
        if (hasConverged(r, newR)):
            print("Number of iterations to converge = ", i)
            print("\n")
            break
        else:
            r = newR
            i = i+1
            
#Test for convergence
#subtract new r from old r 
#if each element is less than e
def hasConverged(oldR, newR):
    e = 0.00001
    sub = np.absolute(np.subtract(newR, oldR))
    result = sub < e
    return  np.all(result)
    

np.set_printoptions(precision=5, suppress=True)
#create M array 
M = np.array([
    [0,0,0,1,0],
    [(1/2),0,0,0,0],
    [0,(1/2),0,0,0],
    [(1/2),(1/2),0,0,1],
    [0,0,0,0,0]
    ])
print("M =", M)
print("\n")


#create r
r =  np.array([(1/5),(1/5),(1/5),(1/5),(1/5)])
print("r =", r)
print("\n")

#create teleportation matrix
teleportMatrix = np.array([
    [(1/5),(1/5),(1/5),(1/5),(1/5)],
    [(1/5),(1/5),(1/5),(1/5),(1/5)],
    [(1/5),(1/5),(1/5),(1/5),(1/5)],
    [(1/5),(1/5),(1/5),(1/5),(1/5)],
    [(1/5),(1/5),(1/5),(1/5),(1/5)]
    ])
print("Teleportation Matrix =", teleportMatrix)
print("\n")

#Create A using beta = .85
A = (M *(.85)) + (teleportMatrix*(.15)); 
print("A =", A)
print("\n")


print("Converging M.......")
Converge(M,r)
print("Converging A.......")
Converge(A,r)

