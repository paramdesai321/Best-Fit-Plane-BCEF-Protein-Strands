import numpy as np
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
#from parsingcoord import Parsing
from CA_C_N_parsing import x_coord,y_coord,z_coord
import time
from Strands import B_X,B_Y,B_Z,C_X,C_Y,C_Z,E_X,E_Y,E_Z,F_X,F_Y,F_Z,rest_X,rest_Y,rest_Z

# Retrieve the coordinates
st= time.time()
x = x_coord 
y = y_coord 
z = z_coord

x = np.array(x)

y = np.array(y)
z = np.array(z)


# Plot the initial data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rest_X, rest_Y, rest_Z, label='Data',color='m')  # Plot the original data points
ax.scatter(B_X,B_Y,B_Z,label="B",color='g')
ax.scatter(C_X,C_Y,C_Z,label="C",color='g')
ax.scatter(F_X,F_Y,F_Z,label="F",color='g')
ax.scatter(E_X,E_Y,E_Z,label="E",color='g')

#w1 = np.random.rand(1)
#w2 = np.random.rand(1)

def Init_plane():
    X1 = x[:100]
    X1 = np.mean(X1)
    X2 = x[100:]
    X2 = np.mean(X2)
    Y1=  y[:100]
    Y1 = np.mean(Y1)
    Y2 = y[100:]
    Y2 = np.mean(Y2)
    Z2 = z[100:]
    Z2 = np.mean(Z2)
    Z1 = z[:100]
    Z1 = np.mean(Z1)
    w1 = ((Z2-Z1)/(X2-X1))
    w2 = ((Z2-Z1)/(Y2-Y1))
    
   
    

# Initialize weights and bias
w1 = np.random.rand(1)
w2 = np.random.rand(1)
b = np.random.rand(1)
#Init_plane()
#b = random.uniform(-100, 100)
# Linear regression function
def f(x, y, w1, w2, b):
    return x * w1 + y * w2 + b
# Learning rate
k = 0.0001
loss =100
# Gradient Descent
for i in range(400):
    z_pred = f(x, y, w1, w2, b)
    #print(z_pred)
   # print("Predicted z:", z_pred)  
   # print("Actual z:", z)              
    loss = (((z_pred - z) ** 2).mean())
    #print(f"Iteration {i+1}, Loss: {loss}")  

    w1_grad = 2 * ((z_pred - z) * x).mean()
    w2_grad = 2 * ((z_pred - z) * y).mean()
    b_grad = 2 * (z_pred - z).mean()
    #w2_grad = 2 * ((z_pred - z) * y).mean()
    #b_grad = (1/(len(x)**0.5))

    w1 -= k * w1_grad
    w2 -= k * w2_grad
    b -= k * b_grad

def plot_perpendicular_plane(ax, x, y, z_pred, w1, w2, b):
    # Step 1: Point on original plane
    point_on_plane = np.array([np.mean(x), np.mean(y), np.mean(z_pred)])

    # Step 2: Original normal vector
    n = np.array([w1[0], w2[0], -1])

    # Step 3: Choose a perpendicular normal vector
    a, b_val = 1, 0
    c = w1[0]
    n_perp = np.array([a, b_val, c])

    # Step 4: Create a grid of x and z values
    x_range = np.linspace(min(x), max(x), 20)
    z_range = np.linspace(min(z_pred), max(z_pred), 20)
    X, Z = np.meshgrid(x_range, z_range)

    # Avoid divide-by-zero: if n_perp[1] == 0, pick different values
    if n_perp[1] == 0:
        a, b_val = 0, 1
        c = w2[0]
        n_perp = np.array([a, b_val, c])

    # Step 5: Solve for Y using the plane equation
    Y = ((-n_perp[0]*(X - point_on_plane[0]) - n_perp[2]*(Z - point_on_plane[2])) / n_perp[1]) + point_on_plane[1]

    # Step 6: Plot the perpendicular plane
    ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')

# Final prediction
z_pred = f(x, y, w1, w2, b)
print(f"w1: {w1}")

print(f"w2: {w2}")
print(f"b: {b}")
z_pred = np.array(z_pred)
print(f"z_pred: {z_pred[:3]}")

print(f"x: {x[:3]}")
print(f"Y: {y[:3]}")

# Plot the final results

plot_perpendicular_plane(ax, x, y, z_pred, w1, w2, b)
ax.plot_trisurf(x, y,z_pred, color='red', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()

# Final loss
loss = (((z_pred - z) ** 2).mean())
loss = (loss**0.5)
print('Final loss:', loss)
#print('Predicted values:', z_pred)
#print("Target values:", z)
et = time.time()
print(et-st)

above_plane = np.sum(z > z_pred)
below_plane = np.sum(z < z_pred)
print(f"Points above the Plane: {(above_plane/400)*100}")
print(f"Points below the Plane: {(below_plane/400)*100}" )
print(min(x))
print(min(y))
print(min(z_pred))
#print(z_pred)
print("----------------")
print(max(x))
print(max(y))
print(max(z_pred))

