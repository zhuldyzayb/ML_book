import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from pulp import *

##################   Linear optimisation  ####################
## Example 1: max (3x+y) subject to: x+y<=2, x<=1, y<=2 and x,y>=0
c = [-3,-1]
A = [[1,0],[0,1],[1,1], [-1,0], [0,-1]]
b = [1,2,2,0,0]
result = linprog(c, A_ub=A, b_ub=b, method="highs")
print("Results for Example 1: max 3x+y")
print("Algorithm convergence: ", result.success)
print("The maximum value attained: ", result.fun*(-1))
print("The optimal values for x and y: ", result.x)
# it is also possible to use the following code with indicated bounds:
# A = [[1,1]]
# b = [2]
# x_b = (0,1)
# y_b = (0,2)
# res = linprog(c, A_ub=A, b_ub=b, bounds = [x_b, y_b], method="highs")


## Example 2: max (40x1 + 30x2) subject to: x1+x2<=12, 2x1+x2<=16 and x1,x2>=0
c = [-40, -30]
A = [[1,1], [2,1], [-1,0], [0,-1]]
b = [12,16,0,0]
result = linprog(c, A_ub=A, b_ub=b, method="highs")
print(" ")
print("Results for Example 2: max 40x1+30x2")
print("Algorithm convergence: ", result.success)
print("The maximum value attained: ", result.fun*(-1))
print("The optimal values for x1 and x2: ", result.x)


##################   Polynomial function optimisation  ####################

## Example 3: min ( x^2 + 2x + 1)
def obj1(x):
    return x**2 + 2*x+1
x0 = -3.0
result = minimize(obj1, x0)
print(" ")
print("Polynomial optimisation examples: ")
print("Results for Example 3: min x^2+2x+1")
print("Algorithm convergence: ", result.success)
print("The minimum value attained: ", result.fun)
print("The optimal value for x: ", result.x)


## Example 4: min (x1^2 + x2^2) subject to x1+x2=1 and 0<= x1,x2<=1
obj2 = lambda x: x[0]**2 + x[1]**2
bnds = ((0,1), (0,1))
cons = ({'type': 'eq', 'fun': lambda x: x[0]+x[1]-1})
result = minimize(obj2, (0,0), bounds=bnds, constraints=cons)
print(" ")
print("Results for Example 4: min x1^2+ x2^2")
print("Algorithm convergence: ", result.success)
print("The minimum value attained: ", result.fun)
print("The optimal value for vector x: ", result.x)

## Example 5: min (x^4 + x^3 + x^2 + x)
##            subject to:  x^4-25>=0
##                         \sum_{i=1}^4 x_i^2 -40 = 0
##                         1<=x_i<=5
def obj5(x):
    return sum(x**4 + x**3 + x**2 + x)
def cons1(x):
    return np.sum(x**4) - 25
def cons2(x):
    return np.sum(x**2) - 40

bnds = [(1,5) for _ in range(4)]
cons = ({'type': 'ineq', 'fun': cons1},
        {'type': 'eq', 'fun': cons2})
x0 = np.array([2,2,2,2])
result = minimize(obj5, x0, bounds=bnds, constraints=cons)
print(" ")
print("Results for Example 5: min \sum_i x_i^4+ x_i^3 + x_i^2 + x_i")
print("Algorithm convergence: ", result.success)
print("The minimum value attained: ", result.fun)
print("The optimal value for vector x: ", result.x)

## Example 6: IntOpt: max ( -100x - 125 y)
##                    s.t.: 3x+6y<=30, 8x+4y<=44 and x,y>=0
# First, start with the simple case of linear programming
c = [-100, -125]
A = [[3,6],[8,4], [-1,0], [0,-1]]
b = [30,44,0,0]
result = linprog(c, A_ub=A, b_ub=b)
print(" ")
print("Results for Example 6: max 100x + 125y")
print("Algorithm convergence: ", result.success)
print("The minimum value attained: ", -result.fun)
print("The optimal values for x and y: ", result.x)

# Alternative package pulp
model = LpProblem(name = 'Maximise-Profit', sense=LpMaximize)
x = LpVariable(name='x', lowBound=0, cat='integer')
y = LpVariable(name='y', lowBound=0, cat='integer')

model += (3*x+6*y<=30, 'constraint1')
model += (8*x+4*y<=44, 'constraint2')
model += 100*x+125*y
result = model.solve()
print(" ")
print("Alternative solution to Example 6:")
print("Convergence: ", model.status)
print("Objective value:", model.objective.value())
print("Estimates of x and y:", x.value(), y.value())

## Example 7: min 10 - 5x1 + 2x2 - x3 subject to; x1+x2+x3 = 15, x1,x2,x3>=0

def obj7(x):
    return 10 - 5*x[0] + 2*x[1] -x[2]
def cons1(x):
    return x[0]+x[1]+x[2]-15
bnds = [(0,None) for _ in range(3)]
cons = ({'type': 'eq', 'fun': cons1})
x0 = np.array([1,1,1])
result = minimize(obj7, x0, bounds=bnds, constraints=cons)
print(result)

