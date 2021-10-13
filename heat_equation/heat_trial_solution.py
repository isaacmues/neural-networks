"""
In the Time-Dependent section of the Mathematica Documentation:

∂ₜu(x,t) - ∂ₓ,ₓu(x,t) = 0, x ∊ [-1, 1], t ∊ [0, π]

With boundary conditions

∂ₓu(x,t) = cos(t), x = 1
u(x,t) = -1, x = -1

And the initial condition

u(x,0) = x
"""

import sympy as sp

x,t = sp.symbols("x t", real=True)
n = sp.Function("N")

trial_sol = (x + 1) * t * (n(x, t) - n(1, t) - 2* sp.diff(n(x,t), x).subs(x, 1))
trial_sol += -1 + (x + 1) * sp.cos(t)

print("The trial solution\n")
sp.pprint(trial_sol)

print("\nThe initial condition\n")
sp.pprint(trial_sol.subs(t, 0))

print("\nThe boundary condition at x = -1\n")
sp.pprint(trial_sol.subs(x, -1))

print("\nThe boundary condition at x = 1\n")
sp.pprint(trial_sol.diff(x).subs(x, 1))
