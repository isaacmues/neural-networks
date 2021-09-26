import sympy as sym

x, y = sym.symbols("x y", real=True)
n = sym.Function("N")

psi_a = sym.exp(-x) * (x + y**3)
f0 = psi_a.subs(x,0)
f1 = psi_a.subs(x,1)
g0 = psi_a.subs(y,0)
g1 = psi_a.subs(y,1)

psi_t = x * (1 - x) * y * (1 - y) * n(x,y)
psi_t += (1 - x) * f0
psi_t += x * f1
psi_t += (1 - y) * (g0 - x * sym.exp(-1))
psi_t += y * (g1 - (1 - x + 2 * x * sym.exp(-1)))

print("La solución analítica es\n")
sym.pprint(psi_a)

print("\nx = 0\n")
sym.pprint(f0)

print("\nx = 1\n")
sym.pprint(f1)

print("\ny = 0\n")
sym.pprint(g0)

print("\ny = 1\n")
sym.pprint(g1)

print("\nEl laplaciano es\n")
pde = sym.simplify(psi_a.diff(x,2) + psi_a.diff(y,2))
sym.pprint(pde)

print("Solución de prueba\n")
sym.pprint(psi_t)

print("\nx = 0\n")
sym.pprint(psi_t.subs(x,0))
if psi_t.subs(x,0) - f0 == 0:
    print("Condición correcta")
else:
    print("Algo anda mal")

print("\nx = 1\n")
sym.pprint(psi_t.subs(x,1))
if psi_t.subs(x,1) - f1 == 0:
    print("Condición correcta")
else:
    print("Algo anda mal")

print("\ny = 0\n")
sym.pprint(psi_t.subs(y,0))
if psi_t.subs(y,0) - g0 == 0:
    print("Condición correcta")
else:
    print("Algo anda mal")

print("\ny = 1\n")
sym.pprint(psi_t.subs(y,1))
if psi_t.subs(y,1) - g1 == 0:
    print("Condición correcta")
else:
    print("Algo anda mal")
