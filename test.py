import cplex

c = cplex.Cplex()
indices = c.variables.add(names = ['x','y','z'])
l = cplex.SparsePair(ind = ['x'], val = [1.0])
q = cplex.SparseTriple(ind1 = ['x','y'], ind2 = ['x','z'], val = [1.0,2])
c.quadratic_constraints.add(name=str(2), lin_expr=l, quad_expr=q, rhs=2,sense='L')

#print(c.quadratic_constraints.get_linear_components())

#cplex.SparseTriple(["x1","x2"],["x1","x2"],[-1.0,1.0])
#
#
#c.quadratic_constraints.add(lin_expr = Data.qlin[q],
#                            quad_expr = Data.quad[q],
#                            sense = Data.qsense[q],
#                            rhs = Data.qrhs[q],
#                            name = Data.qname[q])


