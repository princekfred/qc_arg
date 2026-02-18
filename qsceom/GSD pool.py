def generate_SQ_Operators():
    """
    n_orb is number of spatial orbitals assuming that spin orbitals are labelled
    0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
    """

    print(" Form singlet GSD operators")
    n_orb = 4 #I am passing it ----- Spatial orbitals 
    fermi_ops = []
    for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1

        for q in range(p, n_orb):
            qa = 2*q
            qb = 2*q+1

            termA =  FermionOperator(((pa,1),(qa,0)))
            termA += FermionOperator(((pb,1),(qb,0)))

            termA -= hermitian_conjugated(termA)

            termA = normal_ordered(termA)

            #Normalize
            coeffA = 0
            for t in termA.terms:
                coeff_t = termA.terms[t]
                coeffA += coeff_t * coeff_t

            if termA.many_body_order() > 0:
                termA = termA/np.sqrt(coeffA)
                fermi_ops.append(termA)


    pq = -1
    for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1

        for q in range(p,n_orb):
            qa = 2*q
            qb = 2*q+1

            pq += 1

            rs = -1
            for r in range(0, n_orb):
                ra = 2*r
                rb = 2*r+1

                for s in range(r, n_orb):
                    sa = 2*s
                    sb = 2*s+1

                    rs += 1

                    if(pq > rs):
                        continue


                    termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                    termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                    termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                    termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                    termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                    termA -= hermitian_conjugated(termA)
                    termB -= hermitian_conjugated(termB)

                    termA = normal_ordered(termA)
                    termB = normal_ordered(termB)

                    #Normalize
                    coeffA = 0
                    coeffB = 0
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
                    for t in termB.terms:
                        coeff_t = termB.terms[t]
                        coeffB += coeff_t * coeff_t


                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        fermi_ops.append(termA)

                    if termB.many_body_order() > 0:
                        termB = termB/np.sqrt(coeffB)
                        fermi_ops.append(termB)

    n_ops = len(fermi_ops)
    print(" Number of operators: ", n_ops)
    return fermi_ops
# }}}

fermi_ops = generate_SQ_Operators()
x = [None] * len(fermi_ops)
print('Before loop, len of x', len(x))

for i in range(len(fermi_ops)):
    x[i] = qml.from_openfermion(fermi_ops[i])
print('Total operators after loop  are', len(x))
