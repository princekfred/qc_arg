def inite(elec: int, orb: int) -> list[list[int]]:
    """Generate a list of occupation configurations.

    Parameters
    ----------
    elec
        Number of active electrons.
    orb
        Number of spin-orbitals/qubits.

    Returns
    -------
    list[list[int]]
        A list of configurations, each represented as a list of occupied indices.
    """

    config: list[int] = []
    list1: list[list[int]] = []

    # singles
    for x in range(elec):
        count = orb - elec
        while count < orb:
            for e in range(elec):
                if x == e:
                    if x % 2 == 0:
                        config.append(count)
                        count = count + 2
                    else:
                        config.append(count + 1)
                        count = count + 2
                else:
                    config.append(e)

            list1.append(config)
            config = []

    # doubles
    for x in range(elec):
        for y in range(x + 1, elec):
            for count1 in range(elec, orb, 2):
                for count2 in range(elec, orb, 2):
                    cont = 0
                    if count1 == count2:
                        if (x % 2) != (y % 2):
                            cont = 1
                    else:
                        cont = 1
                    if (x % 2) == (y % 2) and count2 < count1:
                        cont = 0

                    if cont == 1:
                        for e in range(elec):
                            if x == e:
                                if x % 2 == 0:
                                    config.append(count1)
                                else:
                                    config.append(count1 + 1)
                            elif y == e:
                                if y % 2 == 0:
                                    config.append(count2)
                                else:
                                    config.append(count2 + 1)
                            else:
                                config.append(e)

                        list1.append(config)
                        config = []

    return list1