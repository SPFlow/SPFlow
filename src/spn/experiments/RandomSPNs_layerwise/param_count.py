def sum_first_nr_params(D, I, S, R, C):
    return R * (2 ** D * S * I + S ** 3 * sum([2 ** i for i in range(1, (D - 1) + 1)]) + S ** 2 * C)

def prod_first_nr_params(D, I, S, R, C):
    return R * (2 ** (D - 1) * S * I ** 2 + S ** 3 * sum([2 ** i for i in range(1, (D - 2) + 1)]) + S ** 2 * C)

if __name__ == "__main__":
    R = 1
    C = 1
    print(f"R={R}, C={C}")
    I_S = [(5, 10), (10, 5), (10, 10)]
    for D in range(2, 4):
        for I, S in I_S:
            sum_first = sum_first_nr_params(D, I, S, R, C)
            prod_first = prod_first_nr_params(D, I, S, R, C)
            print(f"D={D} I={I}, S={S} - sum layer first: {sum_first} sum params - prod layer first: {prod_first} sum params")
    print("Parameter count equivalences")
    for D in range(2, 4):
        for I, S in I_S:
            sum_first = sum_first_nr_params(D-1, I**2, S, R, C)
            prod_first = prod_first_nr_params(D, I, S, R, C)
            print(f"Sum-first D={D-1} I={I**2}, S={S}: {sum_first} sum params - Prod-first D={D} I={I}, S={S}: {prod_first} sum params")


