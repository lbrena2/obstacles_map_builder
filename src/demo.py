import numpy as np
from scipy import stats

def aggregate(maps):
    # maps is a np.array shape(numbers_of_maps x row_occ_mapp x col_occ_map)
    aggregated_map = np.ones(maps.shape[1]) * -1

    for i in range(0, aggregated_map.shape[0]):
        cells = maps[:, i]
        if not (cells == -1).all():
            aggregated_map[i] = stats.mode(cells)[0]

    return aggregated_map


if __name__ == '__main__':
    for i in range(1,10):
        if i < 5:
            print(i)
        continue

        for j in range(1,20):
            print(j)


    # I suppose that at the end we have a list of np.arrays

    print()