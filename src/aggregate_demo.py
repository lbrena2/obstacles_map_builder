import numpy as np
from scipy import stats


def aggregate(maps):
    aggregated_map = np.full_like(maps[0], -1)
    map_mask = (maps != -1).any(1)
    nonzero_maps = maps[map_mask]
    cell_mask = (nonzero_maps != -1).all(0)
    nonzero_cells = nonzero_maps[:, cell_mask]
    aggregated_map[cell_mask] = stats.mode(nonzero_cells, 0)[0][0]
    return aggregated_map


if __name__ == '__main__':
    occ_map_value_1 = np.full((32,), 0)
    occ_map_value_1[16:] = -1
    occ_map_value_2 = np.full((32,), -1)
    occ_map_value_2[:12] = 1
    occ_map_value_3 = np.full((32,), -1)
    occ_map_value_4 = np.full((32,), -1)
    occ_map_value_5 = np.full((32,), -1)

    occ_maps = list()
    occ_maps.append(occ_map_value_1)
    occ_maps.append(occ_map_value_2)
    occ_maps.append(occ_map_value_3)
    occ_maps.append(occ_map_value_4)
    occ_maps.append(occ_map_value_5)
    occ_maps = np.array(occ_maps)

    aggregated_occ_map = aggregate(occ_maps)

    # I suppose that at the end we have a list of np.arrays

    print(aggregated_occ_map)
