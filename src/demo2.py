from oMap_builder import get_maps
import numpy as np
from scipy import stats

# TODO seem to work but it's super slow, the initial sum approach?
def aggregate(occ_maps):
    occ_maps = np.array(occ_maps)
    occ_maps_aggregated = np.ones(occ_maps[0].shape)
    for i in range(0, occ_maps_aggregated.shape[1]):
        cells = occ_maps[:, :, i]
        if not np.all(cells + 1):
            occ_maps_aggregated[0, i] = -1
        else:
            occ_maps_aggregated[0, i] = stats.mode(cells)[0]

    return occ_maps_aggregated




if __name__ == '__main__':
    # I've got a pose p1 and many other relative poses p2...pn from which I get many occupancy maps.
    # First I should choose the criterion to select the n-1 other poses (time?). Then aggregate all the occupancy maps
    # according to this rule: if all element on a given cell are -1 pick -1 for the final map; otherwise pick the mode

    # So to simulate this I need a relative pose for every couple p1-pi. Then I'll need sensor readings fon any pose.
    # At least, sensors_relative_poses is the same as well as the occ_map_coord. So, I think the best way is just to
    # create the occ_map_values randomly, give em to aggregate.

    # Create dummy occ_map_value to aggregate.
    # get_maps(...) returns a np.array, for this example the shape is (1,120701)
    occ_map_value_1 = np.ones((1, 120701)) * 0
    occ_map_value_2 = np.ones((1, 120701)) * 0
    occ_map_value_3 = np.ones((1, 120701)) * 0
    occ_map_value_4 = np.ones((1, 120701))
    occ_map_value_5 = np.ones((1, 120701))

    occ_maps = list()
    occ_maps.append(occ_map_value_1)
    occ_maps.append(occ_map_value_2)
    occ_maps.append(occ_map_value_3)
    occ_maps.append(occ_map_value_4)
    occ_maps.append(occ_map_value_5)

    aggregated_occ_map = aggregate(occ_maps)


    # I suppose that at the end we have a list of np.arrays

    print()
