import numpy as np
import collections

def transform_inputs_to_count_vector(num_features, inputs):
    """Transforms our placeholder inputs to count representations rather than
    occurrences.

    e.g. for a single example: [1, 1, 2, 0] -> [1, 2, 1]

    """

    def transform_row_to_count_vector(index, row):
        counts = collections.Counter(row)
        vec = np.zeros(shape=(num_features))
        for ind, count in counts.items():
            vec[ind] = count
        return vec

    counts = [ transform_row_to_count_vector(index, inputs[index, :])
            for index in range(inputs.shape[0]) ]
    
    return np.stack(counts)

