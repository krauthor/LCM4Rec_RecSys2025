import os
import json

import numpy as np

def store_results(results, result_file_path):
    # Save the results to a JSON file
    if not os.path.exists(os.path.dirname(result_file_path)):
        os.makedirs(os.path.dirname(result_file_path))

    # Encode numpy arrays as lists to make the JSON file storable
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    with open(result_file_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    print(f"Results saved to {result_file_path}")