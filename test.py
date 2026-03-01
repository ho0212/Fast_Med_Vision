import numpy as np
import fast_med_vision as fmv

# Create dummy data
dummy_data = np.array([
    [[1.0, 2.0], [3.0, 4.0]], 
    [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)

print("Original Data:")
print(dummy_data)
print("=" * 30)

# Call the function from the fast_med_vision library
result = fmv.process_image(dummy_data)

print("Processed Data:")
print(result)