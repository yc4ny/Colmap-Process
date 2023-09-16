import numpy as np 
import pickle

def smooth_matrices(matrix1, matrix2, smoothing_factor):
    smoothed_matrix = (1 - smoothing_factor) * matrix1 + smoothing_factor * matrix2
    return smoothed_matrix

if __name__ == "__main__":

    # with open(f'data/desk/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
    #     head_extrin = pickle.load(f)
    # with open(f'data/desk/camera_extrinsic/left_extrinsic.pkl', 'rb') as f:
    #     left_extrin = pickle.load(f)
    with open(f'data/desk/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
        head_extrin = pickle.load(f)
    
    base_name = "head_"
    total_error = 0
    smoothing_factor = 0.2
    count = 0
    for i in range(len(head_extrin) - 1):
        num_string1 = str(i).zfill(5)
        num_string2 = str(i + 1).zfill(5)
        string_1 = base_name + num_string1 + ".jpg"
        string_2 = base_name + num_string2 + ".jpg"

        try:
            matrix1 = head_extrin[string_1]
            matrix2 = head_extrin[string_2]

            element_wise_squared_diff = (matrix1 - matrix2) ** 2
            frobenius_norm = np.sqrt(np.sum(element_wise_squared_diff))
            average_distance = frobenius_norm / (matrix1.shape[0] * matrix1.shape[1])
            total_error += average_distance
            print(f"Distance between {string_1} and {string_2}: {average_distance}")

            # Perform bilinear interpolation if needed
            if average_distance >= 0.01:
                print("Performing bilinear interpolation")
                smoothed_matrix2 = smooth_matrices(matrix1, matrix2, smoothing_factor)
                head_extrin[string_2] = smoothed_matrix2
                print(smoothed_matrix2)
                count += 1

        except KeyError:
            # Handle the case when the key is not found
            print(f"Key not found for {string_1} or {string_2}. Skipping iteration.")

    if len(head_extrin) > 1:
        average_error = total_error / (len(head_extrin) - 1)
        print(f"Average Distance: {average_error}")

    with open("head_extrinsics.pkl", 'wb') as file:
        pickle.dump(head_extrin, file)

    print("Number of smoothed frames: " + str(count))
          
