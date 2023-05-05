import os

def create_image_list(input_folder, output_file):
    image_files = sorted(os.listdir(input_folder))

    with open(output_file, 'w') as f:
        for image_file in image_files:
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                f.write(f"file '{os.path.join(input_folder, image_file)}'\n")

if __name__ == "__main__":
    input_folder = "preprocessed/reprojected_hand"
    output_file = "preprocessed/images.txt"
    create_image_list(input_folder, output_file)
