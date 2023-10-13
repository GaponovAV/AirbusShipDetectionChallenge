from PIL import Image
import imagehash


'''
I also thought about searching for similar photos in order to remove
 them from the dataframe and not train the sample on duplicates 
'''
def find_duplicates(images_data):
    rle_dict = {}
    for image_name, rle in images_data.items():
        if rle in rle_dict:
            rle_dict[rle].append(image_name)
        else:
            rle_dict[rle] = [image_name]
            
    duplicates = {rle: image_list for rle, image_list in rle_dict.items() if len(image_list) > 1}
    
    for rle, image_list in duplicates.items():
        print(f"RLE {rle} has duplicate images: {', '.join(image_list)}")

    for image_list in duplicates.values():
        to_remove = image_list[1:]
        for image_name in to_remove:
            print(f"Removing image: {image_name}")
    return duplicates



def find_closest_image(target_image_path, image_paths_list, hash_size=8):
    """
    Finds the closest image to the target image from a list of image paths.
    
    Parameters:
        target_image_path (str): Path to the target image.
        image_paths_list (list): List of paths to the images to compare against.
        hash_size (int, optional): Hash size for the image hash function. Defaults to 8.
        
    Returns:
        str: Path to the closest image.
    """
    
    target_image = Image.open(target_image_path)
    target_hash = imagehash.average_hash(target_image, hash_size=hash_size)
    
    min_diff = float('inf')
    closest_image_path = None
    
    for image_path in image_paths_list:
        image = Image.open(image_path)
        curr_hash = imagehash.average_hash(image, hash_size=hash_size)
        
        diff = (target_hash - curr_hash)
        
        if diff < min_diff:
            min_diff = diff
            closest_image_path = image_path
            
    return closest_image_path