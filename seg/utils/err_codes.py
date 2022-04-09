usage_instructions = f'Usage instructions: \n\
\t1. To specify a resize but no crop. Set image_height > 0, image_width > 0, crop_height = None, crop_width = None. NOTE: image_height and image_width are your resize sizes.\n\
\t2. To specify a resize and a crop. Set image_height > 0, image_width > 0, crop_height > 0, crop_width > 0.\n\
\t3. To specify no resize, but a cropped image of the input image. Set image_height = 0, image_width = 0, crop_height > 0, crop_width > 0'
def crop_err1(crop_height, crop_width):
    """
    Both the crop height and the crop width need to be defined. This error code
    is called if one parameter has been given but not the other. 
    """
    err_string = f'Both crop_height and crop_width need to be specified. Values given: (crop_height, crop_width): {crop_height, crop_width}. \n{usage_instructions}'
    return err_string 

def crop_err2(crop_height, crop_width, image_height, image_width):
    """
    Crop_size > Resize_size, resized size should be > crop_size
    """
    err_string = f'Crop_size: {crop_height, crop_width} > Image_size {image_height, image_width}. Look at your crop parameters again make sure theyre lower than your resize size'
    return err_string

def crop_err3(crop_height, crop_width):
    """
    Crop_height or crop_width both need to be 0 if one is
    """
    err_string = f'Both crop sizes should be same. We dont crop just one dimension. Crop size: {crop_height, crop_width}'
    return err_string

def resize_err1(image_height, image_width):
    """
    This error code is called when the image height and width are 0. 
    """
    err_string = f'One val of image_size unspeicified. Values given: (image_height, image_width): {image_height, image_width} \n{usage_instructions}'
    return err_string 

def resize_err2(image_height, image_width):
    """
    This is called when we just want to crop the input images with no resizing. 
    So we need both the image_height and the image_width to be 0. 
    """
    err_string = f'Both image_height and image_width need to be 0, only one has been given as 0. (image_height, image_width): {image_height, image_width} \n{usage_instructions}'
    return err_string

def resize_err3(image_height, image_width):
    err_string = f'Crop size: {None, None}, image_size: {image_height, image_width}. Looks like youre trying to just resize the image without a crop, but havent given a proper image_size = (image_height, image_width) to resize the image to. \n{usage_instructions}'
    return err_string

