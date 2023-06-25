import numpy as np

def read_raw_image(file_path: str, width: int, height: int, black_level: int, bpp: int, bit_depth:int ) -> np.array:
    with open(file_path, 'rb') as raw_file:
        raw_data = raw_file.read()

    image_size = width * height

    image_data = np.frombuffer(raw_data, dtype=np.uint16, count=image_size)

    f = lambda x: x - black_level if x > black_level else 0
    vf = np.vectorize(f)
    image_data = vf(image_data)
    max_value = (2 ** bit_depth) - 1
    image_data = (image_data / max_value) * ((2 ** bpp) - 1)

    image = np.reshape(image_data, (height, width)).astype(np.uint16)

    return image
