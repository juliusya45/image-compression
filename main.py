import math
import numpy as np

from PIL import Image

T = np.array([
    [0.3536,  0.3536,  0.3536,  0.3536,  0.3536,  0.3536,  0.3536,  0.3536],
    [0.4904,  0.4157,  0.2778,  0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
    [0.4619,  0.1913, -0.1913, -0.4619, -0.4619, -0.1913,  0.1913,  0.4619],
    [0.4157, -0.0975, -0.4904, -0.2778,  0.2778,  0.4904,  0.0975, -0.4157],
    [0.3536, -0.3536, -0.3536,  0.3536,  0.3536, -0.3536, -0.3536,  0.3536],
    [0.2778, -0.4904,  0.0975,  0.4157, -0.4157, -0.0975,  0.4904, -0.2778],
    [0.1913, -0.4619,  0.4619, -0.1913, -0.1913,  0.4619, -0.4619,  0.1913],
    [0.0975, -0.2778,  0.4157, -0.4904,  0.4904, -0.4157,  0.2778, -0.0975]
]).astype(np.float64)

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]).astype(np.float64)

Q10 = np.array(
[[ 80,  55,  50,  80, 120, 200, 255, 255],
 [ 60,  60,  70,  95, 130, 255, 255, 255],
 [ 70,  65,  80, 120, 200, 255, 255, 255],
 [ 70,  85, 110, 145, 255, 255, 255, 255],
 [ 90, 110, 185, 255, 255, 255, 255, 255],
 [120, 175, 255, 255, 255, 255, 255, 255],
 [245, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255]]
).astype(np.float64)


def main():
    # Open the image file for reading
    filePath = "bit-256-x-256-Grayscale-Lena-Image.png"
    img = Image.open(filePath).convert('L')  ## we load the image with grayscale values

    # Get the width and height from the image. Size is stored as [width, height]
    width, height = img.size

    # Then split the image up into multiple blocks
    blocks = get_image_blocks(filePath, (8,8))

    dctBlocks =[]

    # Now for every block get the -128 and then do DCT operation on it:
    for block in blocks:
        blockMat = np.array(block).astype(np.float64)
        #print(blockMat)
        # now subtract -128 for every pixel:
        for i in range(8):
            for j in range(8):
                blockMat[i][j] = blockMat[i][j] - 128

        #print("After subtraction:")
        #print(blockMat)
        DCT_result = (T @ blockMat) @ T.T
        #print("After DCT:")
        #print(DCT_result)
        quantized_result = np.round(DCT_result / Q10)
        #print("After Quantization:")
        #print(quantized_result)
        reconstructed_quantized = Q10 * quantized_result
        #print("After Reconstruction:")
        #print(reconstructed_quantized)

        final_result = np.round((T.T @ reconstructed_quantized) @ T) + 128
        dctBlocks.append(Image.fromarray(final_result))
    dctImg = reconstruct_image(dctBlocks, round(width / 8), round(height / 8), 8, 8)
    dctImg.save('dct.png')

# Splits an image into blocks of specified size.
# image_path (str): Path to the image file.
# block_size (tuple): Tuple representing the width and height of each block.

# Returns a list of blocks that
def get_image_blocks(image_path, block_size):

    with Image.open(image_path).convert('L') as img:
        width, height = img.size
        blocks = []

        for y in range(0, height, block_size[1]):
            for x in range(0, width, block_size[0]):
                box = (x, y, x + block_size[0], y + block_size[1])
                block = img.crop(box)
                blocks.append(block)

    return blocks


# Takes in the list of blocks and reconstructs the full image
def reconstruct_image(blocks, blocks_x, blocks_y, block_width, block_height):
    # Reconstruct the full image (grayscale)
    img_array = np.zeros((blocks_y * block_height, blocks_x * block_width), dtype=np.uint8)

    for i in range(blocks_x):
        for j in range(blocks_y):
            img_array[i * block_width:(i + 1) * block_width, j * block_height:(j + 1) * block_height] = blocks[
                i * blocks_y + j]

    # Convert numpy array back to image (grayscale)
    img = Image.fromarray(img_array)
    return img


if __name__ == "__main__":
    main()
