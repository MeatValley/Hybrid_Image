import cv2
import numpy as np

def convolution(img, kernel):#calculate a conv between img*kernel

    """ This function executes the convolution between `img` and `kernel`."""

    print("[{}]\tRunning convolution...\n".format(img))

    image = cv2.imread(img)

    # Flip template before convolution.
    kernel = cv2.flip(kernel, -1)

    # Get size of image and kernel. 3rd value of shape is colour channel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    (pad_h, pad_w) = (kernel_h // 2, kernel_w // 2)

    # Create image to write to.// image base
    output = np.zeros(image.shape)


    # Slide kernel across every pixel.
    for y in range(pad_h, image_h - pad_h):
        for x in range(pad_w, image_w - pad_w):
            # If coloured, loop for colours.
            for colour in range(image.shape[2]):
                # Get center pixel.
                center = image[
                    y - pad_h : y + pad_h + 1, x - pad_w : x + pad_w + 1, colour
                ]
                # Perform convolution and map value to [0, 255].
                # Write back value to output image.
                output[y, x, colour] = (center * kernel).sum() / 255

    # Return the result of the convolution.
    return output

def fourier(img, kernel): #calculate a conv between img*kernel

    """ Compute convolution between `img` and `kernel` using numpy's FFT."""

    image = cv2.imread(img)
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]

    # Apply padding to the kernel.
    padded_kernel = np.zeros(image.shape[:2])
    start_h = (image_h - kernel_h) // 2
    start_w = (image_w - kernel_w) // 2
    padded_kernel[start_h : start_h + kernel_h, start_w : start_w + kernel_w] = kernel

    # Create image to write to.
    output = np.zeros(image.shape)

    # Run FFT on all 3 channels.
    for colour in range(3):
        Fi = np.fft.fft2(image[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        # Inverse fourier.
        output[:, :, colour] = np.fft.fftshift(np.fft.ifft2(Fi * Fk)) / 255

    # Return the result of convolution.
    return output

def gaussian_blur(image, sigma, fourier):
    """ Builds a Gaussian kernel used to perform the LPF on an image.
    """
    print("[{}]\tCalculating Gaussian kernel...".format(image))

    # Calculate size of filter.
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros((size, size))

    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    if fourier: #if want to use fourier
        return fourier(image, kernel)
    else:
        return convolution(image, kernel)


def low_pass(image, cutoff, fourier):
    """ Generate low pass filter of image.
    """
    print("[{}]\tGenerating low pass image...".format(image))
    return gaussian_blur(image, cutoff, fourier)
# def rect_low_pass():


def high_pass(image, cutoff, fourier):
    """ Generate high pass filter of image. This is simply the image minus its
    low passed result.
    """
    print("[{}]\tGenerating high pass image...".format(image))
    return (cv2.imread(image) / 255) - low_pass(image, cutoff, fourier)

def hybrid_image(image, cutoff, fourier):
    """ Create a hybrid image by summing together the low and high frequency
    images.
    """
    # Perform low pass filter and export.
    low = low_pass(image[0], cutoff[0], fourier)
    cv2.imwrite("low.jpg", low * 255)
    # Perform high pass filter and export.
    high = high_pass(image[1], cutoff[1], fourier)
    cv2.imwrite("high.jpg", (high + 0.5) * 255)

    print("Creating hybrid image...")
    low = resize_img(high, low)
    return low + high

def show_img(img):
    cv2.imshow("hybrid img", img)
    cv2.waitKey(0)	
    cv2.destroyAllWindows()

def resize_img(img1, img2):
    h = img1.shape[0]
    w = img1.shape[1]
    dim = (w,h)
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    return resized

def resize_img_p(img, percentage):
    scale_percent = percentage # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    return resized

hyb = hybrid_image(['fer_cinza.png','pb.png'] , [1,4], 0)

show_img(hyb)
