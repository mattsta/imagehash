from PIL import Image, ImageFilter
import numpy as np
import base64

__version__ = "4.1.0"


class ImageHash:
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array=None, dct=None, hashfn=None, restore=None):
        self.precomputedHash = binary_array

        # For creating DCT rotations, we need to store the DCT directly along with
        # how we transform the DCT to get the final hash value (hashfn(dct)):
        self.dct = dct
        self.hashfn = hashfn

        if isinstance(self.hash, np.ndarray):
            # convert boolean tensors into binary
            self.precomputedHash = self.hash.astype(int)

        if restore:
            # we can restore via hex hashes or more compact b85 representations
            if restore == "hex":
                asint = int(self.hash, 16)
                bits = asint.bit_length()

                # Note: no fractional bytes here, so we don't need to do the (+ 7) ceiling trick
                self.precomputedHash = np.unpackbits(
                    np.frombuffer(
                        asint.to_bytes(bits // 8, "big"),
                        dtype=np.uint8,
                    )
                )
            elif restore == "b85":
                # else, restoring from base-85 save()
                self.precomputedHash = np.unpackbits(
                    np.frombuffer(base64.b85decode(self.hash), dtype=np.uint8)
                )
            else:
                raise Exception("Unknown restore type requested? Try 'hex' or 'b85'")

            # restore original rows/cols (all hashes are square)
            rowscols = int(np.sqrt(self.precomputedHash.size))
            self.precomputedHash = self.hash.reshape(rowscols, rowscols)

    @property
    def hash(self):
        if self.precomputedHash is not None:
            return self.precomputedHash

        """ Perform the DCT hash using requested exclusion criteria """
        self.precomputedHash = self.hashfn(self.dct)
        return self.precomputedHash

    def __str__(self):
        # convert base hash to bytes, convert bytes to integer,
        # convert integer to lower case hex string:
        return f"{int.from_bytes(np.packbits(self.precomputedHash), 'big'):x}"

    def save(self):
        # Note: this ONLY works if each hash array element is a binary value.
        #       Does not save/restore any values *not* 0 or 1
        return base64.b85encode(np.packbits(self.hash))

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError("Other hash must not be None")

        if self.hash.size != other.hash.size:
            raise TypeError(
                "ImageHashes must be of the same shape",
                self.hash.shape,
                other.hash.shape,
            )
        return np.count_nonzero(self.hash - other.hash)

    def __eq__(self, other):
        if other is None:
            return False

        return np.array_equal(self.hash, other.hash)

    def __ne__(self, other):
        if other is None:
            return False

        return not self == other

    def __hash__(self):
        # giant number as hash
        return int.from_bytes(np.packbits(self.hash), "big")

    def __len__(self):
        # Returns the bit length of the hash
        return self.hash.size

    def isometric(self):
        return {
            k: self.hashfn(dct).astype(int)
            for k, dct in get_isometric_dct_transforms(self.dct).items()
        }


def average_hash(image, hash_size=8, mean=np.mean):
    """
    Average Hash computation

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    Step by step explanation: https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/

    @image must be a PIL instance.
    @mean how to determine the average luminescence. can try np.median instead.
    """
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    # reduce size and complexity, then covert to grayscale
    image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)

    # find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
    pixels = np.asarray(image)
    avg = mean(pixels)

    # create string of bits
    diff = pixels > avg
    # make a hash
    return ImageHash(diff)


def phash(image, hash_size=8, highfreq_factor=4, blur=True):
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @image must be a PIL instance.
    """
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    import scipy.fftpack

    img_size = hash_size * highfreq_factor
    if blur:
        image = image.filter(ImageFilter.BoxBlur(3))  # a 7x7 blur (radius 3)

    image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
    pixels = np.asarray(image)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)

    # Note: 'perception' lib also supports frequency shifting the hash by not starting from
    #       the start, but by starting from an offset. Not sure how much it improves results:
    #       dct[self.freq_shift : self.hash_size + self.freq_shift,
    #           self.freq_shift : self.hash_size + self.freq_shift]
    dctlowfreq = dct[:hash_size, :hash_size]

    # hashfn:
    # med = np.median(dctlowfreq)
    # diff = dctlowfreq > med

    return ImageHash(dct=dctlowfreq, hashfn=lambda x: x > np.median(x))


def phash_simple(image, hash_size=8, highfreq_factor=4, blur=True):
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @image must be a PIL instance.
    """
    import scipy.fftpack

    img_size = hash_size * highfreq_factor
    if blur:
        image = image.filter(ImageFilter.BoxBlur(3))  # a 7x7 blur (radius 3)

    image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
    pixels = np.asarray(image)
    dct = scipy.fftpack.dct(pixels)
    dctlowfreq = dct[:hash_size, 1 : hash_size + 1]

    # hashfn:
    # avg = dctlowfreq.mean()
    # diff = dctlowfreq > avg

    return ImageHash(dct=dctlowfreq, hashfn=lambda x: x > x.mean())


def get_isometric_dct_transforms(dct: np.ndarray):
    """Convert a DCT into its rotational equivalents.

    Rotates a hash without needing to re-evaluate the image for each rotation"""
    # From https://github.com/thorn-oss/perception/blob/09086f368742e6135cd5eb8497c9f7a59eaa7f0b/perception/hashers/tools.py#L294
    # pylint: disable=invalid-name
    T1 = np.empty_like(dct)
    T1[::2] = 1
    T1[1::2] = -1

    # pylint: disable=invalid-name
    T2 = np.empty_like(dct)
    T2[::2, ::2] = 1
    T2[1::2, 1::2] = 1
    T2[::2, 1::2] = -1
    T2[1::2, ::2] = -1
    return dict(
        r0=dct,
        fv=dct * T1,
        fh=dct * T1.T,
        r180=dct * T2,
        r90=dct.T * T1,
        r90fv=dct.T,
        r90fh=dct.T * T2,
        r270=dct.T * T1.T,
    )


def dhash(image, hash_size=8):
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences horizontally

    @image must be a PIL instance.
    """
    # resize(w, h), but np.array((h, w))
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    image = image.convert("L").resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    pixels = np.asarray(image)
    # compute differences between columns
    diff = pixels[:, 1:] > pixels[:, :-1]
    return ImageHash(diff)


def dhash_vertical(image, hash_size=8):
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences vertically

    @image must be a PIL instance.
    """
    # resize(w, h), but np.array((h, w))
    image = image.convert("L").resize((hash_size, hash_size + 1), Image.ANTIALIAS)
    pixels = np.asarray(image)
    # compute differences between rows
    diff = pixels[1:, :] > pixels[:-1, :]
    return ImageHash(diff)


def whash(image, hash_size=8, image_scale=None, mode="haar", remove_max_haar_ll=True):
    """
    Wavelet Hash computation.

    based on https://www.kaggle.com/c/avito-duplicate-ads-detection/

    @image must be a PIL instance.
    @hash_size must be a power of 2 and less than @image_scale.
    @image_scale must be power of 2 and less than image size. By default is equal to max
            power of 2 for an input image.
    @mode (see modes in pywt library):
            'haar' - Haar wavelets, by default
            'db4' - Daubechies wavelets
    @remove_max_haar_ll - remove the lowest low level (LL) frequency using Haar wavelet.
    """
    import pywt

    if image_scale is not None:
        assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
    else:
        image_natural_scale = 2 ** int(np.log2(min(image.size)))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level = int(np.log2(image_scale))

    level = int(np.log2(hash_size))
    assert hash_size & (hash_size - 1) == 0, "hash_size is not power of 2"
    assert level <= ll_max_level, "hash_size in a wrong range"
    dwt_level = ll_max_level - level

    image = image.convert("L").resize((image_scale, image_scale), Image.ANTIALIAS)
    pixels = np.asarray(image) / 255.0

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    if remove_max_haar_ll:
        coeffs = pywt.wavedec2(pixels, "haar", level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        pixels = pywt.waverec2(coeffs, "haar")

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
    dwt_low = coeffs[0]

    # hashfn:
    # Substract median and compute hash
    # med = np.median(dwt_low)
    # diff = dwt_low > med
    return ImageHash(dct=dwt_low, hashfn=lambda x: x > np.median(x))


def colorhash(image, binbits=3):
    """
    Color Hash computation.

    Computes fractions of image in intensity, hue and saturation bins:

    * the first binbits encode the black fraction of the image
    * the next binbits encode the gray fraction of the remaining image (low saturation)
    * the next 6*binbits encode the fraction in 6 bins of saturation, for highly saturated parts of the remaining image
    * the next 6*binbits encode the fraction in 6 bins of saturation, for mildly saturated parts of the remaining image

    @binbits number of bits to use to encode each pixel fractions
    """

    # bin in hsv space:
    intensity = np.asarray(image.convert("L")).flatten()
    h, s, v = [np.asarray(v).flatten() for v in image.convert("HSV").split()]
    # black bin
    mask_black = intensity < 256 // 8
    frac_black = mask_black.mean()
    # gray bin (low saturation, but not black)
    mask_gray = s < 256 // 3
    frac_gray = np.logical_and(~mask_black, mask_gray).mean()
    # two color bins (medium and high saturation, not in the two above)
    mask_colors = np.logical_and(~mask_black, ~mask_gray)
    mask_faint_colors = np.logical_and(mask_colors, s < 256 * 2 // 3)
    mask_bright_colors = np.logical_and(mask_colors, s > 256 * 2 // 3)

    c = max(1, mask_colors.sum())
    # in the color bins, make sub-bins by hue
    hue_bins = np.linspace(0, 255, 6 + 1)
    if mask_faint_colors.any():
        h_faint_counts, _ = np.histogram(h[mask_faint_colors], bins=hue_bins)
    else:
        h_faint_counts = np.zeros(len(hue_bins) - 1)
    if mask_bright_colors.any():
        h_bright_counts, _ = np.histogram(h[mask_bright_colors], bins=hue_bins)
    else:
        h_bright_counts = np.zeros(len(hue_bins) - 1)

    # now we have fractions in each category (6*2 + 2 = 14 bins)
    # convert to hash and discretize:
    maxvalue = 2 ** binbits
    values = [
        min(maxvalue - 1, int(frac_black * maxvalue)),
        min(maxvalue - 1, int(frac_gray * maxvalue)),
    ]
    for counts in list(h_faint_counts) + list(h_bright_counts):
        values.append(min(maxvalue - 1, int(counts * maxvalue * 1.0 / c)))
    # print(values)
    bitarray = []
    for v in values:
        bitarray += [
            v // (2 ** (binbits - i - 1)) % 2 ** (binbits - i) > 0
            for i in range(binbits)
        ]
    return ImageHash(np.asarray(bitarray).reshape((-1, binbits)))


class ImageMultiHash:
    """
    This is an image hash containing a list of individual hashes for segments of the image.
    The matching logic is implemented as described in Efficient Cropping-Resistant Robust Image Hashing
    """

    def __init__(self, hashes):
        self.segment_hashes = hashes

    def __eq__(self, other):
        if other is None:
            return False
        return self.matches(other)

    def __ne__(self, other):
        if other is None:
            return False
        return not self.matches(other)

    def __sub__(self, other, hamming_cutoff=None, bit_error_rate=None):
        matches, sum_distance = self.hash_diff(other, hamming_cutoff, bit_error_rate)
        max_difference = len(self.segment_hashes)
        if matches == 0:
            return max_difference
        max_distance = matches * len(self.segment_hashes[0])
        tie_breaker = 0 - (float(sum_distance) / max_distance)
        match_score = matches + tie_breaker
        return max_difference - match_score

    def __hash__(self):
        return hash(tuple(hash(segment) for segment in self.segment_hashes))

    def __str__(self):
        # Serialization format is [HASHSIZE],[HASH][HASHSIZE],[HASH]...
        totals = []
        for x in self.segment_hashes:
            h = str(x)
            l = len(h)
            totals.extend([str(l), ",", h])

        return "".join(totals)

    def __repr__(self):
        return repr(self.segment_hashes)

    def hash_diff(self, other_hash, hamming_cutoff=None, bit_error_rate=None):
        """
        Gets the difference between two multi-hashes, as a tuple. The first element of the tuple is the number of
        matching segments, and the second element is the sum of the hamming distances of matching hashes.
        NOTE: Do not order directly by this tuple, as higher is better for matches, and worse for hamming cutoff.
        :param other_hash: The image multi hash to compare against
        :param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
        :param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
        default of 0.25 means that the segment hashes can be up to 25% different
        """
        # Set default hamming cutoff if it's not set.
        if hamming_cutoff is None and bit_error_rate is None:
            bit_error_rate = 0.25
        if hamming_cutoff is None:
            hamming_cutoff = len(self.segment_hashes[0]) * bit_error_rate
        # Get the hash distance for each region hash within cutoff
        distances = []
        for segment_hash in self.segment_hashes:
            lowest_distance = min(
                segment_hash - other_segment_hash
                for other_segment_hash in other_hash.segment_hashes
            )
            if lowest_distance > hamming_cutoff:
                continue
            distances.append(lowest_distance)
        return len(distances), sum(distances)

    def matches(
        self, other_hash, region_cutoff=1, hamming_cutoff=None, bit_error_rate=None
    ):
        """
        Checks whether this hash matches another crop resistant hash, `other_hash`.
        :param other_hash: The image multi hash to compare against
        :param region_cutoff: The minimum number of regions which must have a matching hash
        :param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
        :param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
        default of 0.25 means that the segment hashes can be up to 25% different
        """
        matches, _ = self.hash_diff(other_hash, hamming_cutoff, bit_error_rate)
        return matches >= region_cutoff

    def best_match(self, other_hashes, hamming_cutoff=None, bit_error_rate=None):
        """
        Returns the hash in a list which is the best match to the current hash
        :param other_hashes: A list of image multi hashes to compare against
        :param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
        :param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff.
        Defaults to 0.25 if unset, which means the hash can be 25% different
        """
        return min(
            other_hashes,
            key=lambda other_hash: self.__sub__(
                other_hash, hamming_cutoff, bit_error_rate
            ),
        )


def _find_region(remaining_pixels, segmented_pixels):
    """
    Finds a region and returns a set of pixel coordinates for it.
    :param remaining_pixels: A np bool array, with True meaning the pixels are remaining to segment
    :param segmented_pixels: A set of pixel coordinates which have already been assigned to segment. This will be
    updated with the new pixels added to the returned segment.
    """
    in_region = set()
    not_in_region = set()
    # Find the first pixel in remaining_pixels with a value of True
    available_pixels = np.transpose(np.nonzero(remaining_pixels))
    start = tuple(available_pixels[0])
    in_region.add(start)
    new_pixels = in_region.copy()
    while True:
        try_next = set()
        # Find surrounding pixels
        for pixel in new_pixels:
            x, y = pixel
            neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            try_next.update(neighbours)
        # Remove pixels we have already seen
        try_next.difference_update(segmented_pixels, not_in_region)
        # If there's no more pixels to try, the region is complete
        if not try_next:
            break
        # Empty new pixels set, so we know whose neighbour's to check next time
        new_pixels = set()
        # Check new pixels
        for pixel in try_next:
            if remaining_pixels[pixel]:
                in_region.add(pixel)
                new_pixels.add(pixel)
                segmented_pixels.add(pixel)
            else:
                not_in_region.add(pixel)
    return in_region


def _find_all_segments(pixels, segment_threshold, min_segment_size):
    """
    Finds all the regions within an image pixel array, and returns a list of the regions.

    Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
    rounding in the greyscale conversion.
    :param pixels: A np array of the pixel brightnesses.
    :param segment_threshold: The brightness threshold to use when differentiating between hills and valleys.
    :param min_segment_size: The minimum number of pixels for a segment.
    """
    img_width, img_height = pixels.shape
    # threshold pixels
    threshold_pixels = pixels > segment_threshold
    unassigned_pixels = np.full(pixels.shape, True, dtype=np.bool)

    segments = []
    already_segmented = set()

    # Add all the pixels around the border outside the image:
    already_segmented.update([(-1, z) for z in range(img_height)])
    already_segmented.update([(z, -1) for z in range(img_width)])
    already_segmented.update([(img_width, z) for z in range(img_height)])
    already_segmented.update([(z, img_height) for z in range(img_width)])

    # Find all the "hill" regions
    while np.bitwise_and(threshold_pixels, unassigned_pixels).any():
        remaining_pixels = np.bitwise_and(threshold_pixels, unassigned_pixels)
        segment = _find_region(remaining_pixels, already_segmented)
        # Apply segment
        if len(segment) > min_segment_size:
            segments.append(segment)
        for pix in segment:
            unassigned_pixels[pix] = False

    # Invert the threshold matrix, and find "valleys"
    threshold_pixels_i = np.invert(threshold_pixels)
    while len(already_segmented) < img_width * img_height:
        remaining_pixels = np.bitwise_and(threshold_pixels_i, unassigned_pixels)
        segment = _find_region(remaining_pixels, already_segmented)
        # Apply segment
        if len(segment) > min_segment_size:
            segments.append(segment)
        for pix in segment:
            unassigned_pixels[pix] = False

    return segments


def crop_resistant_hash(
    image,
    hash_func=phash,
    limit_segments=None,
    segment_threshold=128,
    min_segment_size=500,
    segmentation_image_size=300,
    **hashkwargs,
):
    """
    Creates a CropResistantHash object, by the algorithm described in the paper "Efficient Cropping-Resistant Robust
    Image Hashing". DOI 10.1109/ARES.2014.85
    This algorithm partitions the image into bright and dark segments, using a watershed-like algorithm, and then does
    an image hash on each segment. This makes the image much more resistant to cropping than other algorithms, with
    the paper claiming resistance to up to 50% cropping, while most other algorithms stop at about 5% cropping.

    Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
    rounding in the greyscale conversion. This leads to a slightly different result.
    :param image: The image to hash
    :param hash_func: The hashing function to use
    :param limit_segments: If you have storage requirements, you can limit to hashing only the M largest segments
    :param segment_threshold: Brightness threshold between hills and valleys. This should be static, putting it between
    peak and trough dynamically breaks the matching
    :param min_segment_size: Minimum number of pixels for a hashable segment
    :param segmentation_image_size: Size which the image is resized to before segmentation
    """
    orig_image = image.copy()
    # Convert to gray scale and resize
    image = image.convert("L").resize(
        (segmentation_image_size, segmentation_image_size), Image.ANTIALIAS
    )
    # Add filters
    image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
    pixels = np.array(image).astype(np.float32)

    segments = _find_all_segments(pixels, segment_threshold, min_segment_size)

    # If there are no segments, have 1 segment including the whole image
    if not segments:
        full_image_segment = {
            (0, 0),
            (segmentation_image_size - 1, segmentation_image_size - 1),
        }
        segments.append(full_image_segment)

    # If segment limit is set, discard the smaller segments
    if limit_segments:
        segments = sorted(segments, key=lambda s: len(s), reverse=True)[:limit_segments]

    # Create bounding box for each segment
    hashes = []
    for segment in segments:
        orig_w, orig_h = orig_image.size
        scale_w = float(orig_w) / segmentation_image_size
        scale_h = float(orig_h) / segmentation_image_size
        min_y = min(coord[0] for coord in segment) * scale_h
        min_x = min(coord[1] for coord in segment) * scale_w
        max_y = (max(coord[0] for coord in segment) + 1) * scale_h
        max_x = (max(coord[1] for coord in segment) + 1) * scale_w
        # Compute robust hash for each bounding box
        bounding_box = orig_image.crop((min_x, min_y, max_x, max_y))
        if hash_func is phash or hash_func is phash_simple:
            # if using phash, don't re-blur the segment because we already blurred
            # the entire image above
            hashes.append(hash_func(bounding_box, blur=False, **hashkwargs))
        else:
            hashes.append(hash_func(bounding_box, **hashkwargs))
        # Show bounding box
        # im_segment = image.copy()
        # for pix in segment:
        # 	im_segment.putpixel(pix[::-1], 255)
        # im_segment.show()
        # bounding_box.show()

    return ImageMultiHash(hashes)
