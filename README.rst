===================
Cleanup Fork Branch
===================

Changes:

- uh, reformatted everything to look nicer, but now obviously nothing
  will merge against upstream. feel free to copy out or refactor sections
  back to other repos if they match your needs.
- added ``.save()`` interface for saving hashes as hex, base85, ascii85

  - ``saved = imagehash.phash(img).save("b85")``

- added restore capability directly to ImageHash and ImageMultiHash

  - ``imagehash.ImageHash(saved, restore="b85") == imagehash.phash(img)``

- added isometric/dihedral dct transforms from perception

  - ``imagehash.phash(img).isometric()``

- added saving all isometric hashes in a dict with one call

  - ``imagehash.phash(img).isometric(save="b85")``

- improved overall hex import/export infrastructure for single and multi-hashes

  - removed all previous custom hex generation import/export functions
  - hashes are assumed to be for square matrices by default
  - but, non-square hashes can have a prefix to their serialization for
    correct matrix re-creation when imported again

- added pdqhash and pdqhash isometric/dihedreal interface
- added the recommended 7x7 blur convolution to phash before resizing
- changed crop hash default hash to phash (also avoiding phash blur when called from crop hash)
- added pass-through args for crop hash segment hash function
- renamed numpy usage to standard np
- all numpy arrays are now integer (1/0) instead of boolean (True/False) for easier reading


Examples:

.. code-block:: python
 
    In [591]: imagehash.colorhash(i, binbits=66).save("hex")
    Out[591]: b'R14C66,1ffffffffffffff03ffffffffffffff80ffffffffffffffe000000000000000000000000000000000000000000000000000000000000000000000000000000000000003fffffffffffff000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

    In [578]: imagehash.colorhash(i, binbits=66).save("b85")
    Out[578]: b'R14C66,AOHXV|NsB+KmY&#|NsB^5C8xF|NsB~000000000000000000000000000000000000000000000000000000#|NsC0|Nj6000000000000000000000000000000000000000000000000000'

    In [579]: imagehash.colorhash(i, binbits=66).save("a85")
    Out[579]: b"R14C66,+92B@s8W,g5QCc`s8W,o&-)\\0s8W,uzzzzzzzzzz!!!!`s8W-!s8N'!zzzzzzzzzz"

    In [534]: imagehash.colorhash(i, binbits=66) == imagehash.ImageHash(b"R14C66,+92B@s8W,g5QCc`s8W,o&-)\\0s8W,uzzzzzzzzzz!!!!`s8W-!s8N'!zzzzzzzzzz", restore="a85", withPrefix=True)
    Out[534]: True

    In [580]: imagehash.colorhash(i, binbits=33).save("a85")
    Out[580]: b'R14C33,+92B@J,fQK^]4?6huE`Wzzzz!!#7`rVuouzzzz!!!'

    In [581]: imagehash.colorhash(i, binbits=8).save("a85")
    Out[581]: b"R14C8,*WH'=zz!!!"

    In [582]: imagehash.colorhash(i, binbits=16).save("a85")
    Out[582]: b"R14C16,+9)<:s8N'!zz!'^G`zz"

    In [583]: imagehash.colorhash(i, binbits=16).save("hex")
    Out[583]: b'R14C16,1ffefffaffff00000000000000000000003f00000000000000000000'

    In [584]: imagehash.colorhash(i, binbits=3).save("hex")
    Out[584]: b'R14C3,1b8000000000'

    In [585]: imagehash.colorhash(i, binbits=3).save("a85")
    Out[585]: b'R14C3,)h7ng!!!'

    In [586]: imagehash.colorhash(i, binbits=16).save("a85")
    Out[586]: b"R14C16,+9)<:s8N'!zz!'^G`zz"

    In [587]: imagehash.colorhash(i, binbits=16).save("b85")
    Out[587]: b'R14C16,AO8RP|Nj60000000000006zc#0000000000'

    In [567]: imagehash.phash(i, hash_size=8).isometric("b85")
    Out[567]:
    {'r0': b'{FGspVW!4s',
     'fv': b'{A*z<X;{aY',
     'fh': b'slznEJoA?_',
     'r180': b'sXjD6JPeY}',
     'r90': b')>N%8h$=_c',
     'r90fv': b')~l`1fY%2(',
     'r90fh': b'g8}$@-F;G#',
     'r270': b'gYNi+-GWkT'}

    In [568]: imagehash.phash(i, hash_size=9).isometric("b85")
    Out[568]:
    {'r0': b'{7P7Iww*H=&V~R',
     'fv': b'{56<mMw-VNG1dS',
     'fh': b'sbI}GGtZh%WYYi',
     'r180': b'sUHm|<H&AFn}Yx',
     'r90': b')~aho_<=wlt$F|',
     'r90fv': b'*43+82!^~Ka(Dm',
     'r90fh': b'gMhvdP1EU1D=h#',
     'r270': b'gZF+Az0?&-&@BJ'}

    In [569]: imagehash.phash(i, hash_size=16).isometric("b85")
    Out[569]:
    {'r0': b'{2r92VcwOoY2c?P$K_`%)+KE(MqG|A3_Ogq3}U=+',
     'fv': b'{2psqVcu#^Y2aAc$K{yP)+LhAPF!)(Og?i<Ok+cj',
     'fh': b'sY}7~G>gFGJhk$2lCUy)gK#){8wuNc8Ee_;88_)I',
     'r180': b'sY^ahG>d;5Jhct2lCaH(gK)=%8)+nzS!*^PSvMBi',
     'r90': b')|FHhtwAq-h<_*m2e81udIZ(nU*AvCp?w$w<XWje',
     'r90fv': b')|IQ^twGR%h+o(L2e3atdITyXU*E7Tp?K%~<Wg3^',
     'r90fh': b'gTN6=_!N3C-70-mQstHEFHyj$4yqcL^es(Ru@C)f',
     'r270': b'gTVg3_!NcI-712rQsrtDFHwJX3X0`;^ewWfu@3^5'}

    In [570]: imagehash.phash(i, hash_size=16).isometric("hex")
    Out[570]:
    {'r0': b'fc1e94a761de95b169e0a727c7e5672cd6256d2f465c8e2e0c3c8cb40c62bc70',
     'fv': b'fc1e6b5861de6a4e69e058d8c7e598d2d62592d04e5c71d14c3e734b4c63438f',
     'fh': b'a94bc1f2348bc0e43cb5f27292b032798370387a1b09db7b196bd9e91937e92d',
     'r180': b'a94b3e4d348b7f1b3cb50dad92b0cd878370c7851b692494596b361e593716da',
     'r90': b'd6955415ad412f7e887f280007b0c0bf7a04d5dc5fdf4fd2a17d1803e45aa93f',
     'r90fv': b'd695abe0ad41d081885fd7ff07b03f407a042a235fdfb02da178e7fce45256c0',
     'r90fh': b'83c0114af8147a2fdd2a7d5552e595ea2f51c0a90eaa1a97f42d4d56b10ffd6a',
     'r270': b'83c0febff81485d4dd2a82aa52e56a152f517f760a8ae578f42db2a9b10f0297'}

    In [572]: imagehash.pdqhash_isometric(i,"hex")
    Out[572]:
    {'r0': b'131f871e231896981e183a389d317a5bd2355a73d3f1f07283cbc6d4bdc3f294',
     'r90': b'72bffe0a2513600c4f42a5f9fdfd1dd5102ffe8106f0000aff083f7ac15a7c15',
     'r180': b'c64a2db4f64d3c364b4d9092ca6cd0f18760f0d986a45ad8d69e6c7ee896593e',
     'r270': b'07ea54a07046caa65a178f53a8a8b77e457a542b53a5aaa0aa5d95d0940ffebf',
     'fv': b'171f78e1231969671e19c5c79d3985a4d235a58cd3f10d8d83cb392bbdc30c6b',
     'fh': b'464ada4bf64dc3cd4b4d6f6dc86c2f0e87600f2686a4a727d69e9381e896a7c1',
     'r90fv': b'52bf01f525139ff30f425a06fdede22a102f017e06f0fff5fd08c085c15a83ea',
     'r90fh': b'a7eaab5f704635595a17f0acaab84881457aabd453a5555faa5d7a2f940f2940'}

    In [573]: imagehash.pdqhash_isometric(i,"b85")
    Out[573]:
    {'r0': b'6CZ~jBN&#L9vC_}oiTb_(luIh)A8_fgUiO$y~Fa9',
     'r90': b'a=-oxB@<u_PeP^n{rw%)5HJ3L2Jiq1{|G;N!CHJ3',
     'r180': b'#!4-;_DwuCOHGiH%52c_hhXs8hNN29)}CyB=$2VN',
     'r270': b'2kKOya7N0eS{IK~sHnGoMS4^#Q>Ch)s$G@Pln?&D',
     'fv': b'7aw@xBN=ID9vQ{QojHZ1(lw=w)A0?BgUdN9y~7M^',
     'fh': b'MoQXC_D#dhOHFTW$ZRhThhPsThNPz_)}E7r=$5C!',
     'r90fv': b'QojN9B@>_X4?<c7{q5o^5HA6K2Jrv&{RqH?!CHgr',
     'r90fh': b'r|PR;a7Hy*S{LxFs<=pjMS82$Q>9g3s$F_7ln*ID'}


===========
ImageHash
===========

An image hashing library written in Python. ImageHash supports:

* Average hashing
* Perceptual hashing
* Difference hashing
* Wavelet hashing
* HSV color hashing (colorhash)
* Crop-resistant hashing

|Travis|_ |Coveralls|_

Rationale
=========

Image hashes tell whether two images look nearly identical.
This is different from cryptographic hashing algorithms (like MD5, SHA-1)
where tiny changes in the image give completely different hashes. 
In image fingerprinting, we actually want our similar inputs to have
similar output hashes as well.

The image hash algorithms (average, perceptual, difference, wavelet)
analyse the image structure on luminance (without color information).
The color hash algorithm analyses the color distribution and 
black & gray fractions (without position information).

Installation
============

Based on PIL/Pillow Image, numpy and scipy.fftpack (for pHash)
Easy installation through `pypi`_::

	pip install imagehash

Basic usage
===========
::

	>>> from PIL import Image
	>>> import imagehash
	>>> hash = imagehash.average_hash(Image.open('test.png'))
	>>> print(hash)
	d879f8f89b1bbf
	>>> otherhash = imagehash.average_hash(Image.open('other.bmp'))
	>>> print(otherhash)
	ffff3720200ffff
	>>> print(hash == otherhash)
	False
	>>> print(hash - otherhash)
	36
    >>> for r in range(1, 30, 5):
    ...     rothash = imagehash.average_hash(Image.open('test.png').rotate(r))
    ...     print('Rotation by %d: %d Hamming difference' % (r, hash - rothash))
    ...
    Rotation by 1: 2 Hamming difference
    Rotation by 6: 11 Hamming difference
    Rotation by 11: 13 Hamming difference
    Rotation by 16: 17 Hamming difference
    Rotation by 21: 19 Hamming difference
    Rotation by 26: 21 Hamming difference

Each algorithm can also have its hash size adjusted (or in the case of
colorhash, its :code:`binbits`). Increasing the hash size allows an
algorithm to store more detail in its hash, increasing its sensitivity
to changes in detail.

The demo script **find_similar_images** illustrates how to find similar
images in a directory.

Source hosted at GitHub: https://github.com/JohannesBuchner/imagehash

References
-----------

* Average hashing (`aHashref`_)
* Perceptual hashing (`pHashref`_)
* Difference hashing (`dHashref`_)
* Wavelet hashing (`wHashref`_)
* Crop-resistant hashing (`crop_resistant_hashref`_)

.. _aHashref: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _pHashref: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _dHashref: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
.. _wHashref: https://fullstackml.com/2016/07/02/wavelet-image-hash-in-python/
.. _pypi: https://pypi.python.org/pypi/ImageHash
.. _crop_resistant_hashref: https://ieeexplore.ieee.org/document/6980335

Examples
=========

To help evaluate how different hashing algorithms behave, below are a few hashes applied
to two datasets. This will let you know what images an algorithm thinks are basically identical.

Example 1: Icon dataset
-----------------------

Source: 7441 free icons on GitHub (see examples/github-urls.txt).

The following pages show groups of images with the same hash (the hashing method sees them as the same).

* `phash <https://johannesbuchner.github.io/imagehash/art3.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/art9.html>`__)
* `dhash <https://johannesbuchner.github.io/imagehash/art4.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/art10.html>`__)
* `colorhash <https://johannesbuchner.github.io/imagehash/art7.html>`__
* `average_hash <https://johannesbuchner.github.io/imagehash/art2.html>`__ (`with z-transform <https://johannesbuchner.github.io/imagehash/art8.html>`__)

The hashes use hashsize=8; colorhash uses binbits=3.
You may want to adjust the hashsize or require some manhattan distance (hash1 - hash2 < threshold).

Example 2: Art dataset
----------------------

Source: 109259 art pieces from http://parismuseescollections.paris.fr/en/recherche/image-libre/.

The following pages show groups of images with the same hash (the hashing method sees them as the same).

* `phash <https://johannesbuchner.github.io/imagehash/index3.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/index9.html>`__)
* `dhash <https://johannesbuchner.github.io/imagehash/index4.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/index10.html>`__)
* `colorhash <https://johannesbuchner.github.io/imagehash/index7.html>`__
* `average_hash <https://johannesbuchner.github.io/imagehash/index2.html>`__ (`with z-transform <https://johannesbuchner.github.io/imagehash/index8.html>`__)

For understanding hash distances, check out these excellent blog posts:

* https://tech.okcupid.com/evaluating-perceptual-image-hashes-okcupid/
* https://content-blockchain.org/research/testing-different-image-hash-functions/

Changelog
----------

* 4.2: Cropping-Resistant image hashing added by @joshcoales

* 4.1: Add examples and colorhash

* 4.0: Changed binary to hex implementation, because the previous one was broken for various hash sizes. This change breaks compatibility to previously stored hashes; to convert them from the old encoding, use the "old_hex_to_hash" function.

* 3.5: Image data handling speed-up

* 3.2: whash now also handles smaller-than-hash images

* 3.0: dhash had a bug: It computed pixel differences vertically, not horizontally.
       I modified it to follow `dHashref`_. The old function is available as dhash_vertical.

* 2.0: Added whash

* 1.0: Initial ahash, dhash, phash implementations.

Contributing
=============

Pull requests and new features are warmly welcome.

If you encounter a bug or have a question, please open a GitHub issue. You can also try Stack Overflow.

Other projects
==============

* http://blockhash.io/
* https://github.com/acoomans/instagram-filters
* https://pippy360.github.io/transformationInvariantImageSearch/
* https://www.phash.org/
* https://pypi.org/project/dhash/
* https://github.com/thorn-oss/perception (based on imagehash code, depends on opencv)
* https://docs.opencv.org/3.4/d4/d93/group__img__hash.html

.. |Travis| image:: https://travis-ci.com/JohannesBuchner/imagehash.svg?branch=master
.. _Travis: https://travis-ci.com/JohannesBuchner/imagehash

.. |Coveralls| image:: https://coveralls.io/repos/github/JohannesBuchner/imagehash/badge.svg
.. _Coveralls: https://coveralls.io/github/JohannesBuchner/imagehash
