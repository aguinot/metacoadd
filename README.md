# Metacoadd

Create a coadd image at the same time as we create sheared images for the [metadetection][metadetect].

[metadetect]: https://github.com/esheldon/metadetect

## Installation

```python
python -m pip install git+https://github.com/aguinot/metacoadd
```

## Details

Steps:

- [ ] Create sub field for the coadd region ([step 1](#i---create-subfield))
- [ ] Interpolate ([step 2](#ii---interpolation))
- [ ] Deconvolve the PSF ([step 3](#iii---deconvolve-from-psf))
- [ ] Deconvolve from the pixels
- [ ] Apply shear
- [ ] Get larger reconv PSF
- [ ] Reconvolve by PSF
- [ ] Apply coadd WCS
- [ ] Run Detection
- [ ] Run shape measurement


### I - Create subfield

The coadd size is not yet defined use 1 arcmin for testing.

Idea is to make use of the Galsim `bound` method to create cutout in the single exposures for the coadd area.

Question:
1) How to handle borders?
2) What if we want to mask?
3) How to hadle grid difference between exposure and coadd?
4) Does it need to be an interpolated image at this stage?


### II - Interpolation

We would like to use `lanczosX` interpolation to get as close as possible to
standar techniques used in UNIONS (through SWARP) (found a good value for `X`).


### III - Deconvolve from PSF

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
