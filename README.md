# 21cmCAST

This package provides tools to perform Fisher fore**CAST**s from 21cmFAST outputs. It has been built on / complementarily to the [21cmFish](https://21cmfish.readthedocs.io/en/latest/) package[^1]. A detailed example of how it works can be found in [**scripts/exec/fisher.ipynb**](./scripts/exec/fisher.ipynb). 

In order to work properly, the codes requires to have a working installation of
- [**21cmFAST**](https://21cmfast.readthedocs.io/en/latest/)[^2][^3]
- [**21cmSense**](https://21cmsense.readthedocs.io/en/latest/)[^4]

If you are considering using this code for you own work, please cite
- G. Facchinetti, L. Lopez-Honorez, A. Mesinger, Y. Qin (in prep.)

To do list:
- [x] rearrange the code in a self-contained package
- [ ] uniformely use astropy units throughout the entire code
- [ ] add docstrings to the classes

[^1]: *Charlotte A. Mason, Julian B. Muñoz, Bradley Greig, Andrei Mesinger, and Jaehong Park*, 21cmfish: Fisher-matrix framework for fast parameter forecasts from the cosmic 21-cm signal [[arXiv:2212.09797]](https://arxiv.org/abs/2212.09797)

[^2]: *Andrei Mesinger, Steven Furlanetto, and Renyue Cen*, 21cmFAST: A Fast, Semi-Numerical Simulation of the High-Redshift 21-cm Signal [[arXiv:1003.3878]](https://arxiv.org/abs/1003.3878)

[^3]: *Andrei Mesinger and Steven Furlanetto*, Efficient Simulations of Early Structure Formation and Reionization [[arXiv:0704.0946]](https://arxiv.org/abs/0704.0946)

[^4]: *Jonathan C. Pober, Adrian Liu, Joshua S. Dillon, James E. Aguirre, Judd D. Bowman et al.*, What Next-Generation 21 cm Power Spectrum Measurements Can Teach Us About the Epoch of Reionization [[arXiv:1310.7031]](https://arxiv.org/abs/1310.7031)
