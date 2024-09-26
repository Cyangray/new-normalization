Rewriting of the normalization part of the Oslo method, in a more compact, light and usable way.

It starts from the results of the rhosigchi program in the Oslo method software (https://github.com/oslocyclotronlab/oslo-method-software) and carries out the normalization, producing tables with the resulting NLD and GSF, together with properly propagated uncertainties.
There is also the possibility to run TALYS and propagate these uncertainties to the neutron-capture rates and/or Maxwellian-averaged cross sections (MACSs).

Working examples in "example_run_166Ho.py" and "example_run_167Ho.py", use these as tutorials and read the comments in these codes and in "normalization_class.py" to understand what's going on.

In this version of the code, everything is (re)written in Python and sped up with both parallelization and the Numba package, rending it as fast as the one including the code in C from the Oslo method software. This means that no parallel installation of the OMS or compiling of C code will be necessary, just the installation of the necessary Python packages.

Similarly, if one wants to use the TALYS functionality, one has to have a working installation of TALYS. Whenever before using any of the methods where TALYS is involved, specify the path to the TALYS root folder, and its version. For now, only TALYS 1.96 and TALYS 2.00 are supported.
