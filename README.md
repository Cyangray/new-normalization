Rewriting of the normalization part of the Oslo method, in a more compact, light and usable way.

It starts from the results of the rhosigchi program in the Oslo method software (https://github.com/oslocyclotronlab/oslo-method-software) and carries out the normalization, producing tables with the resulting NLD and GSF, together with properly propagated uncertainties.
There is also the possibility to run TALYS and propagate these uncertainties to the neutron-capture rates and/or Maxwellian-averaged cross sections (MACSs).

Working examples in "example_run_166Ho.py" and "example_run_167Ho.py", use these as tutorials and read the comments in these codes and in "normalization_class.py" to understand what's going on.

For the code to work, you need a functioning installation of the Oslo method software (https://github.com/oslocyclotronlab/oslo-method-software), substituting prog/counting.c and prog/normalization.c with the automatic versions in this repository. These are the same programs, but with all input and output prompts commented out, so that they will only read input from file, and write the output to file. If you already have the Oslo method software installed, redo an installation in another folder (calling it e.g. "oslo-method-software-auto"). This path will be then used when the "normalization" class is initialized.

Similarly, if one wants to use the TALYS functionality, one has to have a working installation of TALYS. Whenever before using any of the methods where TALYS is involved, specify the path to the TALYS root folder, and its version. For now, only TALYS 1.96 and TALYS 2.00 are supported.
