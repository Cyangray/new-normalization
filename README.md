Rewriting of the normalization part of the Oslo method, in a more compact, light and usable way.

It starts from the results of the rhosigchi program in the Oslo method software (https://github.com/oslocyclotronlab/oslo-method-software) and carries out the normalization, producing tables with the resulting NLD and GSF, together with properly propagated uncertainties.

Working example in "example_run_166Ho.py", see comments in code and in "normalization_class.py" to understand what's going on.

For the code to work, you need a functioning installation of the Oslo method software (https://github.com/oslocyclotronlab/oslo-method-software) substituting prog/counting.c and prog/normalization.c with the automatic versions in this repository. These are the same programs, but all input and output prompts are commented out, so that they will only read input from file, and write the output to file. If you already have the Oslo method software installed, redo an installation in another folder (calling it e.g. "oslo-method-software-auto"). IMPORTANT: copy the path to the newly compiled counting and normalization executables in the right place at the top of "normalization_class.py".
