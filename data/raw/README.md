# Challenge inputs

The challenge statement PDF and participant workbook used by the reproducibility scripts belong in this directory. Their source-specific redistribution and publication terms have not been established in this repository. Before making a public GitHub repository, confirm that the challenge organizers permit redistribution of these input files. If redistribution is not permitted, remove the files from the public Git history and document their local placement here; `.gitignore` alone does not remove already tracked objects.

The workbook includes multiple sheets. The code's leakage boundaries are stricter than simply avoiding a test target column: Stage 1 explicitly reads only `Stage1_Train`; the Stage 2 data loader reads only `Stage2_Data` and `Stage2_Constraints`; no benchmark selects or reads `Stage1_Test`.
