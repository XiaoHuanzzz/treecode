// Validation-only entry point. All mesh, HNL/MAC, quadrature, moment, and
// tree-evaluation routines are the unchanged production implementation.
#define TREECODE_ERROR_VALIDATION 1
#define TREECODE_INDICATOR_ACCURACY 6
#include "compare_exact.cpp"
