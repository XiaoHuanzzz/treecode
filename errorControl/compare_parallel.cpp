// Compatibility entry point for comparison against a binary uniform-mesh
// reference. The complete shared implementation lives in compare_exact.cpp;
// selecting the mode here keeps both executables numerically synchronized.
#define TREECODE_COMPARE_REFERENCE 1
#include "compare_exact.cpp"
