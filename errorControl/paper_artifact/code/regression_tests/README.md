# Regression tests recovered from Git

These tests and their headers were recovered from Git tree
`0385d7f52d240f150b9febf65fe921979952c554`. The quadrature and multipole
tests exercise orders used by the paper; the HNL tests check the historical
constructor against its integer-index implementation, source-cover
partitioning, family MAC classification, and MAC geometry.

Run:

```bash
make check
```

The `mac_geometry_test` target needs the AFEPack headers and `mpicxx`.
`TEST_RESULTS.txt` records the verified run included in the submission
package. Executables are deliberately excluded.
