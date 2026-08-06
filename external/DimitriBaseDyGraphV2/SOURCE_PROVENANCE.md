# Dimitri BaseDyGraph-V2 source snapshot

These four Python files are copied byte-for-byte from the BaseDyGraph-V2 tree
inside the supplied `KronosDyGraph` experiment archive. They are vendored under
a distinct directory so the dissertation's pinned `external/BaseDyGraph`
submodule remains unmodified.

The exact SHA-256 hashes checked by the project adapter are:

- `model.py`: `b99256db74b84f57513b12715a9ed1f4fc735202bcf933482e3b66ac9cf119d5`
- `modules.py`: `1bd31701b300f6f805dfc53530c66eedd9265620093223d3302e3e74409b51ff`
- `utilities.py`: `cfe849e2963386ddaab15ad7c89e7df92fc957f22c15f45ea40c89ec4c82f40a`
- `data_module.py`: `ecfbe768a1e2c0840043ecf640295b425106843f27062184307534312136cac1`

The source retains its original top-level imports (`model`, `modules`,
`utilities`, `data_module`). `src/models/dimitri_basedygraph_v2.py` imports it
inside an isolated training process and verifies all hashes before use.
