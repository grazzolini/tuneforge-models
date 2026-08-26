# Third-Party Notices

This file records third-party licensing and provenance reviewed for this
repository. It does not authorize publication by itself.

## Crema 0.2.0

- Project: `crema`
- Version: `0.2.0`
- Upstream: <https://github.com/bmcfee/crema>
- Upstream tag commit: `051c91697fd16856a0a1019cc06ee1f11fb52c5f`
- PyPI wheel: `crema-0.2.0-py3-none-any.whl`
- Wheel SHA-256:
  `b2787afd0367463438ca2b9b2944c490308eee1f307e5796078ec540a6281484`
- Copyright: Copyright (c) 2017, Brian McFee
- License text: [BSD 2-Clause](LICENSES/crema-0.2.0-BSD-2-Clause.txt)

The upstream `LICENSE.md`, source distribution `LICENSE.md`, and wheel
`crema-0.2.0.dist-info/LICENSE.md` are byte-identical and state the BSD
2-Clause License. The wheel and source distribution include the pretrained
chord model (`model.h5`) and supporting serialized files under
`crema/models/chord/`.

PyPI package metadata and `setup.cfg` label Crema 0.2.0 as ISC. That metadata
conflicts with the actual license file, the upstream repository's detected
SPDX license (`BSD-2-Clause`), and the project's BSD badge. This repository
therefore preserves the exact license text shipped in the verified wheel,
without rewriting it as ISC.

No separate license or model-file exclusion appears in the wheel, source
distribution, or upstream tag. Crema's packaging configuration includes the
model files as package data, and the distribution-level BSD 2-Clause notice is
the applicable license evidence shipped with those files. That evidence does
not document the model's full training-data provenance or separately resolve
every right that may be relevant to derived or converted weights. TuneForge
project maintainers approved the converted artifact's private publication for
controlled evaluation; the approved artifact ships this notice. That approval
does not establish public redistribution authority or comprehensive legal
clearance. Any approved binary redistribution must reproduce the copyright
notice, conditions, and disclaimer in its documentation or other accompanying
materials.

TuneForge's MIT license does not relicense Crema, its model files, or conversion
output derived from them.

The reproducible conversion pipeline preserves the verified wheel as its source
boundary and records the embedded H5 SHA-256 as
`08b80e5b648e743c89284e9bc0b12b993dad1129157a75e0de70e076b0b8a235`.
Crema does not ship a machine-readable per-track training manifest, so the
converted artifact's model card states that limitation and does not claim
TuneForge trained the model or owns upstream datasets.
