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
project maintainers approved public publication of the converted artifact at
immutable Hugging Face revision
`65af18f49af5101267fd28f15ac8c452d98b8e3d`; the published artifact ships this
notice. This source and license evidence does not provide comprehensive
training-data provenance. Binary redistributions must reproduce the copyright
notice, conditions, and disclaimer in their documentation or other accompanying
materials.

TuneForge's MIT license does not relicense Crema, its model files, or conversion
output derived from them.

The reproducible conversion pipeline preserves the verified wheel as its source
boundary and records the embedded H5 SHA-256 as
`08b80e5b648e743c89284e9bc0b12b993dad1129157a75e0de70e076b0b8a235`.
Crema does not ship a machine-readable per-track training manifest, so the
converted artifact's model card states that limitation and does not claim
TuneForge trained the model or owns upstream datasets.

## Beat This 1.1.0 small0

- Project: `beat-this`
- Version: `1.1.0`; upstream tag commit:
  `ad7974846029835307ba19a3d5cefbf40b243041`
- Upstream: <https://github.com/CPJKU/beat_this>
- Authors: Francesco Foscarin, Jan Schlüter, and Gerhard Widmer
- Copyright: Copyright (c) 2024 Institute of Computational Perception, JKU Linz,
  Austria
- License text: [MIT](LICENSES/beat-this-1.1.0-MIT.txt)
- PyPI source SHA-256:
  `3017c741f972972a650edcaccfe5760687fe4f5587feaa98896d90f866c2435c`
- PyPI wheel SHA-256:
  `3f2b2d1e027c6dac380bf80c71555e3c28a4036a7f1af20129a945915a72a645`
- `small0` checkpoint: 8,451,101 bytes; SHA-256
  `6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b`

The tagged package defines the authoritative checkpoint URL and offers `small0`
for download and reuse. Its shipped MIT file is preserved byte-for-byte here.
Current upstream documentation explicitly states that published model weights are
MIT licensed; that explicit weights sentence was not present in the tagged 1.1.0
README. This repository records both facts and does not infer training-dataset
rights from the model license.

Upstream documents training across multiple datasets and publishes annotations
and spectrogram resources separately. It does not ship one complete
machine-readable per-track training manifest with `small0`. Some training audio
has separate copyright or Creative Commons restrictions. TuneForge does not claim
ownership of training material or broader rights to those datasets.

## ExecuTorch 1.4.0

- Project: `ExecuTorch`
- Version: `1.4.0`; upstream tag commit:
  `3dd7ccd1d863fad22639dd2d918ae34a41ce45f0`
- Upstream: <https://github.com/pytorch/executorch>
- Android AAR: `org.pytorch:executorch-android:1.4.0`; 7,318,312 bytes; SHA-256
  `a4a836b9fadd5b9afdf07b8533b2c3326695d04a58dd37e2cdfe709e804854fb`
- License text:
  [BSD 3-Clause](LICENSES/executorch-1.4.0-BSD-3-Clause.txt)

The PTE does not embed the Android AAR. Release metadata pins the consumer runtime
identity. TuneForge application packaging supplies that runtime separately and must
preserve notices for the AAR and its resolved dependencies, including fbjni,
NativeLoader, AndroidX Core, and Kotlin standard library. This record does not replace
that app packaging review.
