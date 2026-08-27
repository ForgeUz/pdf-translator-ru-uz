# End User License Agreement (EULA)

**pdf-translator-ru-uz** — Copyright (C) 2026 ForgeUz

This document supplements the [GNU Affero General Public License v3](LICENSE)
(the "AGPLv3") that governs this software. It explains the licensing
implications of the third-party libraries this project depends on, and the
conditions under which you may use, modify, and distribute the software.

---

## 1. Governing License

The source code of this project is licensed under the **GNU Affero General
Public License v3 (AGPLv3)**. A full copy of the license is provided in the
[`LICENSE`](LICENSE) file.

By using, copying, modifying, or distributing this software, you agree to be
bound by the terms of the AGPLv3 and this EULA. If you do not agree, you may
not use the software.

## 2. Third-Party Library Licenses

This project links against third-party libraries, each with its own license.
**You are responsible for complying with the licenses of all libraries you
use.** The most important ones are listed below.

### 2.1 PyMuPDF (fitz) — AGPL v3 (Copyleft)

- **License:** GNU Affero General Public License v3
- **Impact:** PyMuPDF is the core PDF parsing and rendering engine used by
  this project. Because PyMuPDF is licensed under the **AGPL v3**, any
  software that links to it is generally considered a derivative work and
  must also be released under the AGPL v3.
- **Consequence for this project:** This is the primary reason this project
  is licensed under the AGPL v3.
- **Commercial use:** If you wish to distribute this software (or a
  derivative) as **closed-source / proprietary** software — for example, to
  sell it to government agencies or enterprises as a proprietary product —
  you **must obtain a separate commercial license from the PyMuPDF
  maintainers** (Artifex Software). The AGPL v3 does not permit closed-source
  distribution.

### 2.2 Poppler / pdftotext — GPL v2/v3 (copyleft)

If you use Poppler-based tools, note that they are GPL-licensed and impose
similar copyleft obligations. This project does **not** depend on Poppler by
default.

### 2.3 Permissive libraries (safe for closed-source)

The following libraries are permissive (MIT / BSD / Apache-2.0) and do **not**
impose copyleft obligations:

| Library | License | Purpose |
|---------|---------|---------|
| `pypdf`, `pdfplumber`, `pdfminer.six` | MIT / BSD | PDF parsing (alternatives to PyMuPDF) |
| `transformers` (HuggingFace) | Apache-2.0 | NLLB model loading |
| `ctranslate2` | MIT | CPU-accelerated inference |
| `fasttext` | MIT | Language ID detection |
| `sacrebleu` | Apache-2.0 | Translation quality scoring |
| `psutil` | BSD | Memory monitoring |

> **Note:** If you need a **closed-source** distribution, you can replace
> PyMuPDF with a permissive alternative (e.g. `pypdf` / `pdfplumber`) and
> re-license your own code accordingly. This project's own code is AGPLv3,
> but the AGPL obligation is triggered by the PyMuPDF dependency.

## 3. Model Licenses

The translation model (`facebook/nllb-200-distilled-600M`) is distributed by
Meta under the **CC-BY-NC-4.0** license for non-commercial use. If you intend
to use the model commercially, review Meta's model license terms and obtain
the appropriate rights.

## 4. No Warranty

This software is provided **"AS IS"**, without warranty of any kind, express
or implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement. In no event shall the
authors or copyright holders be liable for any claim, damages, or other
liability arising from, out of, or in connection with the software or the use
or other dealings in the software.

## 5. Experimental Status

This is an **early-stage, experimental** project. Translation quality is not
production-grade and may contain errors, hallucinations, or structural
corruption (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for a detailed
audit). You are solely responsible for verifying the accuracy of any output
before relying on it.

## 6. Contact

For commercial licensing inquiries, please open an issue on the GitHub
repository: <https://github.com/ForgeUz/pdf-translator-ru-uz>