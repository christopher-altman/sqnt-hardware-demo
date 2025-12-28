# sqnt-hardware-demo

*A minimal, runnable *bridge artifact connecting the theory line from superpositional quantum network topologies and adaptive quantum networks*

<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
[![Hugging Face](https://img.shields.io/badge/huggingface-Cohaerence-white)](https://huggingface.co/Cohaerence)

[![X](https://img.shields.io/badge/X-@coherence-blue)](https://x.com/coherence)
[![Website](https://img.shields.io/badge/website-christopheraltman.com-green)](https://www.christopheraltman.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Altman-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/Altman)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

<br>

## Terminology

**SQNT** – *Superpositional Quantum Network Topologies*
**AQN** – *Adaptive Quantum Networks*

## Lineage

- **Superpositional Quantum Network Topologies** (IJTP 2004)
- **Backpropagation in Adaptive Quantum Networks** (IJTP 2010)
- **Accelerated Training Convergence in Superposed Quantum Networks** (NATO ASI)

<br>

### Quickstart (one command)

```bash
python3 -m venv .venv && source .venv/bin/activate && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install numpy matplotlib && PYTHONPATH=src .venv/bin/python scripts/make_figures.py
```

## What this repo demonstrates (v1)

1. Construct a small **family of graph topologies** (chain, ring, star, complete).
2. Define an **operator-space weight matrix** \(W\) and “spatialize” it onto each graph via a topology mask.
3. Form a **superposed topology** by mixing masks with a single mixture parameter \(\alpha\).
4. Train a tiny model on a synthetic task and output a single canonical figure:

**`figures/sqnt_mixture_curve.png`**: accuracy vs topology mixture parameter.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy matplotlib
PYTHONPATH=src .venv/bin/python scripts/make_figures.py
```


---

## References

1. C. Altman, J. Pykacz & R. Zapatrin, “Superpositional Quantum Network Topologies,” *International Journal of Theoretical Physics* 43, 2029–2041 (2004).  
   DOI: [10.1023/B:IJTP.0000049008.51567.ec](https://doi.org/10.1023/B:IJTP.0000049008.51567.ec) · arXiv: [q-bio/0311016](https://arxiv.org/abs/q-bio/0311016)

2. C. Altman & R. Zapatrin, “Backpropagation in Adaptive Quantum Networks,” *International Journal of Theoretical Physics* 49, 2991–2997 (2010).  
   DOI: [10.1007/s10773-009-0103-1](https://doi.org/10.1007/s10773-009-0103-1) · arXiv: [0903.4416](https://arxiv.org/abs/0903.4416)

3. S. Alexander, W. J. Cunningham, J. Lanier, L. Smolin, S. Stanojevic, M. W. Toomey & D. Wecker, “The Autodidactic Universe,” arXiv (2021).  
   DOI: [10.48550/arXiv.2104.03902](https://doi.org/10.48550/arXiv.2104.03902) · arXiv: [2104.03902](https://arxiv.org/abs/2104.03902)

---

## Citations

If you use or build on this work, please cite:

> SQNT Hardware Demonstration – Adaptive Quantum Networks
```bibtex
@software{altman2025sqnt_hardware_demo,
  author  = {Christopher Altman},
  title   = {sqnt-hardware-demo: SQNT Hardware Demonstration – Adaptive Quantum Networks},
  year    = {2025},
  version = {0.1.0},
  url     = {https://github.com/christopher-altman/sqnt-hardware-demo},
}
```
---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- **Website:** [christopheraltman.com](https://christopheraltman.com)
- **Research portfolio:** https://lab.christopheraltman.com/
- **Portfolio mirror:** https://christopher-altman.github.io/
- **GitHub:** [github.com/christopher-altman](https://github.com/christopher-altman)
- **Google Scholar:** [scholar.google.com/citations?user=tvwpCcgAAAAJ](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
- **Email:** x@christopheraltman.com

---

*Christopher Altman (2025)*
