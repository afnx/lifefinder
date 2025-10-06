<div align="center">

# LifeFinder

### Exoplanet Life Probability Calculator

*Estimate life potential for exoplanets with neural networks and astrophysical data.*

[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://docker.com/)

</div>

---

**LifeFinder** is a Python package and tool to estimate the probability of life on exoplanets using astrophysical parameters and machine learning. It includes a built-in neural network model trained to predict a habitability probability based on stellar and planetary features.

## Installation

You can install LifeFinder via pip:

```bash
pip install lifefinder
```

Or clone the repository and use Docker for containerized environment:

```bash
git clone https://github.com/afnx/lifefinder.git

docker compose up -d
```

## Usage

Before starting to use LifeFinder, configure necessary parameters such as default artifact locations, model settings etc:
```bash
lifefinder configure
```

**Once configured, you could start training models:**

```bash
lifefinder train
```
LifeFinder downloads the latest exoplanet data from the NASA Exoplanet Archive, cleans it, and trains the neural network model.

To retrain an existing model with different hyperparameters, just add the `--retrain` flag:
```bash
lifefinder train --retrain
```

For a full list of commands and options, run:
```bash
lifefinder --help
```

## How It Works

**LifeFinder** does NOT use a classical ground truth for habitability. Instead, it generates a soft, continuous "habitability zone index" (HZ index) for model training, based on the stellar insolation flux received by a planet.

#### How HZ Index is Computed

- **pl_insol**: Stellar insolation flux (in Earth units, S⊕).  
  - Earth = 1.0 S⊕.
- **sigma**: Controls how quickly the index decays as you move away from Earth's flux.  
  - Default: 0.5 (roughly the half-width of the conservative habitable zone).

**Formula:**
```python
import numpy as np
def hz_index(pl_insol, sigma=0.5):
    return np.exp(-0.5 * ((pl_insol - 1.0) / sigma) ** 2)
```
- Planets near Earth's flux (pl_insol ≈ 1.0) have HZ index ≈ 1.0.
- Planets far from Earth's flux get HZ index ≈ 0.0.
- This creates a smooth, interpretable target for machine learning.

## Machine Learning Model

**LifeFinder** uses a neural network regressor built with PyTorch.

- **Inputs:** Astrophysical features (e.g., stellar flux, planet radius, orbital period, etc.)
- **Output:** Planet habitability (continuous, 0.0–1.0)
- **Architecture:** Feedforward neural network with:
  - Multiple dense layers
  - Nonlinear activation functions (e.g., ReLU, sigmoid)
  - Regularization (dropout, L2, etc.)
- **Training:** Supervised learning using the generated HZ index as the ground truth.

## Disclaimer

The outputs and predictions provided by LifeFinder do **not** represent the actual habitability of any exoplanet, nor do they guarantee the presence or absence of life. The model is designed to estimate a probability of habitability based on available astrophysical data and machine learning techniques. The accuracy and reliability of the results depend strongly on the quality, quantity, and completeness of the input data, as well as the modeling approaches used.

**Do not rely solely on LifeFinder's output for scientific conclusions, mission decisions, or any application where actual confirmation of exoplanet habitability is required.** The results should always be interpreted as probabilistic estimates and be used in combination with other scientific methods and expert analysis.

## Contributing

We welcome your contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- [Kopparapu et al., 2013, ApJ, "Habitable Zones Around Main-Sequence Stars: New Estimates"](https://iopscience.iop.org/article/10.1088/0004-637X/765/2/131)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## License

This project is licensed under the **Business Source License 1.1** - see the [LICENSE](LICENSE) file for details.

## Authors

- **Ali Fuat Numanoglu** - *Initial work* - [@afnx](https://github.com/afnx)

## Support

- 🐛 [Report Issues](https://github.com/afnx/dream-job/issues)
- 📧 [Contact Me](https://alifuatnumanoglu.com/contact)
- ⭐ **If this project helped you, please give it a star!**

---

<sub>Made with ❤️ by Ali Fuat Numanoglu</sub>
