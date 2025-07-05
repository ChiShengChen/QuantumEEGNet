# QuantumEEGNet

## QEEGNet: Quantum Machine Learning for Enhanced Electroencephalography Encoding (IEEE SiPS 2024)
[![arXiv](https://img.shields.io/badge/arXiv-2407.19214-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.19214)  
[IEEE SiPS 2024](https://ieeexplore.ieee.org/abstract/document/10768221/)
 

![image](https://github.com/user-attachments/assets/76786943-880b-4134-a34e-d828f89a00f1)



## Overview

**QEEGNet** is a hybrid neural network integrating **quantum computing** and the classical **EEGNet** architecture to enhance the encoding and analysis of EEG signals. By incorporating **variational quantum circuits (VQC)**, QEEGNet captures more intricate patterns within EEG data, offering improved performance and robustness compared to traditional models.

This repository contains the implementation and experimental results for **QEEGNet**, evaluated on the **BCI Competition IV 2a** dataset.



## Key Features

- **Hybrid Architecture**: Combines the EEGNet convolutional framework with quantum encoding layers for advanced feature extraction.
- **Quantum Layer Integration**: Leverages the unique properties of quantum mechanics, such as superposition and entanglement, for richer data representation.
- **Improved Robustness**: Demonstrates enhanced accuracy and resilience to noise in EEG signal classification tasks.
- **Generalizability**: Consistently outperforms EEGNet across most subjects in benchmark datasets.



## Architecture

QEEGNet consists of:
1. **Classical EEGNet Layers**: Initial convolutional layers process EEG signals to extract temporal and spatial features.
2. **Quantum Encoding Layer**: Encodes classical features into quantum states using a parameterized quantum circuit.
3. **Fully Connected Layers**: Converts quantum outputs into final classifications.


---

## Dataset

The **BCI Competition IV 2a** dataset was used for evaluation, featuring EEG signals from motor-imagery tasks.  
- **Subjects**: 9  
- **Classes**: Right hand, left hand, feet, tongue  
- **Preprocessing**: Downsampled to 128 Hz, band-pass filtered (4-38 Hz).  

For more details, refer to the [dataset documentation](https://www.bbci.de/competition/iv/).  
Or you can use the orgnized data format in https://github.com/CECNL/MAtt repo.  

---

## Usage
Coming soon!

## Citation
Hope this idea is helpful. I would appreciate you citing us in your paper, and the github.

```

@article{chen2024qeegnet,
  title={Qeegnet: Quantum machine learning for enhanced electroencephalography encoding},
  author={Chen, Chi-Sheng and Chen, Samuel Yen-Chi and Tsai, Aidan Hung-Wen and Wei, Chun-Shu},
  journal={arXiv preprint arXiv:2407.19214},
  year={2024}
}
```

