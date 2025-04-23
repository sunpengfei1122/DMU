
# README

This is the official code repository for the paper *"Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate"*.

**Paper Details:**

> - **Title:** Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate
> - **Authors:** Pengfei Sun, Jibin Wu*, Malu Zhang, Paul Devos, Dick Botteldooren
> - **Journal:** IEEE Transactions on Neural Networks and Learning Systems
> - **Year:** 2024
> - **DOI:** 10.1109/TNNLS.2024.3490833
> - **URL:** [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0893608024006026)](https://ieeexplore.ieee.org/document/10757311)

## Abstract
Recurrent Neural Networks (RNNs) are widely recognized for their proficiency in modeling temporal dependencies,
making them highly prevalent in sequential data processing
applications. Nevertheless, vanilla RNNs are confronted with
the well-known issue of gradient vanishing and exploding,
posing a significant challenge for learning and establishing
long-range dependencies. Additionally, gated RNNs tend to be
over-parameterized, resulting in poor computational efficiency
and network generalization. To address these challenges, this
paper proposes a novel Delayed Memory Unit (DMU). The
DMU incorporates a delay line structure along with delay gates
into vanilla RNN, thereby enhancing temporal interaction and
facilitating temporal credit assignment. Specifically, the DMU
is designed to directly distribute the input information to the
optimal time instant in the future, rather than aggregating and
redistributing it over time through intricate network dynamics.
Our proposed DMU demonstrates superior temporal modeling
capabilities across a broad range of sequential modeling tasks,
utilizing considerably fewer parameters than other state-of-the-art
gated RNN models in applications such as speech recognition,
radar gesture recognition, ECG waveform segmentation, and
permuted sequential image classification.



## Installation
**How to run:**
You can just copy the network under your training framework, and then use
output= DMU(input_dim=256, delay_dim=20)  # input_dim is the dim of inputs, delay_dim is the delay line length n
Try to make sure the input dimension is [x: (batch, time_steps, input_dim)]



