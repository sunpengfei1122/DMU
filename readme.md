
# README

This is the code repository for the paper *"Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate"*.

**Paper Details:**

> - **Title:** Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate
> - **Authors:** Pengfei Sun, Jibin Wu*, Malu Zhang, Paul Devos, Dick Botteldooren
> - **Journal:** IEEE Transactions on Neural Networks and Learning Systems
> - **Year:** 2024
> - **DOI:** 10.1109/TNNLS.2024.3490833
> - **URL:** [IEEE Article](https://ieeexplore.ieee.org/document/10757311)](https://ieeexplore.ieee.org/document/10757311)
<!-- 1. Delayed Memory Unit (TNNLS 2024) -->
<a href="https://arxiv.org/abs/2310.14982" target="_blank"><button>PDF</button></a>
<button onclick="showBibtex('bib8')">Cite</button>
<div id="bib8" style="display:none; position:fixed; top:20%; left:50%; transform:translateX(-50%); background:#fff; border:1px solid #ccc; padding:1em; z-index:100; max-width:600px;">
  <pre id="txt8" style="white-space:pre-wrap;">
@article{sun2024delayed,
  title={Delayed memory unit: modeling temporal dependency through delay gate},
  author={Sun, Pengfei and Wu, Jibin and Zhang, Malu and Devos, Paul and Botteldooren, Dick},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}


 
## Abstract
Recurrent Neural Networks (RNNs) are widely recognized for their proficiency in modeling temporal dependencies, 
making them highly prevalent in sequential data processing applications. Nevertheless, vanilla RNNs are confronted 
with the well-known issue of gradient vanishing and exploding, posing a significant challenge for learning and 
establishing long-range dependencies. Additionally, gated RNNs tend to be over-parameterized, resulting in poor 
computational efficiency and network generalization. To address these challenges, this paper proposes a novel 
Delayed Memory Unit (DMU). The DMU incorporates a delay line structure along with delay gates into vanilla RNN, 
thereby enhancing temporal interaction and facilitating temporal credit assignment.  Specifically, the DMU is 
designed to directly distribute the input information to the optimal time instant in the future, rather than 
aggregating and redistributing it over time through intricate network dynamics. Our proposed DMU demonstrates 
superior temporal modeling capabilities across a broad range of sequential modeling tasks, utilizing considerably 
fewer parameters than other state-of-the-art gated RNN models in applications such as speech recognition, radar 
gesture recognition, ECG waveform segmentation, and permuted sequential image classification.



## Installation
**How to run:**
To use the DMU in your training framework, simply instantiate and apply it as follows:
#input_dim: number of input features (e.g., 256)
#delay_dim: delay-line length n (e.g., 20)
output = DMU(input_dim=256, delay_dim=20)(x)
Ensure that your input tensor x has shape (batch_size, time_steps, input_dim).

