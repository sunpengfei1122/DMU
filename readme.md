
# README

This is the code repository for the paper *"Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate"*.


## Abstract
The DMU incorporates a delay line structure along with delay gates into vanilla RNN, 
thereby enhancing temporal interaction and facilitating temporal credit assignment.  Specifically, the DMU is 
designed to directly distribute the input information to the optimal time instant in the future, rather than 
aggregating and redistributing it over time through intricate network dynamics. 


## Installation
**How to run:**
To use the DMU in your training framework, simply instantiate and apply it as follows:
#input_dim: number of input features (e.g., 256)
#delay_dim: delay-line length n (e.g., 20)
#input tensor x has shape (batch_size, time_steps, input_dim) 

output = DMU(input_dim=256, delay_dim=20)(x)   

**Paper Details:**
<!-- 1. Delayed Memory Unit (TNNLS 2025) -->
<a href="https://10.1109/TNNLS.2024.3490833" target="_blank"><button>PDF</button></a>
<button onclick="showBibtex('bib8')">Cite</button>
<div id="bib8" style="display:none; position:fixed; top:20%; left:50%; transform:translateX(-50%); background:#fff; border:1px solid #ccc; padding:1em; z-index:100; max-width:600px;">
  <pre id="txt8" style="white-space:pre-wrap;">
@ARTICLE{sun2024delayed,
  author={Sun, Pengfei and Wu, Jibin and Zhang, Malu and Devos, Paul and Botteldooren, Dick},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate}, 
  year={2025},
  volume={36},
  number={6},
  pages={10808-10818},
  doi={10.1109/TNNLS.2024.3490833}}


 






