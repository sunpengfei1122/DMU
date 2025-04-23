
# README

This is the official code repository for the paper *"Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate"*.

**Paper Details:**

> - **Title:** Delayed Memory Unit: Modeling Temporal Dependency Through Delay Gate
> - **Authors:** Pengfei Sun, Jibin Wu*, Malu Zhang, Paul Devos, Dick Botteldooren
> - **Journal:** IEEE Transactions on Neural Networks and Learning Systems
> - **Year:** 2024
> - **DOI:** 10.1109/TNNLS.2024.3490833
> - **URL:** [IEEE Article](https://ieeexplore.ieee.org/document/10757311)](https://ieeexplore.ieee.org/document/10757311)
<!-- 1. Delayed Memory Unit (TNNLS 2024) -->
📄 <strong>Sun, P.</strong>, Wu, J., Zhang, M., Devos, P., &amp; Botteldooren, D.  
<strong>Delayed Memory Unit: Modelling Temporal Dependency Through Delay Gate.</strong>  
<em>IEEE Trans. on Neural Networks and Learning Systems, 2024.</em>  
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
  </pre>
  <button onclick="copyBib('txt8')">Copy</button>
  <button onclick="hideBib('bib8')">Close</button>
</div>
<script>
// Show 对应 id 的弹窗，并在 3 秒后自动隐藏
function showBibtex(id) {
  // 先把其它弹窗都关掉，保证一次只开一个
  document.querySelectorAll('div[id^="bib"]').forEach(d => d.style.display = 'none');
  const bib = document.getElementById(id);
  if (!bib) return;
  bib.style.display = 'block';
  setTimeout(() => bib.style.display = 'none', 3000);  // 3000ms 后自动隐藏
}

// “Close” 按钮手动隐藏
function hideBib(id) {
  const bib = document.getElementById(id);
  if (bib) bib.style.display = 'none';
}

// “Copy” 按钮
function copyBib(txtId) {
  const txt = document.getElementById(txtId).textContent;
  navigator.clipboard.writeText(txt);
}
</script>

 
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



