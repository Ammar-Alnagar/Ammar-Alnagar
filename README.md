# 🚀 Hi there, I'm Ammar Alnagar! 👋  

**Machine Learning Systems Engineer** | **LLM Researcher** | **AI Optimization Specialist**  
🔬 Specializing in **LLM Reasoning**, **Multimodal AI**, **GPU Optimization**, and **Quantization**  
⚡ Currently focused on **Triton, CUDA, and high-performance AI kernels**  

---

## 🧠 About Me  
- 🤖 Passionate about **LLMs, VLLMs, and advanced reasoning techniques**  
- ⚡ **Optimizing AI workflows** with **Triton, CUDA, and quantization (GPTQ, AWQ)**  
- 📜 Creator of **Zireal**, a reasoning-optimized LLM with efficient thought chains  
- 🧩 Building **Spectrum**, a vision-language AI to support autistic children’s communication skills  
- 🚀 Exploring **high-speed transformer inference and LLM acceleration**  

---

## 🔬 Projects & Research  

### 🚀 Featured Projects  
- **[Zireal](https://huggingface.co/Daemontatox/Zireal-0)** – High-speed reasoning LLM optimized for structured inference  
- **[Llama 70B CogniLink](https://huggingface.co/Daemontatox/Llama3.3-70B-CogniLink)** – Advanced SOTA reasoning model  
- **[Cogito-R1](https://huggingface.co/Daemontatox/Cogito-R1)** – Qwen2.5-32B fine-tuned for optimized reasoning  

### 🛠 Open-Source & Research  
- 🔥 **LLM Quantization & Inference Optimization** – GPTQ, AWQ, FlexGen  
- 📝 **Triton GPU Kernels for Fast Transformer Computation**  
- 📚 **Multimodal AI & Cognitive AI Workflows**  

---

## ⚙️ Tech Stack & Languages  

### 💻 Programming Languages  
- ![Python](https://img.shields.io/badge/Python-FFD43B?style=flat&logo=python&logoColor=blue)  
- ![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)  
- ![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)  
- ![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=java&logoColor=white)  

### 🛠 AI & ML Frameworks  
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)  
- ![Triton](https://img.shields.io/badge/Triton-3498DB?style=flat&logo=triton&logoColor=white)  
- ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)  
- ![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat&logo=nvidia&logoColor=white)  
- ![JAX](https://img.shields.io/badge/JAX-007ACC?style=flat&logo=jax&logoColor=white)  

### 📡 System & Optimization Tools  
- ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)  
- ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)  
- ![LLVM](https://img.shields.io/badge/LLVM-555555?style=flat&logo=llvm&logoColor=white)  

---

## 📊 GitHub Stats & Contributions  

<br clear="both">

<div align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Ammar-Alnagar&hide_title=false&hide_rank=false&show_icons=true&include_all_commits=true&count_private=true&disable_animations=false&theme=dracula&locale=en&hide_border=false" height="150" alt="stats graph"  />
  <img src="https://streak-stats.demolab.com?user=Ammar-Alnagar&locale=en&mode=daily&theme=dracula&hide_border=false&border_radius=5" height="150" alt="streak graph"  />
  <img src="https://github-readme-stats.vercel.app/api/top-langs?username=Ammar-Alnagar&locale=en&hide_title=false&layout=compact&card_width=320&langs_count=5&theme=dracula&hide_border=false" height="150" alt="languages graph"  />
</div>

[![trophy](https://github-profile-trophy.vercel.app/?username=ammar-alnagar&theme=onedark)](https://github.com/ryo-ma/github-profile-trophy)
###

<br clear="both">

<img src="https://raw.githubusercontent.com/Ammar-Alnagar/Ammar-Alnagar/output/snake.svg" alt="Snake animation" />

###
---

## 📫 Connect With Me  
💻 [GitHub](https://github.com/Ammar-Alnagar) | 📜 [LinkedIn](https://www.linkedin.com/in/ammar-alnagar-393413201/) | 🤗 [Huggingface](https://huggingface.co/Daemontatox)  

---

### 🐍 Fun Fact  
```python
# First Triton kernel - Vector Addition  
import triton  
import triton.language as tl  

@triton.jit  
def vector_add(X, Y, Z, N: tl.constexpr):  
    pid = tl.program_id(axis=0)  
    offsets = pid * 128 + tl.arange(0, 128)  
    mask = offsets < N  
    x = tl.load(X + offsets, mask=mask)  
    y = tl.load(Y + offsets, mask=mask)  
    tl.store(Z + offsets, x + y, mask=mask)

```

