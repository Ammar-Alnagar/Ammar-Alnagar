# 🚀 Hi there, I'm Ammar Alnagar! 👋  

**Senior Machine Learning Software Engineer** | **LLM Researcher** | **AI Systems Optimization Specialist**  
🔬 Specializing in **LLM Reasoning**, **Multimodal AI**, **Quantization**, and **High-Performance AI Kernels**  
⚡ Currently focused on **Triton, PyTorch, CUDA, and Sub-4-Bit LLM Quantization**  

---

## 🧠 About Me  
- 🤖 Passionate about **LLMs, VLLMs, and advanced reasoning techniques**  
- ⚡ **Optimizing AI systems** with **Triton, CUDA, and efficient quantization (GPTQ, AWQ, FlexGen)**  
- 📜 Creator of **Zireal**, a reasoning-optimized LLM with structured inference  
- 🧩 Building **Spectrum**, a vision-language AI to enhance communication skills in autistic children  
- 🚀 Exploring **LLM acceleration, reinforcement learning, and Rust for AI optimization**  

---

## 🔬 Projects & Research  

### 🚀 Featured Projects  
- **[Zireal](https://huggingface.co/Daemontatox/Zireal-0)** – A high-speed reasoning LLM optimized for structured inference  
- **[Llama 70B CogniLink](https://huggingface.co/Daemontatox/Llama3.3-70B-CogniLink)** – An advanced SOTA reasoning model  
- **[Cogito-R1](https://huggingface.co/Daemontatox/Cogito-R1)** – Qwen2.5-32B fine-tuned for optimized reasoning  

### 🛠 Open-Source & Research  
- 🔥 **LLM Quantization & Inference Optimization** – GPTQ, AWQ, FlexGen, Sub-4-Bit Quantization  
- 📝 **Triton GPU Kernels for Optimized Transformer Computation**  
- 📚 **Multimodal AI, Cognitive AI, and Agentic Workflows**  
- 🚀 **Rust & Zig for High-Performance AI Applications**  

---

## ⚙️ Tech Stack & Languages  

### 💻 Programming Languages  
- ![Python](https://img.shields.io/badge/Python-FFD43B?style=flat&logo=python&logoColor=blue)  
- ![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)  
- ![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)  
- ![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)  
- ![Zig](https://img.shields.io/badge/Zig-F7A41D?style=flat&logo=zig&logoColor=black)  

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

<p align="left"> <a href="https://github.com/ryo-ma/github-profile-trophy"><img src="https://github-profile-trophy.vercel.app/?username=ammar-alnagar" alt="ammar-alnagar" /></a> </p>


<br clear="both">

<img src="https://raw.githubusercontent.com/Ammar-Alnagar/Ammar-Alnagar/output/snake.svg" alt="Snake animation" />

---
---

## 📫 Connect With Me  
💻 [GitHub](https://github.com/Ammar-Alnagar) | 📜 [LinkedIn](https://www.linkedin.com/in/ammar-alnagar-393413201/) | 🤗 [Huggingface](https://huggingface.co/Daemontatox)  

---

### 🧩 Fun Fact  
```python
# High-Performance Triton Kernel - Matrix Multiplication  
import triton  
import triton.language as tl  

@triton.jit  
def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):  
    pid = tl.program_id(axis=0)  
    offsets_m = pid * 128 + tl.arange(0, 128)  
    offsets_n = tl.arange(0, 128)  
    mask_m = offsets_m < M  
    mask_n = offsets_n < N  
    acc = tl.zeros([128, 128], dtype=tl.float32)  
    for k in range(K):  
        a = tl.load(A + offsets_m * K + k, mask=mask_m)  
        b = tl.load(B + k * N + offsets_n, mask=mask_n)  
        acc += a[:, None] * b[None, :]  
    tl.store(C + offsets_m * N + offsets_n, acc, mask=mask_m[:, None] & mask_n[None, :])  
```  

---

### 🔥 Latest Updates  
- 📌 Deep dive into **Triton + PyTorch for AI kernel optimization**  
- 🚀 Researching **LLM sub-4-bit quantization and low-bit inference**  
- ⚡ Exploring **Rust and Zig for AI system optimization**  

---
