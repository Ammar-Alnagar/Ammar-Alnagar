
🚀 Hi there, I'm Ammar Alnagar! 👋

Machine Learning Systems Engineer | LLM Researcher | AI Optimization Specialist
🔬 Specializing in LLM Reasoning, Multimodal AI, GPU Optimization, and Quantization
⚡ Currently focused on Triton, CUDA, and high-performance AI kernels


---

🧠 About Me

🤖 Passionate about LLMs, VLLMs, and advanced reasoning techniques

⚡ Optimizing AI workflows with Triton, CUDA, and quantization (GPTQ, AWQ)

📜 Creator of Zireal, a reasoning-optimized LLM with efficient thought chains

🧩 Building Spectrum, a vision-language AI to support autistic children’s communication skills

🚀 Exploring high-speed transformer inference and LLM acceleration



---

🔬 Projects & Research

🚀 Featured Projects

Zireal – High-speed reasoning LLM optimized for structured inference

Llama 70B CogniLink – Advanced SOTA reasoning model

Cogito-R1 – Qwen2.5-32B fine-tuned for optimized reasoning


🛠 Open-Source & Research

🔥 LLM Quantization & Inference Optimization – GPTQ, AWQ, FlexGen

📝 Triton GPU Kernels for Fast Transformer Computation

📚 Multimodal AI & Cognitive AI Workflows



---

⚙️ Tech Stack

Core Specialties:






Programming Languages:





---

📊 GitHub Stats & Contributions





---

📫 Connect With Me

💻 GitHub | 📜 LinkedIn | 🤗 Huggingface


---

🐍 Fun Fact

```

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

> 🚀 Mastering GPU acceleration, one Triton kernel at a time!