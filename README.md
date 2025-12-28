# Hey, I'm Ammar Alnagar


I build inference engines and optimize LLMs at the GPU level. Started with high-level frameworks, got frustrated with black-box performance, so now I'm down in CUDA/C++ land writing kernels and profiling memory bandwidth. Currently exploring Mojo because writing GPU code that looks like Python but runs at C++ speeds is too good to ignore.

**Fun fact:** I've probably spent more time optimizing sub-millisecond latencies than I have sleeping this year. My GPU utilization dashboard has better uptime than my work-life balance. At least the tokens/sec charts look pretty 

---

### Projects I'm Actually Proud Of

**[Helios-Engine](https://github.com/Ammar-Alnagar/Helios-Engine)** - Rust agent framework for when LangChain is too slow and you need tool calling that actually works. Streaming, orchestration, the works.

**[Zllm](https://github.com/Ammar-Alnagar/Zllm)** - Building an LLM inference engine from scratch in C++/CUDA. Flash Attention, PagedKV cache, RadixAttention. Learning by doing because reading vLLM source code at 2am wasn't cutting it.

**[MILI](https://github.com/Ammar-Alnagar/MILI)** - End-to-end LLM inference engine in Mojo. Turns out you can write GPU kernels that look like Python but run like C++. RoPE, RMSNorm, FlashAttention—all without the usual CUDA pain.

**[Axion](https://github.com/Ammar-Alnagar/Axion)** - Pure Rust inference engine. Sub-10ms p99 latency, beats vLLM by 20%+ on my benchmarks. Built when I cared about Rust, keeping it here because the benchmarks still slap.

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⡶⠿⠿⠷⣶⣄⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⠁⠀⠀⢀⣀⡀⠙⣷⡀⠀⠀⠀
⠀⠀⠀⡀⠀⠀⠀⠀⠀⢠⣿⠁⠀⠀⠀⠘⠿⠃⠀⢸⣿⣿⣿⣿
⠀⣠⡿⠛⢷⣦⡀⠀⠀⠈⣿⡄⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⠟
⢰⡿⠁⠀⠀⠙⢿⣦⣤⣤⣼⣿⣄⠀⠀⠀⠀⠀⢴⡟⠛⠋⠁⠀
⣿⠇⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠈⣿⡀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀⠀
⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡇⠀⠀⠀
⠸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠀⠀⠀⠀
⠀⠹⣷⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣰⡿⠁⠀⠀⠀⠀
⠀⠀⠀⠉⠙⠛⠿⠶⣶⣶⣶⣶⣶⠶⠿⠟⠛⠉⠀⠀⠀⠀⠀⠀
```
