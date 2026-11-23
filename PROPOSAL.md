**Title**

A Hybrid Parallel AI Dental Hygiene Feedback Generator using a Panel of Small LLM Judges

**Problem**

We are working on a dental hygiene coaching simulator. The current setup uses a hosted LLM to generate dental hygiene examples. These examples are used to fine-tune a base model SLM (Llama 3.1 8B) with LoRA. The merged model is then used to generate responses to dental hygiene scenarios. A larger judge model (27B+) will be used to score the responses of the merged models. The biggest challenges of using this process are that it currently runs as a series of disconnected steps, and that large judges create a bottleneck with each response taking at least 30 seconds per judged scenario per judge.

I propose to integrate the disconnected process into a single pipeline and replace the larger judges with a panel of smaller judges (\<10B). The panel of judges idea is motivated by Verga et al., who proposed a Panel of LLMs (PoLL) methodology and suggested that a panel of smaller models from different model families agreed better with human judgements than a single larger judge at 7-8x lower cost \[1\]. For this experiment, one set of training data will go through the entire pipeline: LoRA fine-tuning, response generation and judging by two different judge models. The pipeline can later be extended so that the output of one pass is used to train a new model to generate new training data.

The training data for this experiment comes from Claude Sonnet 5.5. The goal of this project is to minimize the end-to-end time to train, generate, and judge a fixed set of dental hygiene scenarios, and to maximize judged responses per second, by scaling each pipeline stage across multiple GPU nodes. Because the focus is HPC performance, the full 4,000-scenario set serves as both the generation input and the judging workload. This includes scenarios used in training, so the scores are not a measure of model quality. Scores are recorded only to confirm the pipeline runs correctly and are not an evaluation of model or judge quality. The results are also not clinically valid since the models were trained using synthetic data.

<img src="media/image1.png" style="width:6.5in;height:2.52818in" />

**Computing Platform**

I will use the V100 GPU nodes on ODU’s Wahab HPC cluster for all scaling experiments. Wahab has 18 GPU nodes available, each with 4 V100 GPUs, for 72 V100 GPUs in total. Each node has an Intel(R) Xeon(R) Gold 6130 CPU at 2.1 GHz with 32 cores, 128 GB of memory and AVX512 support \[2\]. Each V100 GPU has 16 GB of VRAM, 5,120 CUDA cores, 640 tensor cores, and 900 GB/s of memory bandwidth. All nodes are connected by a Mellanox InfiniBand EDR network providing 100 Gbps throughput \[3\].

Nodes are scheduled through a shared SLURM queue, so run-to-run timing may vary with other users’ load. Each configuration will be repeated at least three times to reduce the impact of the variability. All V100 nodes are the same type, so scaling points are comparable across node counts. The largest planned configuration uses 12 GPUs (3 nodes) in the 8-bit quantized judging case, and 24 GPUs (6 nodes) in the pipeline-parallel judging case. Runs with more than 4 GPUs span multiple nodes.

The pipeline will be packaged as a Docker image and run through the cluster’s container runtime, which also makes it portable to other platforms and the cloud.

**Hybrid Parallelism**

The pipeline has three stages: LoRA fine-tuning, response generation and judging. Each stage can be parallelized, but each stage runs one after another because each needs the previous stage’s output. A driver script submits each stage as a SLURM job with dependencies, and a shared filesystem passes data between stages.

The LoRA fine-tuning uses Ray Train to run distributed data-parallel training across GPUs and nodes. The training data is split across GPUs and gradients are aggregated between steps over the network.

The fine-tuned models will be served through a vLLM server using pipeline parallelism for response generation. The 8B model will be split into layer groups across GPUs, and each GPU handles one group. Using this method, several responses can be in flight at once. The served model must be initialized which has a startup cost.

The two judges (Qwen3-8B and Granite-4.1-8B) are different tasks that run at the same time on separate GPUs, launched by their own SLURM job using task parallelism. Within each judge, the responses are sharded across that judge’s replicas. Each judge replica occupies one GPU if 8-bit quantization is validated, otherwise two GPUs using pipeline parallelism. Initial testing showed that Qwen3-8B runs approximately 2x slower than Granite-4.1-8B when used as a judge. The number of resources allocated to the judges will reflect the ratio of the processing speeds. Every response is judged by both judges. Each judge writes its scores on the shared filesystem, and a final step joins the scores by response ID and aggregates them into one score per response.

**Objective Difficulties**

LoRA fine-tuning is compute-bound, and its network traffic is small because only LoRA gradients are exchanged. Generation and judging are dominated by decoding, which is limited by GPU memory bandwidth, so they scale by adding replicas rather than by faster compute. The main limit to scaling is startup: loading a vLLM server takes minutes and does not shrink as GPUs are added, so it acts as a serial fraction that lowers efficiency at high GPU counts. SLMs need significant GPU memory. An 8B model in 16-bit precision does not fit on a 16GB V100, so each model replica uses either 8-bit quantization or two-GPU pipeline parallelism. Judging time differs between models. Even with GPUs allocated in the speed ratio, a response is finished only when its slower judge finishes, which can cause idle GPUs. Judging time also varies by response because output length of response varies. The speed ratio is the average ratio across responses from a sample of responses. Generating new synthetic data is limited by token rate limits on online hosted LLMs such as Claude or ChatGPT. Due to this constraint, we will only be able to scale up to 4,000 dental hygiene scenarios, which also caps the weak-scaling workload.

**Scaling Experiments**

Runs are repeated at least three times, and the mean and spread are reported. Startup time (model loading and server initialization) is measured separately from processing time.

| **Stage**                        | **Strong Scaling**                                                                                                       | **Weak Scaling**                                                     |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| LoRA Fine-tuning                 | 4,000 examples on 1, 2, 4, 8 GPUs                                                                                        | 1,000 examples per GPU on 1, 2, 3, 4 GPUs                            |
| Response Generation              | Fixed workload of 1,000 scenarios on 1, 2, 3, 4, 6 replicas (2, 4, 6, 8, 12 GPUs)                                        | 1,000 scenarios per replica on 1, 2, 3, 4 replicas (2, 4, 6, 8 GPUs) |
| Judging (8-bit case)             | Fixed workload of 1,000 responses on 1 GPU then 3, 6, 9, 12 GPUs (units of 2 Qwen + 1 Granite)                           | 1,000 responses per unit on units 1–4 (3, 6, 9, 12 GPUs)             |
| Judging (pipeline-parallel case) | Units of 6 GPUs (4 Qwen + 2 Granite); baseline is one 2-GPU replica running both judges in turn, then 6, 12, 18, 24 GPUs | 1,000 responses per unit on units 1–4 (6, 12, 18, 24 GPUs)           |

Scaling experiments will also happen on the complete pipeline. At scaling unit *k* (1 to 4), the pipeline uses *k* GPUs for LoRA, *k* generation replicas, and *k* judge units, on a workload of 1,000×*k* responses, for weak scaling. Stages run one after another, so the maximum GPU count in use at any time is 12 in the 8-bit case. Strong scaling of the full pipeline uses the same fixed workloads as the stage-level runs. Start-up overhead and parallel overhead are measured for each run.

**Application/Computer Architecture Parameters to Tune**

| **Stage**           | **Parameters**                                                                                                                                                                                                                    |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LoRA Fine-tuning    | Number of GPUs and nodes; per-device batch size and gradient accumulation; Ray workers and CPUs per worker                                                                                                                        |
| Response Generation | Number of replicas; pipeline-parallel size versus replica count (e.g. 2 GPUs × 6 replicas); vLLM max concurrent sequences and batch size; GPU memory utilization; client request concurrency                                      |
| Judging             | GPUs per judge and the Qwen-to-Granite GPU ratio; 8-bit quantization versus two-GPU pipeline parallelism; vLLM max concurrent sequences; thinking mode on or off, or a cap on thinking tokens; number of judge calls per response |
| Whole Pipeline      | Nodes allocated per stage; container and SLURM settings such as CPUs per task                                                                                                                                                     |

**Analysis Plan**

The main metrics will be runtime, responses generated and judged per second, strong scaling speedup and efficiency, weak scaling efficiency, parallel overhead, and start up time as a share of the total time. The main analysis will plot the strong and weak scaling numbers using Python and matplotlib. Ray dashboard and timeline will be used to see per worker activity. SLURM accounting will be used for job start and end times and time between stages. vLLM metrics endpoints will be used to measure requests in flight, and tokens per second. Per-judge completion times will also be measured to check that the resource allocation balance the judge timing differences.

**References**:

\[1\] Verga, P., Hofstätter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., and Lewis, P. (2024). Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. arXiv:2404.18796. <https://arxiv.org/abs/2404.18796>

\[2\] ODU HPC Wiki, Wahab cluster technical specifications, https://wiki.hpc.odu.edu/en/Cluster/Wahab

\[3\] ODU Research Computing, Request HPC Resources (Wahab cluster description), <https://www.odu.edu/research-computing/compute>
