# Addressing the alignment problem in transportation policy making: an LLM approach
Author: Xiaoyu Yan, Tianxing Dai, Yu (Marco) Nie 


### Abstract 
A key challenge in transportation planning is that the collective preferences of heterogeneous travelers often diverge from the policies produced by model-driven decision tools. This misalignment frequently results in implementation delays or failures. Here, we investigate whether large language models (LLMs)—noted for their capabilities in reasoning and simulating human decision-making—can help inform and address this alignment problem.
We develop a multi-agent simulation in which LLMs, acting as agents representing residents from different communities in a city, participate in a referendum on a set of transit policy proposals.  Using chain-of-thought reasoning, LLM agents generate Ranked-Choice or approval-based preferences, which are aggregated using instant-runoff voting (IRV) to model democratic consensus.
We implement this simulation framework with both GPT-4o and Claude-3.5, and apply it for Chicago and Houston. 
Our findings suggest that LLM agents are capable of approximating plausible collective preferences and responding to local context, while also displaying model-specific behavioral biases and modest divergences from optimization-based benchmarks. These capabilities underscore both promise and limitations of LLMs as tools for solving the alignment problem in transportation decision-making. 


<img width="3694" height="1526" alt="image" src="https://github.com/user-attachments/assets/66780e72-af67-4cf7-a167-f31a410cb473" />
