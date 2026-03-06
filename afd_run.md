CPATH=/hpelustre/wuhanjia/vllm/.venv/lib/python3.12 \
CUDA_VISIBLE_DEVICES=2,3 \
nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite true \
  --trace-fork-before-exec=true \
  --output=/hpelustre/wuhanjia/vllm/profiler-logs/run-afd/ffn-profile \
  /hpelustre/wuhanjia/vllm/.venv/bin/vllm fserver \
  deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 \
  --enforce-eager \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/hpelustre/wuhanjia/vllm/profiler-logs/run-afd/ffn-profile \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}}'


CPATH=/hpelustre/wuhanjia/vllm/.venv/lib/python3.12 \
CUDA_VISIBLE_DEVICES=0,1 \
nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite true \
  --trace-fork-before-exec=true \
  --output=/hpelustre/wuhanjia/vllm/profiler-logs/run-afd/attn-profile \
  /hpelustre/wuhanjia/vllm/.venv/bin/vllm serve \
  deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 \
  --enforce-eager \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --dbo-prefill-token-threshold 12 \
  --dbo-decode-token-threshold 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/hpelustre/wuhanjia/vllm/profiler-logs/run-afd/attn-profile \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}}'
 
 /hpelustre/wuhanjia/vllm/.venv/bin/vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite --dataset-name random --input-len 200 --output-len 20 --num-prompts 20 --request-rate inf 2>&1
  