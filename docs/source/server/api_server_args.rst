APIServer 参数详解
=============================

.. code-block:: console

    usage: -m lightllm.server.api_server [-h] [--host HOST] [--port PORT] [--model_dir MODEL_DIR] [--tokenizer_mode TOKENIZER_MODE] [--load_way LOAD_WAY] [--max_total_token_num MAX_TOTAL_TOKEN_NUM]
                        [--batch_max_tokens BATCH_MAX_TOKENS] [--eos_id EOS_ID [EOS_ID ...]] [--running_max_req_size RUNNING_MAX_REQ_SIZE] [--tp TP] [--max_req_input_len MAX_REQ_INPUT_LEN]
                        [--max_req_total_len MAX_REQ_TOTAL_LEN] [--nccl_port NCCL_PORT] [--mode MODE [MODE ...]] [--trust_remote_code] [--disable_log_stats]
                        [--log_stats_interval LOG_STATS_INTERVAL] [--router_token_ratio ROUTER_TOKEN_RATIO] [--router_max_new_token_len ROUTER_MAX_NEW_TOKEN_LEN]
                        [--router_max_wait_tokens ROUTER_MAX_WAIT_TOKENS] [--use_dynamic_prompt_cache] [--splitfuse_block_size SPLITFUSE_BLOCK_SIZE] [--splitfuse_mode] [--beam_mode]
                        [--diverse_mode] [--token_healing_mode] [--enable_multimodal] [--cache_capacity CACHE_CAPACITY] [--cache_reserved_ratio CACHE_RESERVED_RATIO]
                        [--data_type {fp16,float16,bf16,bfloat16,fp32,float32}] [--return_all_prompt_logprobs] [--use_reward_model] [--long_truncation_mode {None,head,center}] [--use_tgi_api]
                        [--health_monitor] [--metric_gateway METRIC_GATEWAY] [--job_name JOB_NAME] [--grouping_key GROUPING_KEY] [--push_interval PUSH_INTERVAL] [--enable_monitor_auth]

