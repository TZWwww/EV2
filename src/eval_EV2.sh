prefix=YOUR_WORKSPACE

# model=gpt-3.5-turbo
# model=gpt-4-0125-preview
model_name=Qwen2-7B-Instruct
# model_name=Mistral-7B-Instruct-v0.2
# model_name=Llama-2-7b-chat-hf
# model_name=Baichuan2-7B-Chat
# model_name=Meta-Llama-3-8B-Instruct
# model_name=vicuna-7b-v1.3
# model_name=phi-2
# model_name=Qwen1.5-7B-Chat
# model_name=Orca-2-7b
# model_name=chatglm2-6b
# model_name=internlm2_5-7b-chat
model=/YOUR_MODEL_DIR/${model_name}

task_name=S_CEC
format=CEC
version=0
K=0
desc=${task_name}_${model_name}_K${K}_${version}
echo ${desc}
python eval.py \
	  --openai_key YOUR_OPENAI_KEY \
	  --cache_dir ${prefix}models/ \
	  --model_version ${model} \
          --task_name ${task_name} \
	  --desc ${desc} \
          --data_dir ${task_name}.jsonl \
          --output_dir ${prefix}experiments/EV2/${desc}/ \
	  --num_gpus 0 \
          --per_device_eval_batch_size 1 \
          --number_of_folds ${version} \
          --data_type ${K}

task_name=I_CEC
format=CEC
version=0
K=0
desc=${task_name}_${model_name}_K${K}_${version}
echo ${desc}
python eval.py \
	  --openai_key YOUR_OPENAI_KEY \
	  --cache_dir ${prefix}models/ \
	  --model_version ${model} \
          --task_name ${task_name} \
	  --desc ${desc} \
          --data_dir ${task_name}.jsonl \
          --output_dir ${prefix}experiments/EV2/${desc}/ \
	  --num_gpus 0 \
          --per_device_eval_batch_size 1 \
          --number_of_folds ${version} \
          --data_type ${K}


task_name=S_CRR
format=CRR
version=0
K=0
desc=${task_name}_${model_name}_K${K}_${version}
echo ${desc}
python eval.py \
	  --openai_key YOUR_OPENAI_KEY \
	  --cache_dir ${prefix}models/ \
	  --model_version ${model} \
          --task_name ${task_name} \
	  --desc ${desc} \
          --data_dir ${task_name}.jsonl \
          --output_dir ${prefix}experiments/EV2/${desc}/ \
	  --num_gpus 0 \
          --per_device_eval_batch_size 1 \
          --number_of_folds ${version} \
          --data_type ${K}


task_name=I_CRR
format=CRR
version=0
K=0
desc=${task_name}_${model_name}_K${K}_${version}
echo ${desc}
python eval.py \
	  --openai_key YOUR_OPENAI_KEY \
	  --cache_dir ${prefix}models/ \
	  --model_version ${model} \
          --task_name ${task_name} \
	  --desc ${desc} \
          --data_dir ${task_name}.jsonl \
          --output_dir ${prefix}experiments/EV2/${desc}/ \
	  --num_gpus 0 \
          --per_device_eval_batch_size 1 \
          --number_of_folds ${version} \
          --data_type ${K}









