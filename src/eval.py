import os
import json
import logging
import os
import copy
import time
from typing import Dict
import pdb
import bdb

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import openai
from openai import OpenAI

from args import parse_args
from data import EV2Dataset
from metrics import EV2Metrics
from aiohttp.client_exceptions import ClientConnectorError

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def call_API_OpenAI(model_version, text_list, max_tokens=20):
    client = OpenAI(api_key=args.openai_key)
        
    preds = []
    for text in text_list:
        messages = [{
                    'role': 'user',
                    'content': text,
                }]

        completion = client.chat.completions.create(
                model=args.model_version,
                messages=messages,
                max_tokens=max_tokens,
                # logit_bias=logit_bias,
                )

        preds.append(completion.choices[0].message.content)

        max_calls_per_minute: int = 60
        time.sleep(60 / max_calls_per_minute)
    return preds


def evaluate(args, eval_dataloader, metrics):
    '''Evaluation'''

    all_src = []
    all_preds = []
    all_labs = []
    all_probs = []
    all_tgts = []
    all_steps = []
 
    missing_steps = args.missing_steps
    cur_missing_steps = []
    steps_to_run = list(range(len(eval_dataloader)))

    if not args.model_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-0125-preview', 'gpt-4o-2024-05-13']:
        tokenizer = AutoTokenizer.from_pretrained(args.model_version, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_version, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    device = 'cuda'

    while steps_to_run:
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"evaluating",
                                position=0,
                                leave=True):
            if step not in steps_to_run:
                continue
            if missing_steps is not None and step not in missing_steps:
                continue
            if args.test > 0 and step > args.test:
                steps_to_run = []
                break
            src_ = batch['src']
            stm_ = batch['stm']
            labs = batch['labs']

            try:
                if args.model_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-0125-preview', 'gpt-4o-2024-05-13']:
                    preds = call_API_OpenAI(args.model_version, src_[0], max_tokens=512)
                else:
                    instruction = src_[0][0]
                    if 'WizardLM' in args.model_version:
                        prompt = f"{instruction}\n\n### Response:"
                    elif 'Llama-3' in args.model_version:
                        messages = [
                            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                            {"role": "user", "content": instruction},
                        ]

                        prompt = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors='pt'
                        )
                    elif 'Qwen2' in args.model_version:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": instruction}
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    elif any([s in args.model_version for s in ['phi', 'chatglm2', 'internlm']]):
                        prompt = instruction
                    elif any([s in args.model_version for s in ['alpaca', 'TimeLlama']]):
                        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
                    elif 'vicuna' in args.model_version:
                        prompt = f"USER: {instruction} ASSISTANT:"
                    elif any([s in args.model_version for s in ['mistral', 'Mistral']]):
                        prompt = f"<s>[INST]{instruction}[/INST]"
                    elif any([s in args.model_version for s in ['llama', 'Llama']]):
                        prompt = f"[INST]{instruction}[/INST]"
                    elif any([s in args.model_version for s in ['Orca', 'Qwen']]):
                        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"
                    elif 'Baichuan2' in args.model_version:
                        prompt = f"<reserved_106>{instruction}<reserved_107>"
                    else:
                        prompt = f"User: {instruction} Assistant:"

                    
                    if 'Llama-3' in args.model_version:
                        input_ids = prompt.to(model.device)

                        terminators = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]

                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=256,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                        )
                        response = outputs[0][input_ids.shape[-1]:]
                        preds = [tokenizer.decode(response, skip_special_tokens=True)]

                    else:
                        model_inputs = tokenizer(prompt, return_tensors='pt').to(device)
                        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        preds = [answer]

                all_src.extend(src_[0])
                all_preds.extend(preds)
                all_labs.extend(labs)
                all_tgts.extend(stm_[0])
                all_steps.append(step)

                steps_to_run.remove(step)

            except Exception as e:
                if isinstance(e, bdb.BdbQuit):
                    raise e
                else:
                    print(e)
                    print(f'Break at step: {step}')
                    print(f'Batchsize: {args.per_device_eval_batch_size}')
                    cur_missing_steps.append(step)
    
    result = metrics.compute_result(all_preds, all_tgts, all_labs)

    logger.info(f'Total steps num: {len(all_preds)}')
    logger.info(f'Current missing steps: {cur_missing_steps}')
    logger.info(f'Current missing steps num: {len(cur_missing_steps)}')
    return result, all_src, all_preds, all_labs, all_steps


def main(args, k, i):
    logger.info(f'evaluating on {args.task_name} k = {k} i = {i}')
    output_file = os.path.join(args.output_dir, f'{k}_{i}')

    if not os.path.exists(output_file):
       os.makedirs(output_file)

    metrics = EV2Metrics(args.task_name, use_cot=args.use_cot, cluster_dir=args.cluster_dir)

    testset = EV2Dataset(args.task_name,
                          args.data_dir,
                          k=k,
                          i=i,
                          max_seq_length=args.max_source_length,
                          # shuffle=True if args.generative_format in ['lmICL'] else False,
                          split=args.split,
                          cluster_dir=args.cluster_dir,
                          )
    test_dataloader = DataLoader(testset, collate_fn=testset.collect_fn, batch_size=args.per_device_eval_batch_size, pin_memory=False)

    # Evaluate!
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(testset)}")

    r, srcs, preds, labs, all_steps = evaluate(args, test_dataloader, metrics)
    labs = [[lab] if isinstance(lab, str) else lab for lab in labs]

    if output_file is not None:
        logger.info(r)
        f = open(os.path.join(output_file, "results.json"), "w")
        json.dump(r, f)

        if args.missing_steps is None:
            f = open(output_file + "/output.jsonl", "w")
        else:
            f = open(output_file + "/output.jsonl", "a")

        to_write = [{'id': step, 'src': src, 'pred': pred, 'lab': '|'.join(lab)} for src, pred, lab, step in zip(srcs, preds, labs, all_steps)]
        for line in to_write:
            f.write(json.dumps(line) + '\n')
        f.close()
    
    return r

if __name__ == "__main__":
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    final_results = {}
    avg_std = {}
    name2score = {'acc': None, 'F1': None}

    for k in args.data_type:
        for i in args.number_of_folds:
            # print(torch.cuda.is_available())
            result = main(copy.deepcopy(args), int(k), i)
            for m, p in result.items():
                final_results[f'{m}_{k}_{i}'] = p

        name2score = dict([(k, v) for k, v in name2score.items() if v is not None])

        for name in name2score.keys():
            name2score[name] = [final_results[f'{name}_{k}_{i}'] for i in args.number_of_folds]

        for name, ms in name2score.items():
            avg = round(float(np.mean(ms)), 2)
            std = round(float(np.std(ms)), 2)
            final_results[f'{name}_{k}_avg'] = avg
            final_results[f'{name}_{k}_std'] = std
            avg_std[f'{name}_{k}_avg'] = avg
            avg_std[f'{name}_{k}_std'] = std
    
    logger.info(f'dataset: {args.task_name} desc: {args.desc}')
    for name in name2score.keys():
        k_str = []
        res_str = []
        for k in args.data_type:
            k_str.append(k)
            avg = avg_std[f'{name}_{k}_avg']
            std = avg_std[f'{name}_{k}_std']
            res_str.append(f'{name}_{avg}-{std}')
        k_str = '\t'.join(k_str)
        res_str = '\t'.join(res_str)
        logger.info(k_str)
        logger.info(res_str)

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
            for k in args.data_type:
                res = '/'.join([format(final_results[f'{name}_{k}_avg'], '.2f') for name in name2score.keys()])
                final_results[k] = res
                print(res)

            json.dump(final_results, f)

    logger.info('Finished!')
