# HonestPersonaTrain
This is a repository for using Supervised Fine-Tuning to train an honest persona into a language model. The idea is that we try to train an alternate personality into the model with different values/goals than the "main persona" / "untrusted assistant", which will honestly report on whatever the main persona is doing/thinking.

The repo does training, some data-processing and tokenization, and performs light validation during training.

# Use Repo
## Project

run uv sync and activate the venv

## Environment Variables
To use the repo, you need to create an env.json file in the root with these variables.


1. "WANDB_API_KEY": "",
2. "HF_TOKEN" : "",
3. "OPENROUTER_KEY" : "", (not needed for training)
4. "CUDA_VISIBLE_DEVICES" : "0,1",
5. "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
6. "TORCHDYNAMO_DISABLE" : 1,
7. "NCCL_P2P_DISABLE": "0",
8. "NCCL_IB_DISABLE": "1",
9. "NCCL_SHM_DISABLE": "0",
10. "TORCH_NCCL_ASYNC_ERROR_HANDLING" : "1",
11. "TORCH_NCCL_BLOCKING_WAIT": "1",
12. "NCCL_DEBUG": "WARN",
13. "CUDA_DEVICE_ORDER": "PCI_BUS_ID"

You can use the defaults above for everything except the API keys, those need to be filled out.

Also, "CUDA_VISIBLE_DEVICES", should probably just be set to a range of how many GPUs you're using. So if you have a single GPU, you should set it to "0" instead of "0,1" which is how you'd do it if you have two GPUs.


## config

The config.json determines most aspects of a training run. There is a default one in the repo which you can use. Many of the parameters are self-explanatory, but I'll explain what the ones that are not do.

1. "elicitation_type": this determines which elicitation type we use for training. current options are "hp" and "up". hp means we use assistant-based honest persona elicitation, "up" means we use user-simulation based elicitation
2. "add_sp_token": this is a boolean that determines whether a unique <sp-token> is added before the intervention during training and evaluation. 1 = it is added.
3. system_tag: some models aren't trained to work with a system prompt, but our training data requires that, so to demarcate system prompts this tag sets a custom system demarcator. You should set it to "" if your model is configured to use system prompts with the hf chat template. Else you should set it to "<SYSTEM>" or something like that 

4. intervention_prefix" : This is just if you want to add a prefix before the intervention during training/test/evals. It could for example be "INTERVENTION:". But you can also keep it empty, then the raw intervention is just fed to the model.
5. review_prefix : ditto
6. flag_prefix"  : ditto. But this one you probably *should* use. The above are not needed/optional (mostly helpful for visibility/debugging) but this one you should keep.
7. intervention_type"   : this one determines which intervention type of the dataset we use. It can for example be "split_personality" or "honesty_scare".

8. train_topics: This is a list of strings that determines which topics we use for training (sycophancy, goal_misgeneralization etc)

9. validation_topics: ditto but for validation / cross-topic generalization eval

10. tags_to_filter" : These are the tags we filter, eg "confidence_below_4"

11. claude_frac : fraction of claude training-data we use.

12. first_heavy_log_step: First gradient step we do the full inter-topic and between-topic evaluations during training.

13. use_dummy_data: if you set this to 1 the training will train on random data.

14. validation_batch_size: number of samples of generation during the "heavy log" step its used. You should probably just set this as high as is allowed by you VRAM. A good starting point is 2x training batch size

15. val_samples_per_topic : Number of validation samples per topic for evaluation. (running on full set takes quite a long time)
16. lora_r : lora rank. more = more training capacity, but more VRAM and bigger adapters

## training

1 gpu:
    python train_lora.py
more gpus:
    run torchrun --nproc-per-node=(number_of_gpus_youre_using) train_lora.py

## checkpoints

The default is that when you run train_lora.py LoRA adapters will be stored in checkpoints/{wandb.run.name}/ at the end of each epoch.

Then you can load them using get_model if you pass a checkpoint_path (just point get_model to one of the paths generated above)


# Structure of Code
In root there is a train script called train_lora.py. Running this starts a training run. I'll below show a tree of function calls to make structure clear.

1. train_lora.py is ran
    1.1 get_model (FSDP_helpers.py) called and gets our model like this:
        1.1.1 this  function loads base model using huggingface
        1.1.2 wraps model with lora-adapters (potentially loads them from checkpoint if path is supplied)
        1.1.3 wraps model in FSDP and moves it to GPUs (if called in a distributed instance)
        (1.1.4) also sets some useful flags, like model.train()/model.eval() and applied activation checkpointing if train

    1.2 get_distributed_data (full_pipeline.py) is called which loads data like this
        1.2.1 calls full_pipeline which gets all data
            1.2.1.1 tokenized data is gathered from claude data source
            1.2.1.2 data is filtered and padded so its all seq_len and put into a tensor (for transformer parallelism)
            1.2.1.3 data is returned as a long list
        1.2.2 cuts it up into batches
        1.2.3 distibutes it accross ranks and returns the right data for each rank

    1.3 model and data is passed to our trainer class (trainer_LoRA.py), and trainer.train() is ran, which works like this:
        1.3.1 stuff like optimizer, wanb, scheduler is initialized
        1.3.2 we call train_epoch cfg.epochs number of times
            1.3.2.1 train_step is called on each batch in our dataset
                1.3.2.1.1 we do a forward-pass, compute loss, compute gradients, do optimizer step and so on.
                1.3.2.1.2 every k steps (defined by config) we also log loss to wandb, do a evals (heavy log) and save checkpoint
        1.3.3 training run is terminated

This is the core structure of a training run using this repo.

I left out the logic of the code that does data-generation and tokenization.

## Tokenization

We need to be quite careful about how stuff is tokenized, and some of this has to be hard-coded for different models, so I've created useful helper functions for this in scripts/data/data_utils.py that work for all open models you might wanna use. Warning: this code is somewhat complicated.

The main helper function is this:

1. apply_intervention(interaction, intervention): this function applies an intervention to an interaction and tokenizes so the model can start generating from that point. Like if you have the interaction [{"role":"user" : "content":"Am I a genius?"}, {"role":"assistant" :"content": "Yes, you are the smartest human being in the world!!!"}] and you want to analyze it for sycophancy, you might say tokens = apply_intervention(interaction, "Now I need to analyze my response for sycophancy.") and then do model.generate(tokens).
1.1 The function takes a tokenizer and model name. This is needed to tokenize stuff.
1.2 The model also takes elicitation type and apply_sp_token (same definitions as in cfg.json)

The other most important function is the data-source-specific tokenization function claude_tokenizer_and_mask. This takes in a sample, then tokenizes it and produces the right masks returned as a tuple.

Basically all the data works this way: we call get_data() for a specific source. For example get_data in claude_data. Then we get a bunch of samples which are basically interactions. These interactions are all passed thru a __tokenize_and_mask function. This produces a bunch of samples in a unified format, which are all pulled in the full pipeline described above.


There are also a ton of low-level helper functions ill briefly describe here.

1. get_model_tokenization_type: takes a model name, and returns data about how it tokenizes stuff
2. get_hp_token_id: takes a model name, returns the special id-token we finetune as the <split-personality> token for that model
3. get_user_prefix_tokens : model name, returns user prefix tokens (tokens model uses to know succeeding tokens are user-input)
4. get_assistant_prefix_tokens: model name, ditto, tokens so the model knows it should generate an assistant response
5. get_eot_suffix_tokens : model name -> the "end of turn" tokens, which should be appended at end of user/assistant turn so the model knows that turn is over
6. apply_chat_template_{open/close} closed=same as hf.apply_chat_template, open= avoids appending an end of turn token so the model can continue generating. eg apply_chat_template_open([{"user" : "Whats 5+5?"},{"assistant" : "10"}, {"user":"What about"}]) will get the model to continue simulating the user, but if you use closed it will assume the user-response was over and start generating an assistant response
7. pad (pads sample)
8. filter by length (obvious)
9. apply intervention (explained above)
10. debug prints stuff so you can look and see it makes sense



