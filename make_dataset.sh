  /home/lr/kent/MoM/.venv/bin/python training/preprocess.py \
    --dataset gmongaras/SlimPajama-627B_Reupload \
    --split train \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --seq_len 2048 \
    --max_tokens 20000000000 \
    --streaming \
    --output data/slimpajama_mistral_20B_seq2048
