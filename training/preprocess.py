# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    ctx_len: int = None,
    return_offsets: bool = False
) -> Dict[str, List[List[int]]]:
    """
    Tokenize the input text and split into chunks of specified context length.

    Args:
        examples:
            Dictionary containing the input text.
        tokenizer:
            Initialized tokenizer.
        seq_len:
            Total sequence length for each training sample. Default: 2048.
        ctx_len:
            Max contiguous length to preserve (will not be split). Default: `None`.
        return_offsets:
            Return cumulative offsets for concatenated inputs. Default: `False`.

    Returns:
        Dictionary containing tokenized and chunked input ids, and optionally offsets.
    """
    text = examples['text']
    input_ids = tokenizer(text)['input_ids']
    # further split each input into chunks of length `ctx_len` if provided
    if ctx_len is not None:
        input_ids = [seq[i:i+ctx_len] for seq in input_ids for i in range(0, len(seq), ctx_len)]
    lengths = [len(seq) for seq in input_ids]
    if len(lengths) == 0:
        return {'input_ids': [], 'offsets': []} if return_offsets else {'input_ids': []}
    lens = []
    running_len = 0
    for length in lengths:
        running_len += length
        lens.append(running_len)
    total_len = lens[-1] // seq_len * seq_len

    input_ids = list(chain(*input_ids))
    # each yielded sample is of length `seq_len`
    input_ids = [input_ids[i:i+seq_len] for i in range(0, total_len, seq_len)]

    if not return_offsets:
        return {'input_ids': input_ids}

    # insert boundaries into cumulative offsets
    import torch
    lens = torch.tensor(lens)
    offsets = torch.cat((lens, torch.arange(0, total_len, seq_len))).unique().sort()[0] % seq_len
    # split offsets according the start positions
    offsets = [i.tolist() + [seq_len] for i in offsets.tensor_split(torch.where(offsets.eq(0))[0][1:])][:len(input_ids)]
    return {'input_ids': input_ids, 'offsets': offsets}


def preprocess(
    dataset: str,
    name: Optional[str] = None,
    split: str = 'train',
    seed: int = 42,
    output: str = 'data',
    tokenizer: str = 'mistralai/Mistral-7B-v0.1',
    num_proc: int = 64,
    batch_size: int = 2048,
    seq_len: int = 2048,
    ctx_len: int = None,
    return_offsets: bool = False,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    shuffle_buffer_size: int = 10_000,
    generator_cache_dir: Optional[str] = None,
    legacy_output_path: bool = False,
    text_column: str = 'text'
) -> None:
    """
    Load, tokenize, and save the processed dataset.

    Args:
        dataset:
            Path or name of the dataset. Default: 'HuggingFaceFW/fineweb-edu'.
        name:
            Name of the dataset configuration. Default: `None`.
        split:
            Dataset split to process. Default: 'train'.
        seed:
            Random seed for shuffling the dataset. Default: 42.
        output:
            Output directory. Default: 'data'.
        tokenizer:
            Tokenizer name. Default: 'mistralai/Mistral-7B-v0.1'.
        num_proc:
            Number of processes for parallel processing. Default: 64.
        batch_size:
            Batch size for processing. Default: 2048.
        seq_len:
            Total sequence length for each training sample. Default: 2048.
        ctx_len:
            Max contiguous length to preserve (will not be split). Default: `None`.
        return_offsets:
            Return cumulative offsets for concatenated inputs. Default: `False`.
        streaming:
            Stream the source dataset instead of materializing it before tokenization.
        max_tokens:
            Maximum number of tokens to emit. The saved dataset contains only full
            `seq_len` chunks, so the final count is rounded down to a multiple of `seq_len`.
        shuffle_buffer_size:
            Buffer size for streaming shuffle. Set to 0 to disable.
        generator_cache_dir:
            Cache directory used by `Dataset.from_generator`. Defaults to
            `<output>.generator_cache` in streaming mode.
        legacy_output_path:
            Save to `output/dataset/name/split` for compatibility with the original script.
        text_column:
            Name of the text column in the source dataset.
    """
    tokenized_path = (
        f'{output}/{dataset}/{name}/{split}'
        if legacy_output_path and name is not None
        else f'{output}/{dataset}/{split}'
        if legacy_output_path
        else output
    )

    if ctx_len is not None and ctx_len > seq_len:
        raise ValueError(f'ctx_len ({ctx_len}) must be less than or equal to seq_len ({seq_len})')
    if streaming and return_offsets:
        raise ValueError('--return_offsets is not supported with --streaming')

    logger.info(f'Loading tokenizer {tokenizer}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f'Tokenizer initialized:\n {tokenizer}')

    if streaming:
        preprocess_streaming(
            dataset=dataset,
            name=name,
            split=split,
            seed=seed,
            output=tokenized_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            ctx_len=ctx_len,
            max_tokens=max_tokens,
            shuffle_buffer_size=shuffle_buffer_size,
            generator_cache_dir=generator_cache_dir,
            text_column=text_column
        )
        return

    logger.info(f'Loading dataset: {dataset}')
    dataset = load_dataset(dataset, name=name, split=split)
    dataset = dataset.shuffle(seed=seed)
    logger.info(f'Dataset loaded: {dataset}')

    remove_columns = list(next(iter(dataset)).keys())
    logger.info(f'Tokenizing and processing the dataset with batch size {batch_size}')
    dataset = dataset.map(
        lambda examples: tokenize(examples, tokenizer, seq_len, ctx_len, return_offsets),
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Running tokenizer on dataset"
    )

    logger.info(f'Saving processed dataset to {tokenized_path}')
    dataset.save_to_disk(tokenized_path, num_proc=num_proc)


def preprocess_streaming(
    dataset: str,
    name: Optional[str],
    split: str,
    seed: int,
    output: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    seq_len: int,
    ctx_len: Optional[int],
    max_tokens: Optional[int],
    shuffle_buffer_size: int,
    generator_cache_dir: Optional[str],
    text_column: str
) -> None:
    """
    Stream, tokenize, chunk, and save a fixed-token-budget dataset.

    This path is intended for very large corpora such as SlimPajama where loading
    or mapping the full dataset before stopping at a token budget is impractical.
    """
    if max_tokens is not None and max_tokens < seq_len:
        raise ValueError(f'max_tokens ({max_tokens}) must be at least seq_len ({seq_len})')

    target_tokens = None if max_tokens is None else (max_tokens // seq_len) * seq_len
    if max_tokens is not None and target_tokens != max_tokens:
        logger.warning(
            f'max_tokens={max_tokens} is not divisible by seq_len={seq_len}; '
            f'saving {target_tokens} tokens in full chunks'
        )

    logger.info(f'Loading streaming dataset: {dataset}')
    stream = load_dataset(dataset, name=name, split=split, streaming=True)
    if shuffle_buffer_size > 0:
        logger.info(f'Shuffling streaming dataset with seed={seed}, buffer_size={shuffle_buffer_size}')
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    if generator_cache_dir is None:
        generator_cache_dir = f'{output}.generator_cache'

    features = Features({'input_ids': Sequence(Value('int32'))})
    logger.info(f'Stream-tokenizing with generator cache {generator_cache_dir}')
    tokenized = Dataset.from_generator(
        lambda: iter_streaming_chunks(
            stream=stream,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            ctx_len=ctx_len,
            target_tokens=target_tokens,
            text_column=text_column
        ),
        features=features,
        cache_dir=generator_cache_dir
    )
    logger.info(f'Generated dataset: {tokenized}')
    logger.info(f'Saving processed dataset to {output}')
    tokenized.save_to_disk(output)


def iter_streaming_chunks(
    stream: Any,
    tokenizer: AutoTokenizer,
    batch_size: int,
    seq_len: int,
    ctx_len: Optional[int],
    target_tokens: Optional[int],
    text_column: str
) -> Iterator[Dict[str, List[int]]]:
    buffer: List[int] = []
    emitted_tokens = 0
    texts: List[str] = []

    def emit_from_texts(batch_texts: List[str]) -> Iterator[Dict[str, List[int]]]:
        nonlocal buffer, emitted_tokens

        tokenized = tokenizer(batch_texts, return_attention_mask=False)['input_ids']
        if ctx_len is not None:
            tokenized = [seq[i:i+ctx_len] for seq in tokenized for i in range(0, len(seq), ctx_len)]

        for ids in tokenized:
            buffer.extend(ids)
            while len(buffer) >= seq_len:
                if target_tokens is not None and emitted_tokens >= target_tokens:
                    return
                yield {'input_ids': buffer[:seq_len]}
                del buffer[:seq_len]
                emitted_tokens += seq_len

    for sample in stream:
        if target_tokens is not None and emitted_tokens >= target_tokens:
            break
        text = sample.get(text_column)
        if text is None:
            raise KeyError(f"Text column '{text_column}' was not found in sample keys: {list(sample.keys())}")
        texts.append(text)
        if len(texts) == batch_size:
            yield from emit_from_texts(texts)
            texts = []

    if texts and (target_tokens is None or emitted_tokens < target_tokens):
        yield from emit_from_texts(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="Path or name of the dataset")
    parser.add_argument("--name", default=None, help="Name of the dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1", help="Tokenizer name")
    parser.add_argument("--num_proc", type=int, default=64, help="Number of processes for parallel processing")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing")
    parser.add_argument("--seq_len", type=int, default=2048, help="Total sequence length for each training sample")
    parser.add_argument("--ctx_len", type=int, default=None, help="Max contiguous length to preserve (will not be split)")
    parser.add_argument("--return_offsets", action="store_true", help="Return cumulative offsets for concatenated inputs")
    parser.add_argument("--streaming", action="store_true", help="Stream the dataset and stop at --max_tokens")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to save")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000, help="Streaming shuffle buffer size; 0 disables shuffle")
    parser.add_argument("--generator_cache_dir", default=None, help="Cache directory for Dataset.from_generator")
    parser.add_argument("--legacy_output_path", action="store_true", help="Save to output/dataset/name/split like the original script")
    parser.add_argument("--text_column", default="text", help="Text column to tokenize")
    args = parser.parse_args()

    preprocess(
        dataset=args.dataset,
        name=args.name,
        split=args.split,
        seed=args.seed,
        output=args.output,
        tokenizer=args.tokenizer,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        ctx_len=args.ctx_len,
        return_offsets=args.return_offsets,
        streaming=args.streaming,
        max_tokens=args.max_tokens,
        shuffle_buffer_size=args.shuffle_buffer_size,
        generator_cache_dir=args.generator_cache_dir,
        legacy_output_path=args.legacy_output_path,
        text_column=args.text_column
    )
