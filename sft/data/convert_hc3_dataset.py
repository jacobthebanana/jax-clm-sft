"""
Utilities for converting contrastive HC dataset to text-label dataset.

Label: 0 for human, 1 for AI.
"""
import argparse
import re
from typing import Any, Dict, List, Literal

import datasets

InputDatasetKeys = Literal["question", "human_answers", "chatgpt_answers"]
InputDatasetExample = Dict[InputDatasetKeys, Any]
OutputDatasetKeys = Literal["prompt", "response"]
OutputDatasetBatch = Dict[OutputDatasetKeys, List[Any]]


def convert_dataset_batch(input_example: InputDatasetExample) -> OutputDatasetBatch:
    output: OutputDatasetBatch = {"prompt": [], "response": []}

    for question, ai_answers in zip(
        input_example["question"],
        input_example["chatgpt_answers"],
    ):
        if len(question.split()) <= 128:
            for answer in ai_answers[0:1]:
                output["prompt"].append(question)
                output["response"].append(answer)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl_path")
    parser.add_argument("output_dataset_path")
    args = parser.parse_args()

    input_jsonl_path: str = args.input_jsonl_path
    output_dataset_path: str = args.output_dataset_path

    read_instruction = {
        "train": datasets.ReadInstruction("train", from_=0, to=65, unit="%"),
        "validation": datasets.ReadInstruction("train", from_=65, to=75, unit="%"),
        "test": datasets.ReadInstruction("train", from_=75, to=100, unit="%"),
    }
    dataset_dict = datasets.load_dataset(
        "json",
        data_files=input_jsonl_path,
        split=read_instruction,  # type: ignore
    )
    processed_dataset_dict = dataset_dict.map(
        convert_dataset_batch,
        remove_columns=dataset_dict["train"].column_names,  # type: ignore
        batched=True,
    )

    print(processed_dataset_dict)

    assert isinstance(processed_dataset_dict, datasets.DatasetDict)
    processed_dataset_dict.save_to_disk(output_dataset_path)
