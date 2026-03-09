from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MATH-plus", split="train")


def concat(example):
    example["text"] = example["instruction"] + "\n" + example["output"]
    return example


ds = ds.map(concat, num_proc=8)
ds = ds.remove_columns(["instruction", "output"])
ds.save_to_disk("/scratch4/workspace/rbhirud_umass_edu-RL_experiments/colm/data/math_plus_text")
print("Done.", len(ds), "examples")
