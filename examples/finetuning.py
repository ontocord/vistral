# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import sys
sys.path.append("<YOUR_ABSOLUTE_PATH_TO>/llama-recipes/src")

import fire
from llama_recipes.finetuning import main

if __name__ == "__main__":
    fire.Fire(main)