from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import sys
from time import time

def main(args):
    model_name = os.getenv("MODEL_NAME")
    if model_name is None:
        model_name = "meta-llama/Llama-2-7b-chat-hf"


    print("Loading tokenizer and model for model name %s" % (model_name))

    start = time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_4bit=True,
                                            use_auth_token=True
                                            )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    end = time()
    print(f"Loading model took {end-start} seconds")

    while(True):
        # Get user input:
        prompt = input("Enter the prompt you would like to give the model:\n>")
        if len(prompt) == 0:
            continue

        start = time()
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1000,
        )
        end = time()

        for seq_ind,seq in enumerate(sequences):
            print(f"************* Response {seq_ind} *************")
            print(f"{seq['generated_text']}")
            print(f"************* End response {seq_ind} ************\n\n")

        print(f"Response generated in {end-start} s")

if __name__ == '__main__':
    main(sys.argv[1:])
