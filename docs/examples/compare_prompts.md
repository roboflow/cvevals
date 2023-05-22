# Compare Prompts

You can compare different prompts for CLIP and Grounding DINO using the `compare` examples.

To use these examples, open up the appropriate file (i.e. `examples/clip_example.py`) and update the `evals` dictionary with the class names and confidence levels you want to evaluate.

Then, run the script using all of the required arguments. To find the arguments for a script, run the script with no arguments, like in this case for the Grounding DINO compare example:

```
python3 examples/dino_compare_example.py
```

## Available Comparisons

- `examples/clip_compare_example.py`: Compare CLIP prompts
- `examples/dino_compare_example.py`: Compare Grounding DINO prompts