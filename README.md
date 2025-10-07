<h1> North Robot - Cytation 5 uSDL </h1>

All code written by Owen Melville with contributions by Ilya Yakavets, Monique Ngan and ChatGPT. 


<h2>File List:</h2>

- <b>Locator.py</b>: Contains a list of physical locations used by the Robot

- <b>North_Safe.py</b> Contains the reusable code for the control of the North Robot and the North Track which goes between the Cytation 5 and the North Robot. This is the most in-depth file.
  
- <b> biotek.py </b> Contains the code for control of the Cytation 5. Ilya has a newer version with more capabilities.
  
- <b> master_usdl_coordinator.py </b> Contains reusable workflow code that uses multiple instruments: Cytation 5, North Robot & Track, Photoreactor
  
- <b> north_gui.py </b> Mostly written by chatGPT, intended as a base for a resuable gui for different workflows
  
- <b> photoreactor_controller </b> Controls the photoreactor via Raspberry Pi Pico. reactor_test.py is the local Pico program.
  
- <b> requirements.txt </b> What packages need installation to use all this?
  
- <b> slack_agent.txt </b> Controls messages sent to slack from the robot. Mostly written by ChatGPT.

<h2> Folder List: </h2>

- <b> analysis </b>: Contains programs that analyze data from the robot to produce a result usable by the recommenders

- <b> photo-reactor </b>: Contains the RPi program for the photoreactor.
  
- <b> recommenders: </b> Contains programs (eg using Baybe) to recommend conditions
  
- <b> status: </b> Contains transferrable data structures (eg vial status). In the long run these will represent objects with states that transfer between modules
  
- <b> tests: </b> Contains short test or commonly used programs for the setup.
  
- <b> workflows: </b> Contains active workflows that are designed for the setup, such as the color matching workflow. 

<h2> Sample Workflow: </h2>

The sample workflow contained in this video is labelled and contains the programming elements in the sample_workflow.py program in the "workflows" directory. 

[![Sample Workflow](https://img.youtube.com/vi/b5xtcwcI-eg/0.jpg)](https://www.youtube.com/watch?v=b5xtcwcI-eg)

## Literature Relevance Labeling (LLM Assisted)

Pipeline components live under `research/literature_search/`:

1. Fetch + extract + score produce `scored_candidates.csv` and parsed JSONL.
2. Gating selects top percentile candidates; exploration sampling can add mid/low segments.
3. `prompt_preview.py` (already run) renders offline prompts for inspection.
4. `llm_label.py` sends prompts to an OpenAI-compatible model and writes JSONL labels.

### .env Setup
Create a `.env` file at repo root (already git-ignored via `*.env`) with:

```
OPENAI_API_KEY=sk-...
# Optional self-host / proxy endpoint:
# OPENAI_BASE_URL=https://your-proxy.example.com
```

### Install Dependencies
Ensure `openai` and `python-dotenv` were installed (added to `requirements.txt`).

### Dry Run (Schema / Flow Check)
```
python research/literature_search/scripts/llm_label.py --input research/literature_search/data/prompt_preview_new.jsonl --output research/literature_search/data/llm_labels_dry.jsonl --dry-run
```

### Real Labeling Run
```
python research/literature_search/scripts/llm_label.py --input research/literature_search/data/prompt_preview_new.jsonl --output research/literature_search/data/llm_labels.jsonl --model gpt-4o-mini --rate-limit-per-min 40
```

`--rate-limit-per-min` is a client-side throttle; adjust per your quota. Output lines contain:

```
{"id": "...", "model_output": { ... schema ... }}
```

### Next Steps (Optional)
- Aggregate labels and fit calibration (logistic) from axis vector -> relevance probability.
- Consider renaming `device_penalty` axis after initial calibration so all axes share the same directionality (higher=better) before probability fitting.


