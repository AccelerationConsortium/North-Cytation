# Lab Slack Agent

A modular Slack bot that monitors messages from a human user and a robot
messenger bot, then routes them to appropriate AI workflows via LangGraph.

## Architecture

```
Slack Event
  → event_handler.py       (receive & filter Slack messages)
  → graph/router.py        (classify source type and intent)
  → graph/main_graph.py    (LangGraph router)
  → analysis_graph.py      (experiment analysis workflow)
  → literature_graph.py    (literature research workflow)
  → discussion_agent.py    (follow-up discussion / plot request)
  → tools/                 (Python tools: data, plots, stats, literature)
  → Slack reply            (text + uploaded figures in thread)
```

## Setup

### 1. Copy the example env file and fill in your credentials

```bash
cp .env.example .env
# Edit .env with your real API keys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the bot

```bash
python app.py
```

## Project Structure

```
lab_slack_agent/
  app.py                    # Entry point — starts Slack Socket Mode
  config.py                 # Loads .env, exposes config constants
  slack/
    slack_client.py         # Slack WebClient wrapper
    event_handler.py        # Registers Slack event listeners
    file_upload.py          # post_text, upload_file, upload_multiple_files
  graph/
    state.py                # AgentState TypedDict (shared across all nodes)
    router.py               # classify_message, route_message nodes
    main_graph.py           # Top-level LangGraph graph + run_graph()
    analysis_graph.py       # Analysis sub-graph (10 nodes)
    literature_graph.py     # Literature sub-graph (6 nodes)
  agents/
    analysis_agent.py       # LLM node: write_analysis_report
    literature_agent.py     # LLM nodes: summarize_paper, synthesize_answer
    discussion_agent.py     # Follow-up discussion + plot_request handlers
  tools/
    data_tools.py           # load_run_data, summarize_dataframe, run_quality_checks
    plot_tools.py           # generate_standard_plots, get_existing_plot
    stats_tools.py          # detect_outliers, compare_to_previous_runs
    literature_tools.py     # search_semantic_scholar, search_pubmed, etc.
  memory/
    thread_memory.py        # SQLite persistence for thread context
  prompts/
    analysis_prompt.py      # System prompts for analysis LLM calls
    literature_prompt.py    # System prompts for literature LLM calls
    discussion_prompt.py    # System prompts for discussion LLM calls
  outputs/
    plots/                  # Generated PNG figures (gitignored)
    reports/                # Generated report text files (gitignored)
  tests/
    test_router.py          # Unit tests for message classification
```

## Message Routing

| Source             | Message pattern                    | Workflow                |
|--------------------|------------------------------------|-------------------------|
| `ROBOT_BOT_USER_ID`| "Experiment complete: run_id=..."  | analysis_workflow       |
| `ROBOT_BOT_USER_ID`| "Run finished: path=..."           | analysis_workflow       |
| `HUMAN_USER_ID`    | mentions bot + "analyze / data"    | data_analysis           |
| `HUMAN_USER_ID`    | mentions bot + "research / find"   | literature_research     |
| `HUMAN_USER_ID`    | mentions bot + "show / plot"       | plot_request            |
| `HUMAN_USER_ID`    | mentions bot + follow-up reply     | follow_up_discussion    |

## Environment Variables

| Variable              | Description                                   |
|-----------------------|-----------------------------------------------|
| `SLACK_BOT_TOKEN`     | xoxb-... bot token                            |
| `SLACK_APP_TOKEN`     | xapp-... app-level token (Socket Mode)        |
| `SLACK_SIGNING_SECRET`| Request signing secret                        |
| `ROBOT_BOT_USER_ID`   | Slack user ID of the robot messenger bot      |
| `HUMAN_USER_ID`       | Slack user ID of the human operator           |
| `LLM_PROVIDER`        | "openai" or "anthropic"                       |
| `LLM_MODEL`           | Model name, e.g. "gpt-4o" or "claude-3-5-sonnet-20241022" |
| `OPENAI_API_KEY`      | OpenAI API key (if LLM_PROVIDER=openai)       |
| `ANTHROPIC_API_KEY`   | Anthropic API key (if LLM_PROVIDER=anthropic) |
| `DATA_ROOT`           | Base path to experiment data files            |

## Adding New Workflows

1. Add a new intent string to `graph/router.py` (`INTENT_KEYWORDS`)
2. Create a new graph file in `graph/` or a handler in `agents/`
3. Register the new node + edge in `graph/main_graph.py`
4. Add relevant tools to `tools/`
5. Add prompts to `prompts/`
