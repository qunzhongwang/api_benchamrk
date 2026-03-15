# API Concurrency Stress Tester

High-concurrency stress tester for OpenAI-compatible API endpoints (e.g. Poe). Fires 100 synthetic questions with configurable parallelism and gives you rich terminal output with latency percentiles, throughput, and per-request details.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Usage

```bash
# Set your API key (or pass --api-key each time)
export POE_API_KEY=your_key_here

# Quick smoke test: 5 questions, 3 concurrent
python main.py -c 3 -n 5

# Medium load: 20 questions, 10 concurrent
python main.py -c 10 -n 20

# Full blast: all 100 questions, 50 concurrent
python main.py -c 50

# Custom endpoint and model
python main.py --base-url https://api.poe.com/v1 --model gpt-5.3-codex -c 30

# Show all per-request details (not just first 20)
python main.py -c 20 --show-all

# Don't shuffle question order
python main.py -c 20 --no-shuffle

# Limit response length
python main.py -c 20 --max-tokens 128
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | `$POE_API_KEY` | API key |
| `--base-url` | `https://api.poe.com/v1` | API base URL |
| `--model` | `gpt-5.3-codex` | Model name |
| `-c`, `--concurrency` | `20` | Max concurrent requests |
| `-n`, `--num-questions` | `100` | Number of questions (max 100) |
| `--max-tokens` | `256` | Max tokens per response |
| `--shuffle` / `--no-shuffle` | shuffle on | Randomize question order |
| `--show-all` | off | Show all per-request rows |

## Output

- Live progress bar with per-request success/fail indicators
- Summary table: success/fail counts, wall time, throughput (req/s)
- Latency stats: min, avg, max, p50, p95, p99
- Token usage: prompt, completion, total
- Error table (if any failures)
- Per-request detail table with response previews
