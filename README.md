# Game Bot
### By Arda Mavi

Artificial intelligence that can assist gameplay by watching your screen.

The project now has two modes:

1. **LLM gameplay assistant (default)** - captures the visible screen, sends it to a modern multimodal model through OpenAI or a local Ollama server, and applies one safe keyboard or mouse action from an allowlist.
2. **Legacy trained model** - replays a locally trained Keras model from captured examples.

## LLM gameplay assistant

The default mode uses the OpenAI Responses API with a vision-capable model. Set your API key first:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then open the game window and run:

```bash
python3 ai.py --goal "Help me play this game and avoid danger"
```

Useful options:

```bash
python3 ai.py \
  --model gpt-5.2 \
  --goal "Win the race while staying on the track" \
  --allowed-keys "w,a,s,d,space,left,right,up,down" \
  --interval 1.0
```

Environment variables are also supported:

- `OPENAI_API_KEY` - required by the OpenAI SDK when `--provider openai` is used.
- `GAME_BOT_PROVIDER` - defaults to `openai`; set to `ollama` to use Ollama by default.
- `OPENAI_GAME_MODEL` - defaults to `gpt-5.2`.
- `OLLAMA_GAME_MODEL` - defaults to `gemma3`.
- `OLLAMA_HOST` - defaults to `http://localhost:11434`.
- `GAME_BOT_GOAL` - default gameplay instruction when `--goal` is omitted.

### Using Ollama instead of OpenAI

Ollama can work for this project when you use a vision-capable model. Install Ollama, pull a model that supports images, and keep the Ollama server running:

```bash
ollama pull gemma3
```

Then run the bot with the Ollama provider:

```bash
python3 ai.py \
  --provider ollama \
  --model gemma3 \
  --goal "Help me play this game and avoid danger"
```

If your Ollama server is not on the default local URL, pass it explicitly:

```bash
python3 ai.py --provider ollama --ollama-host http://localhost:11434
```

Ollama support uses `/api/chat` with a screenshot in the `images` array and requests a JSON-schema action response. Local models can be slower or less capable than hosted frontier models, so start with a longer `--interval` and a small `--allowed-keys` list.


Safety notes:

- The LLM is restricted to the keys in `--allowed-keys` and a single mouse click action.
- Keep the game focused and stop the bot with `Ctrl-C` in the terminal.
- Prefer a slower `--interval` while testing so you can observe every action.

## Playing with the legacy trained model

1. Open your desired game after training the local model.
2. Run:

```bash
python3 ai.py --mode trained
```

## Creating a training dataset for legacy mode

1. Run:

```bash
python3 create_dataset.py
```

2. Play your desired game.
3. Stop `create_dataset.py` with `Ctrl-C` in the terminal.

## Model training for legacy mode

```bash
python3 train.py
```

## Using TensorBoard

```bash
tensorboard --logdir=Data/Checkpoints/logs
```

## Installation

```bash
pip3 install -r requirements.txt
```

## Important notes

- The OpenAI provider requires an OpenAI API key and network access; the Ollama provider requires a running local Ollama server and a vision-capable model.
- Screen capture and input control permissions may need to be enabled in your operating system.
- This project is still being worked on.
