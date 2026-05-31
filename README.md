# Game Bot
### By Arda Mavi

Artificial intelligence that can assist gameplay by watching your screen.

The project now has two modes:

1. **LLM gameplay assistant (default)** - captures the visible screen, sends it to a modern OpenAI multimodal model, and applies one safe keyboard or mouse action from an allowlist.
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

- `OPENAI_API_KEY` - required by the OpenAI SDK.
- `OPENAI_GAME_MODEL` - defaults to `gpt-5.2`.
- `GAME_BOT_GOAL` - default gameplay instruction when `--goal` is omitted.

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

- The LLM assistant requires an OpenAI API key and network access.
- Screen capture and input control permissions may need to be enabled in your operating system.
- This project is still being worked on.
