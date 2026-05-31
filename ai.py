"""Run the game bot with a trained local model or a vision LLM.

The original project only supported replaying a Keras model trained from local
screen/input examples. The default mode now uses a multimodal model from either
OpenAI or a local Ollama server to look at the current screen and return a small,
structured gameplay action.
"""

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from io import BytesIO
from time import sleep

import numpy as np
from PIL import ImageGrab
from keras.models import model_from_json

from game_control import click, get_key, press, release
from predict import predict

DEFAULT_PROVIDER = os.getenv("GAME_BOT_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("OPENAI_GAME_MODEL", "gpt-5.2")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_GAME_MODEL", "gemma3")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_GOAL = os.getenv(
    "GAME_BOT_GOAL",
    "Assist gameplay on the visible screen. Choose the next useful, safe input.",
)
ALLOWED_ACTIONS = {"none", "press", "release", "tap", "click"}
ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "key": {"type": ["string", "null"]},
        "x": {"type": ["integer", "null"]},
        "y": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "key", "x", "y", "reason"],
}


def capture_screen():
    """Capture the current screen as a PIL image."""
    return ImageGrab.grab()


def encode_screen_for_llm(screen, as_data_url=True):
    """Encode a PIL screenshot as compact JPEG base64 for a vision model."""
    rgb_screen = screen.convert("RGB")
    buffer = BytesIO()
    rgb_screen.save(buffer, format="JPEG", quality=70, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if as_data_url:
        return f"data:image/jpeg;base64,{encoded}"
    return encoded


def build_llm_prompt(goal, allowed_keys):
    """Build the instruction prompt sent with every screenshot."""
    keys = ", ".join(allowed_keys)
    return (
        "You are a gameplay assistant controlling only the listed inputs. "
        "Inspect the screenshot and choose exactly one next action that helps the player's goal. "
        "Do not attempt to control anything outside the game. Prefer 'none' when uncertain, "
        "on menus, or when an action could be unsafe. Return only JSON with keys: "
        "action, key, x, y, reason. action must be one of none, press, release, tap, click. "
        "For key actions, key must be one of the allowed keys. For click, x and y are screen "
        "coordinates from the screenshot. Keep reason short.\n\n"
        f"Goal: {goal}\n"
        f"Allowed keys: {keys}"
    )


def parse_action_text(text):
    """Parse and validate a JSON action returned by any model provider."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    action = json.loads(text)
    action_name = action.get("action", "none")
    if action_name not in ALLOWED_ACTIONS:
        return {"action": "none", "reason": f"Ignored unsupported action: {action_name}"}
    return action


def parse_openai_action(response):
    """Extract the JSON action from an OpenAI Responses API result."""
    text = getattr(response, "output_text", "") or ""
    if not text:
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text += getattr(content, "text", "")
    return parse_action_text(text)


def parse_ollama_action(response_body):
    """Extract the JSON action from an Ollama /api/chat response body."""
    message = response_body.get("message", {})
    return parse_action_text(message.get("content", ""))


def ask_openai_for_action(client, model, screen, goal, allowed_keys):
    """Send the screenshot to OpenAI and return one structured gameplay action."""
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_llm_prompt(goal, allowed_keys)},
                    {"type": "input_image", "image_url": encode_screen_for_llm(screen)},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "gameplay_action",
                "strict": True,
                "schema": ACTION_SCHEMA,
            }
        },
    )
    return parse_openai_action(response)


def normalize_ollama_host(host):
    """Return an Ollama host URL with a scheme and no trailing slash."""
    host = host.rstrip("/")
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}"


def ask_ollama_for_action(host, model, screen, goal, allowed_keys):
    """Send the screenshot to Ollama and return one structured gameplay action."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_llm_prompt(goal, allowed_keys),
                "images": [encode_screen_for_llm(screen, as_data_url=False)],
            }
        ],
        "format": ACTION_SCHEMA,
        "stream": False,
        "think": False,
    }
    request = urllib.request.Request(
        f"{normalize_ollama_host(host)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start Ollama and pull a vision model "
            f"such as '{model}' first."
        ) from exc

    return parse_ollama_action(response_body)


def apply_llm_action(action, allowed_keys):
    """Apply one validated LLM action through pynput."""
    action_name = action.get("action", "none")
    key = action.get("key")

    if action_name == "none":
        return

    if action_name in {"press", "release", "tap"}:
        if key not in allowed_keys:
            print(f"Skipping disallowed key from LLM: {key}")
            return
        if action_name == "press":
            press(key)
        elif action_name == "release":
            release(key)
        else:
            press(key)
            sleep(0.05)
            release(key)
        return

    if action_name == "click":
        x = action.get("x")
        y = action.get("y")
        if isinstance(x, int) and isinstance(y, int):
            click(x, y)


def run_llm_assistant(provider, model, goal, interval, allowed_keys, ollama_host):
    """Run the screenshot-to-action loop using the configured LLM provider."""
    client = None
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()

    print(f"LLM gameplay assistant started with {provider} model '{model}'.")
    print("Press Ctrl-C in this terminal to stop.")

    while True:
        screen = capture_screen()
        if provider == "openai":
            action = ask_openai_for_action(client, model, screen, goal, allowed_keys)
        else:
            action = ask_ollama_for_action(ollama_host, model, screen, goal, allowed_keys)
        print(f"LLM action: {action}")
        apply_llm_action(action, allowed_keys)
        sleep(interval)


def load_keras_model():
    """Load the legacy Keras model from Data/Model."""
    with open("Data/Model/model.json", "r", encoding="utf-8") as model_file:
        model_json = model_file.read()
    model = model_from_json(model_json)
    model.load_weights("Data/Model/weights.h5")
    return model


def run_trained_model():
    """Run the original local neural-network control loop."""
    model = load_keras_model()
    print("Local trained model started.")

    while True:
        screen = np.array(capture_screen())[:, :, :3]
        prediction = predict(model, screen)
        y = prediction[0].tolist() if hasattr(prediction, "shape") and len(prediction.shape) > 1 else prediction

        if y == [0, 0, 0, 0]:
            continue
        if y[0] == -1 and y[1] == -1:
            key = get_key(int(y[3]))
            if y[2] == 1:
                press(key)
            else:
                release(key)
        elif y[2] == 0 and y[3] == 0:
            click(int(y[0]), int(y[1]))
        else:
            click(int(y[0]), int(y[1]))
            key = get_key(int(y[3]))
            if y[2] == 1:
                press(key)
            else:
                release(key)


def parse_args():
    parser = argparse.ArgumentParser(description="Assist gameplay from the current screen.")
    parser.add_argument(
        "--mode",
        choices=["llm", "trained"],
        default="llm",
        help="Use vision LLM assistance or the legacy locally trained model.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=DEFAULT_PROVIDER,
        help="LLM provider for --mode llm.",
    )
    parser.add_argument("--model", default=None, help="Model for --mode llm. Defaults by provider.")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Gameplay objective for the LLM assistant.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between LLM screen checks.")
    parser.add_argument(
        "--allowed-keys",
        default="w,a,s,d,space,up,down,left,right,enter,esc",
        help="Comma-separated key allowlist for the LLM assistant.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL for --provider ollama.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "trained":
        run_trained_model()
        return

    model = args.model or (DEFAULT_OLLAMA_MODEL if args.provider == "ollama" else DEFAULT_LLM_MODEL)
    allowed_keys = [key.strip() for key in args.allowed_keys.split(",") if key.strip()]
    run_llm_assistant(args.provider, model, args.goal, args.interval, allowed_keys, args.ollama_host)


if __name__ == "__main__":
    main()
