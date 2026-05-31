# Arda Mavi
from pynput.mouse import Button, Controller as Mouse
from pynput.keyboard import Controller as Keyboard, Key

# For encoding keyboard keys:
def get_keys():
    return ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace', 'browserback', 'browserfavorites', 'browserforward', 'browserhome', 'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear', 'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete', 'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20', 'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja', 'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail', 'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack', 'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn', 'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn', 'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator', 'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab', 'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen', 'command', 'option', 'optionleft', 'optionright']

def get_key(id):
    return get_keys()[id]

def get_id(key):
    return get_keys().index(normalize_key(key))

keyboard = Keyboard()
mouse = Mouse()

SPECIAL_KEYS = {
    'alt': Key.alt,
    'altleft': Key.alt_l,
    'altright': Key.alt_r,
    'backspace': Key.backspace,
    'capslock': Key.caps_lock,
    'cmd': Key.cmd,
    'command': Key.cmd,
    'ctrl': Key.ctrl,
    'ctrlleft': Key.ctrl_l,
    'ctrlright': Key.ctrl_r,
    'del': Key.delete,
    'delete': Key.delete,
    'down': Key.down,
    'end': Key.end,
    'enter': Key.enter,
    'esc': Key.esc,
    'escape': Key.esc,
    'f1': Key.f1,
    'f2': Key.f2,
    'f3': Key.f3,
    'f4': Key.f4,
    'f5': Key.f5,
    'f6': Key.f6,
    'f7': Key.f7,
    'f8': Key.f8,
    'f9': Key.f9,
    'f10': Key.f10,
    'f11': Key.f11,
    'f12': Key.f12,
    'home': Key.home,
    'insert': Key.insert,
    'left': Key.left,
    'pagedown': Key.page_down,
    'pageup': Key.page_up,
    'pause': Key.pause,
    'printscreen': Key.print_screen,
    'right': Key.right,
    'shift': Key.shift,
    'shiftleft': Key.shift_l,
    'shiftright': Key.shift_r,
    'space': Key.space,
    'tab': Key.tab,
    'up': Key.up,
}


def normalize_key(key):
    if hasattr(key, 'char') and key.char is not None:
        return key.char
    if hasattr(key, 'name'):
        return key.name
    key_text = str(key)
    if key_text.startswith('Key.'):
        return key_text[4:]
    return key_text.strip("'")


def resolve_key(key):
    if not isinstance(key, str):
        return key
    return SPECIAL_KEYS.get(key.lower(), key)

# Mouse:
def move(x, y):
    mouse.position = (x, y)
    return

def scroll(x, y):
    mouse.scroll(x, y)
    return

def click(x, y):
    mouse.position = (x, y)
    mouse.click(Button.left)
    return

# Keyboard:
def press(key):
    keyboard.press(resolve_key(key))
    return

def release(key):
    keyboard.release(resolve_key(key))
    return
