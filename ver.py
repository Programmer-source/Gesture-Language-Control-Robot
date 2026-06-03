import cv2
import math
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1
)

mp_draw = mp.solutions.drawing_utils

# Колір (BGR) та назва для кожної з 21 точки руки
point_colors = [
    ((0, 0, 255),     "червоний"),       # 0  - WRIST
    ((0, 128, 255),   "помаранчевий"),   # 1  - THUMB_CMC
    ((0, 255, 255),   "жовтий"),         # 2  - THUMB_MCP
    ((0, 255, 128),   "салатовий"),      # 3  - THUMB_IP
    ((0, 255, 0),     "зелений"),        # 4  - THUMB_TIP
    ((128, 255, 0),   "смарагдовий"),    # 5  - INDEX_MCP
    ((255, 255, 0),   "бірюзовий"),      # 6  - INDEX_PIP
    ((255, 128, 0),   "блакитний"),      # 7  - INDEX_DIP
    ((255, 0, 0),     "синій"),          # 8  - INDEX_TIP
    ((255, 0, 128),   "індиго"),         # 9  - MIDDLE_MCP
    ((255, 0, 255),   "фіолетовий"),     # 10 - MIDDLE_PIP
    ((128, 0, 255),   "пурпурний"),      # 11 - MIDDLE_DIP
    ((180, 105, 255), "рожевий"),        # 12 - MIDDLE_TIP
    ((33, 67, 101),   "коричневий"),     # 13 - RING_MCP
    ((0, 128, 128),   "оливковий"),      # 14 - RING_PIP
    ((128, 128, 0),   "морський"),       # 15 - RING_DIP
    ((0, 0, 128),     "бордовий"),       # 16 - RING_TIP
    ((0, 215, 255),   "золотий"),        # 17 - PINKY_MCP
    ((192, 192, 192), "сріблястий"),     # 18 - PINKY_PIP
    ((60, 20, 220),   "малиновий"),      # 19 - PINKY_DIP
    ((203, 192, 255), "лавандовий"),     # 20 - PINKY_TIP
]

landmark_styles = {
    i: mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=5)
    for i, (color, _) in enumerate(point_colors)
}

# ====================== ЖЕСТИ ======================
EXT_THRESHOLD    = -0.6
CURL_THRESHOLD   =  0.3
DIR_DOMINANCE    = 1.5
STABILITY_FRAMES = 5

# ---------- БАЗОВІ ХЕЛПЕРИ ----------

def _angle_cos(a, b, c):
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return dot / mag

def _finger_state(lm, mcp, pip, tip):
    c = _angle_cos(lm[mcp], lm[pip], lm[tip])
    if c < EXT_THRESHOLD:  return 'ext'
    if c > CURL_THRESHOLD: return 'curl'
    return 'amb'

def _compute_states(lm):
    return {
        'index':  _finger_state(lm, 5, 6, 8),
        'middle': _finger_state(lm, 9, 10, 12),
        'ring':   _finger_state(lm, 13, 14, 16),
        'pinky':  _finger_state(lm, 17, 18, 20),
    }

def _only_index_extended(states):
    return (states['index']  == 'ext'  and
            states['middle'] == 'curl' and
            states['ring']   == 'curl' and
            states['pinky']  == 'curl')

def _index_direction(lm):
    dx = lm[8].x - lm[5].x
    dy = lm[8].y - lm[5].y
    adx, ady = abs(dx), abs(dy)
    if ady > adx * DIR_DOMINANCE:
        return 'up' if dy < 0 else 'down'
    if adx > ady * DIR_DOMINANCE:
        return 'right' if dx > 0 else 'left'
    return None

# ---------- ДЕТЕКТОРИ ЖЕСТІВ ----------

def gesture_open_palm(lm, states):
    if all(s == 'ext' for s in states.values()):
        return "Відкрита долоня"
    return None

def gesture_fist(lm, states):
    if all(s == 'curl' for s in states.values()):
        return "Кулак"
    return None

def gesture_pointing_up(lm, states):
    if _only_index_extended(states) and _index_direction(lm) == 'up':
        return "Палець вгору"
    return None

def gesture_pointing_down(lm, states):
    if _only_index_extended(states) and _index_direction(lm) == 'down':
        return "Палець вниз"
    return None

def gesture_pointing_left(lm, states):
    if _only_index_extended(states) and _index_direction(lm) == 'left':
        return "Палець вліво"
    return None

def gesture_pointing_right(lm, states):
    if _only_index_extended(states) and _index_direction(lm) == 'right':
        return "Палець вправо"
    return None

GESTURE_DETECTORS = [
    gesture_open_palm,
    gesture_fist,
    gesture_pointing_up,
    gesture_pointing_down,
    gesture_pointing_left,
    gesture_pointing_right,
]

def detect_gesture(lm):
    states = _compute_states(lm)
    for detector in GESTURE_DETECTORS:
        result = detector(lm, states)
        if result is not None:
            return result
    return None

# ---------- ХЕНДЛЕРИ ЖЕСТІВ ----------
# Викликаються коли відповідний жест ВПЕРШЕ зареєстровано стабільно.
# Зараз - просто print. Сюди можна додавати будь-яку логіку (керування, події, тощо).

def on_open_palm(hand_idx):
    print(f"[Рука {hand_idx + 1}] Відкрита долоня")

def on_fist(hand_idx):
    print(f"[Рука {hand_idx + 1}] Кулак")

def on_pointing_up(hand_idx):
    print(f"[Рука {hand_idx + 1}] Палець вгору")

def on_pointing_down(hand_idx):
    print(f"[Рука {hand_idx + 1}] Палець вниз")

def on_pointing_left(hand_idx):
    print(f"[Рука {hand_idx + 1}] Палець вліво")

def on_pointing_right(hand_idx):
    print(f"[Рука {hand_idx + 1}] Палець вправо")

# Мапа назва_жесту -> хендлер
GESTURE_HANDLERS = {
    "Відкрита долоня": on_open_palm,
    "Кулак":           on_fist,
    "Палець вгору":    on_pointing_up,
    "Палець вниз":     on_pointing_down,
    "Палець вліво":    on_pointing_left,
    "Палець вправо":   on_pointing_right,
}
# ===================================================

camera = cv2.VideoCapture(0)

hand_coords = {}
gesture_buffers = {}
last_stable = {}

while True:
    success, frame = camera.read()

    # Дзеркальне відображення кадру
    frame = cv2.flip(frame, 1)

    # Отримуємо розміри кадру (висоту, ширину та кількість каналів)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_coords.clear()

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Зберігаємо координати всіх 21 точок руки (без виводу)
            coords = []
            for id, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                coords.append((cx, cy, landmark.z))
            hand_coords[hand_idx] = coords

            # Сирий жест поточного кадру
            raw = detect_gesture(hand_landmarks.landmark)

            # Кільцевий буфер для згладжування
            if hand_idx not in gesture_buffers:
                gesture_buffers[hand_idx] = deque(maxlen=STABILITY_FRAMES)
            gesture_buffers[hand_idx].append(raw)
            buf = gesture_buffers[hand_idx]

            # Стабільний жест - тільки коли всі N кадрів однакові (і не None)
            stable = None
            if len(buf) == STABILITY_FRAMES and buf[0] is not None and len(set(buf)) == 1:
                stable = buf[0]

            # При зміні стабільного жесту - викликаємо відповідний хендлер
            prev = last_stable.get(hand_idx)
            if stable != prev:
                if stable is not None:
                    handler = GESTURE_HANDLERS.get(stable)
                    if handler is not None:
                        handler(hand_idx)
                last_stable[hand_idx] = stable

            # Малюємо точки (кожна своїм кольором) та лінії
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_styles
            )
    else:
        gesture_buffers.clear()
        last_stable.clear()

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()