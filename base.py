import cv2
import math
import mediapipe as mp
from collections import deque
import asyncio
import threading
import time
from bleak import BleakClient, BleakScanner

RX_CHARACTERISTIC_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

ble_loop = None
command_queue = asyncio.Queue()
client_instance = None

async def ble_worker():
    """Фоновий асинхронний потік для сканування, підключення та відправки даних"""
    global client_instance
    print("Шукаю робота у Bluetooth-оточенні...")
    
    target_device = None
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name and ("esp32" in d.name.lower() or "kulya" in d.name.lower() or "robot" in d.name.lower()):
            target_device = d
            break
            
    if not target_device and devices:
        for d in devices:
            if d.name:
                target_device = d
                break

    if not target_device:
        print("Робота не знайдено. Переконайтеся, що на ESP32 подано живлення і блимає синій світлодіод.")
        return

    print(f"Знайдено пристрій: {target_device.name} [{target_device.address}]. Підключаюся...")
    
    try:
        async with BleakClient(target_device.address) as client:
            client_instance = client
            print("З'єднання встановлено! Пульт ДУ активний. Покажіть жест у камеру.")
            
            while True:
                cmd = await command_queue.get()
                if cmd == "STOP_WORKER":
                    command_queue.task_done()
                    break
                
                if client.is_connected:
                    try:
                        await client.write_gatt_char(RX_CHARACTERISTIC_UUID, (cmd + '\n').encode('utf-8'))
                    except Exception as e:
                        print(f"Помилка відправки пакета [{cmd}]: {e}")
                
                command_queue.task_done()
    except Exception as e:
        print(f"Помилка BLE сесії: {e}")

def start_ble_thread():
    global ble_loop
    ble_loop = asyncio.new_event_loop()
    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ble_worker())
    t = threading.Thread(target=run_loop, args=(ble_loop,), daemon=True)
    t.start()

def send_cmd(cmd_text):
    """Функція для безпечного додавання команд у чергу відправки з потоку OpenCV"""
    if ble_loop and ble_loop.is_running():
        asyncio.run_coroutine_threadsafe(command_queue.put(cmd_text), ble_loop)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1
)

mp_draw = mp.solutions.drawing_utils
landmark_styles = {i: mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4) for i in range(21)}

EXT_THRESHOLD    = -0.6
CURL_THRESHOLD   =  0.3
STABILITY_FRAMES = 5

current_index_angle = None
last_send_time = 0
SEND_INTERVAL = 0.06  

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
    return (states['index']  == 'ext' and
            states['middle'] == 'curl' and
            states['ring']   == 'curl' and
            states['pinky']  == 'curl')

def _index_angle_deg(lm):
    global current_index_angle
    dx = lm[8].x - lm[5].x
    dy = -(lm[8].y - lm[5].y) 
    angle = math.degrees(math.atan2(dy, dx))
    current_index_angle = angle
    return angle

def gesture_open_palm(lm, states):
    if all(s == 'ext' for s in states.values()): return "Відкрита долоня"
    return None

def gesture_fist(lm, states):
    if all(s == 'curl' for s in states.values()): return "Кулак"
    return None

def gesture_driving(lm, states):
    if _only_index_extended(states): return "Хода за пальцем"
    return None

GESTURE_DETECTORS = [gesture_open_palm, gesture_fist, gesture_driving]

def detect_gesture(lm):
    states = _compute_states(lm)
    for detector in GESTURE_DETECTORS:
        result = detector(lm, states)
        if result is not None: return result
    return None

def on_open_palm(hand_idx):
    print("Подія: ДОЛОНЯ -> Розкрити робота (expand)")
    send_cmd("ce_value:100")
    send_cmd("gait:WALK")

def on_fist(hand_idx):
    print("Подія: КУЛАК -> Закрити/Скласти робота (collapse)")
    send_cmd("gait:CE")
    send_cmd("ce_value:5")

def handle_driving(angle):
    """Вираховує тригонометричний вектор нахилу пальця і транслює в координати J_XY"""
    global last_send_time
    now = time.time()
    if now - last_send_time < SEND_INTERVAL:
        return 
        
    rad = math.radians(angle)
    j_x = int(math.cos(rad) * 100)
    j_y = int(math.sin(rad) * 100)
    
    if abs(j_x) < 15: j_x = 0
    if abs(j_y) < 15: j_y = 0
        
    print(f"Подія: ПАЛЕЦЬ (Кут {int(angle)}°) -> Крок вектора J_XY:{j_x}|{j_y}")
    send_cmd(f"J_XY:{j_x}|{j_y}")
    last_send_time = now

GESTURE_HANDLERS = {
    "Відкрита долоня": on_open_palm,
    "Кулак":           on_fist,
}

start_ble_thread()

camera = cv2.VideoCapture(0)
gesture_buffers = {}
last_stable = {}

while True:
    success, frame = camera.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_index_angle = None 

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            raw = detect_gesture(hand_landmarks.landmark)

            if hand_idx not in gesture_buffers:
                gesture_buffers[hand_idx] = deque(maxlen=STABILITY_FRAMES)
            gesture_buffers[hand_idx].append(raw)
            buf = gesture_buffers[hand_idx]

            stable = None
            if len(buf) == STABILITY_FRAMES and buf[0] is not None and len(set(buf)) == 1:
                stable = buf[0]

            prev = last_stable.get(hand_idx)
            if stable != prev:
                if stable in GESTURE_HANDLERS:
                    GESTURE_HANDLERS[stable](hand_idx)
                last_stable[hand_idx] = stable

            if raw == "Хода за пальцем":
                angle = _index_angle_deg(hand_landmarks.landmark)
                handle_driving(angle)
            elif prev == "Хода за пальцем" and raw != "Хода за пальцем":
                print("Палець прибрано -> Зупинка руху")
                send_cmd("J_XY:0|0")
                last_stable[hand_idx] = None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        if any(v == "Хода за пальцем" for v in last_stable.values()):
            send_cmd("J_XY:0|0")
        gesture_buffers.clear()
        last_stable.clear()

    if current_index_angle is not None:
        cv2.putText(frame, f"Angle: {int(current_index_angle)} deg", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("KULYA BLE Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
send_cmd("STOP_WORKER")
cv2.destroyAllWindows()
