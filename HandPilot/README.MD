
---

# 🖐️ Hand Gesture Control using OpenCV, MediaPipe & PyAutoGUI

A Python-based project that allows you to control your computer using hand gestures via webcam. This project uses **MediaPipe** to detect hand landmarks, interprets gestures, and simulates key presses using **PyAutoGUI**.

---

## 🎯 Features

* Real-time hand tracking using **MediaPipe**
* Detects 4 gestures:

  * ✊ **Fist (0 fingers)** → Presses **Down**
  * ☝️ **1 Finger (Index)** → Presses **Right**
  * ✌️ **2 Fingers (Index + Middle)** → Presses **Left**
  * 🖐️ **4 Fingers (All except thumb)** → Presses **Up**
* Simulates keyboard key presses using **PyAutoGUI**
* Displays hand landmarks and tracks FPS
* Lightweight and works on most systems with a webcam

---

## 🛠️ Requirements

Install the following Python packages before running the code:

```bash
pip install opencv-python mediapipe pyautogui
```

---

## 🚀 How to Run

1. Make sure you have a webcam connected.
2. Clone this repository or copy the Python file.
3. Run the Python script:

```bash
python main.py
```

4. A window will open showing your webcam feed with hand tracking enabled.
5. Use the following gestures to simulate key presses:

| Gesture                     | Key Press |
| --------------------------- | --------- |
| ✊ Fist (No fingers up)      | `Down`    |
| ☝️ Index finger only        | `Right`   |
| ✌️ Index + Middle fingers   | `Left`    |
| 🖐️ Four fingers (no thumb) | `Up`      |

6. Press `q` to quit the application.

---

## 📷 Demo Preview (Optional)

You can add a screenshot or gif showing the gestures in action.

---

## 📄 Code Overview

* **OpenCV** captures the video stream.
* **MediaPipe** detects and tracks hand landmarks.
* Logic determines which fingers are up.
* Based on finger positions, specific gestures are mapped to key presses.
* **PyAutoGUI** triggers the corresponding keyboard event.

---

## 🧠 Future Improvements

* Support for more gestures (e.g., volume control, scroll).
* Multi-hand support.
* Add GUI to customize gesture mappings.
* Add voice feedback using `pyttsx3` or `gTTS`.

---

## 🧑‍💻 Author

**Pankaj Mishra**
Made with ❤️ for gesture-based automation projects.

---

## 📜 License

This project is licensed under the MIT License.

