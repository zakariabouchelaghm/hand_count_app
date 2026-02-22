"""
✋ Hand Count AI
Teaches children to show numbers 1-10 using hand gestures.
Pure numpy inference (no TF/keras/pandas/requests).
"""
import sys, os, random, threading
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import pygame
import h5py

# ──────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────
def resource_path(rel):
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, rel)


# ──────────────────────────────────────────────
# LOAD MODEL WEIGHTS (h5py → pure numpy)
# ──────────────────────────────────────────────
def load_model_weights(path):
    """Extract dense layer weights from Keras .h5 file."""
    layers = []
    with h5py.File(path, 'r') as f:
        mw = f['model_weights']
        for layer_name in ['dense_3', 'dense_4', 'dense_5']:
            g = mw[layer_name]['sequential_1'][layer_name]
            W = g['kernel'][:]   # (in, out)
            b = g['bias'][:]     # (out,)
            layers.append((W, b))
    return layers   # [(W3, b3), (W4, b4), (W5, b5)]


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(layers, row):
    """Forward pass: input(126) → dense64(relu) → dense32(relu) → dense10(softmax)"""
    x = np.asarray(row, dtype=np.float32)
    W, b = layers[0]; x = relu(x @ W + b)
    W, b = layers[1]; x = relu(x @ W + b)
    W, b = layers[2]; x = softmax(x @ W + b)
    return x  # shape (10,)


# ──────────────────────────────────────────────
# AUDIO
# ──────────────────────────────────────────────
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.mixer.init()

def _load_sound(path):
    try:
        return pygame.mixer.Sound(resource_path(path))
    except Exception:
        return None

correct_sound = _load_sound("sounds/correct.wav")
number_sounds = [_load_sound(f"sounds/{i}.wav") for i in range(1, 11)]


# ──────────────────────────────────────────────
# MEDIAPIPE
# ──────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path=resource_path('hand_landmarker.task'))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
hands_detector = vision.HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────
# APP CONSTANTS
# ──────────────────────────────────────────────
WIN_W, WIN_H   = 960, 680
CAM_W, CAM_H   = 860, 500
TOTAL_ROUNDS   = 7
CONFIDENCE_THR = 0.75
INFER_EVERY    = 4          # run inference every N frames (speed)

NUMBER_WORDS = {
    1:"واحد", 2:"اثنان", 3:"ثلاثة", 4:"أربعة", 5:"خمسة",
    6:"ستة", 7:"سبعة", 8:"ثمانية", 9:"تسعة", 10:"عشرة"
}

NUMBER_EMOJI = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣",
                "6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

COLORS = {
    "bg":       "#0D1117",
    "card":     "#161B22",
    "accent":   "#7C3AED",
    "accent2":  "#A78BFA",
    "green":    "#22C55E",
    "red":      "#EF4444",
    "yellow":   "#F59E0B",
    "text":     "#F1F5F9",
    "subtext":  "#94A3B8",
    "border":   "#30363D",
}


# ──────────────────────────────────────────────
# MAIN APP CLASS
# ──────────────────────────────────────────────
class HandCountApp:
    def __init__(self):
        self.model_layers = load_model_weights(resource_path("hand1_5_v3.h5"))
        self.cap          = None
        self.target       = None
        self.round_count  = 0
        self.frame_count  = 0
        self.label        = None
        self.running      = False
        self._after_id    = None

        self._load_app_logo()

        self._build_window()
        self._show_start_screen()

    def _load_app_logo(self):
        try:
            img = Image.open(resource_path("images/favicon.png"))
            self.logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(32, 32))
            
            # Large version for start screen
            large_img = img.copy()
            large_img.thumbnail((320, 320), Image.Resampling.LANCZOS)
            self.large_logo_img = ctk.CTkImage(light_image=large_img, dark_image=large_img, size=large_img.size)
        except Exception:
            self.logo_img = None
            self.large_logo_img = None

    # ── WINDOW ──────────────────────────────────
    def _build_window(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("✋ Hand Count AI")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(False, False)
        self.root.configure(fg_color=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            icon_img = ImageTk.PhotoImage(Image.open(resource_path("images/favicon.png")))
            self.root.iconphoto(True, icon_img)
        except Exception as e:
            print(f"Failed to load favicon: {e}")


        # Master container
        self.master = ctk.CTkFrame(self.root, fg_color=COLORS["bg"],
                                   corner_radius=0)
        self.master.pack(fill="both", expand=True)

    # ── START SCREEN ────────────────────────────
    def _show_start_screen(self):
        self._clear_master()

        # ── Hero card ──
        # Wider hero card to fit both text and slideshow
        card = ctk.CTkFrame(self.master, fg_color=COLORS["card"],
                            corner_radius=24, border_width=1,
                            border_color=COLORS["border"])
        card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.82)

        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        left_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        left_frame.pack(side="left", padx=20)

        # Big hand emoji
        ctk.CTkLabel(left_frame, text="✋🤚", font=("Segoe UI Emoji", 72)).pack(pady=(16, 8))

        ctk.CTkLabel(left_frame, text="Hand Count AI",
                     font=("Segoe UI", 36, "bold"),
                     text_color=COLORS["text"]).pack()

        ctk.CTkLabel(left_frame,
                     text="!أظهر الأرقام من 1 إلى 10 بيديك\n🎉 .سوف تسألك الكاميرا وأنت تجيب",
                     font=("Segoe UI", 16),
                     text_color=COLORS["subtext"],
                     justify="right").pack(pady=(12, 30))

        # Stats row
        stats_row = ctk.CTkFrame(left_frame, fg_color="transparent")
        stats_row.pack(pady=(0, 28))
        for icon, label in [("🎯", f"{TOTAL_ROUNDS} جولات"),
                             ("🎵", "أصوات ممتعة")]:
            b = ctk.CTkFrame(stats_row,
                             fg_color=COLORS["bg"],
                             corner_radius=12,
                             border_width=1,
                             border_color=COLORS["border"])
            b.pack(side="left", padx=10, pady=4, ipadx=14, ipady=8)
            ctk.CTkLabel(b, text=icon, font=("Segoe UI Emoji", 20)).pack()
            ctk.CTkLabel(b, text=label, font=("Segoe UI", 12),
                         text_color=COLORS["subtext"]).pack()

        # START BUTTON
        start_btn = ctk.CTkButton(
            left_frame,
            text="▶  ابدأ التمرين",
            font=("Segoe UI", 20, "bold"),
            height=56,
            width=280,
            corner_radius=16,
            fg_color=COLORS["accent"],
            hover_color="#6D28D9",
            text_color="white",
            command=self._start_game
        )
        start_btn.pack(pady=(0, 8))

        ctk.CTkLabel(left_frame,
                     text="📷 تأكد من أن كاميرتك جاهزة",
                     font=("Segoe UI", 12),
                     text_color=COLORS["subtext"]).pack(pady=(2, 16))

        # Logo frame
        if hasattr(self, 'large_logo_img') and self.large_logo_img:
            right_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
            right_frame.pack(side="left", padx=30)
            
            self.logo_label = ctk.CTkLabel(right_frame, text="", 
                                              image=self.large_logo_img,
                                              width=340, height=340,
                                              corner_radius=16,
                                              fg_color=COLORS["bg"])
            self.logo_label.pack()

    # ── GAME SCREEN ─────────────────────────────
    def _show_game_screen(self):
        self._clear_master()

        # ── Top bar ──
        top = ctk.CTkFrame(self.master, fg_color=COLORS["card"],
                           corner_radius=0, height=70)
        top.pack(fill="x")
        top.pack_propagate(False)

        title_frame = ctk.CTkFrame(top, fg_color="transparent")
        title_frame.pack(side="left", padx=22, pady=16)

        if hasattr(self, 'logo_img') and self.logo_img:
            ctk.CTkLabel(title_frame, text="", image=self.logo_img).pack(side="left", padx=(0, 10))
            ctk.CTkLabel(title_frame, text="Hand Count AI",
                         font=("Segoe UI", 18, "bold"),
                         text_color=COLORS["text"]).pack(side="left")
        else:
            ctk.CTkLabel(title_frame, text="✋ Hand Count AI",
                         font=("Segoe UI", 18, "bold"),
                         text_color=COLORS["text"]).pack(side="left")


        # Round
        self.round_lbl = ctk.CTkLabel(top,
                                       text=f"الجولة {self.round_count}/{TOTAL_ROUNDS}",
                                       font=("Segoe UI", 15),
                                       text_color=COLORS["accent2"])
        self.round_lbl.pack(side="right", padx=(10, 22))

        # EXIT BUTTON
        exit_btn = ctk.CTkButton(
            top,
            text="✖",
            font=("Segoe UI", 16, "bold"),
            width=40,
            height=40,
            corner_radius=10,
            fg_color="transparent",
            hover_color=COLORS["red"],
            text_color=COLORS["subtext"],
            command=self._exit_game
        )
        exit_btn.pack(side="right", padx=10)

        # ── Number prompt card ──
        prompt_card = ctk.CTkFrame(self.master, fg_color=COLORS["card"],
                                   corner_radius=0, height=88)
        prompt_card.pack(fill="x")
        prompt_card.pack_propagate(False)

        inner = ctk.CTkFrame(prompt_card, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # To keep the number box completely centered, we will place it exactly at center,
        # and place the text to its right (visually pulling it using place layout)
        self.number_badge = ctk.CTkLabel(inner, text="?",
                                          font=("Segoe UI", 42, "bold"),
                                          text_color="white",
                                          fg_color=COLORS["accent"],
                                          corner_radius=14,
                                          width=80, height=60)
        self.number_badge.pack(side="left")

        ctk.CTkLabel(inner, text=":أظهر هذا الرقم",
                     font=("Segoe UI", 15),
                     text_color=COLORS["subtext"],
                     width=120, anchor="w").pack(side="left", padx=(12,0))

        # Add a dummy label on the left of the badge to balance it perfectly
        ctk.CTkLabel(inner, text="", width=120+12).pack(side="left", before=self.number_badge)



        # ── Camera feed ──
        cam_frame = ctk.CTkFrame(self.master, fg_color=COLORS["bg"],
                                  corner_radius=0)
        cam_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.cam_label = tk.Label(cam_frame, bg=COLORS["bg"])
        self.cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # ── Status bar ──
        self.status_bar = ctk.CTkLabel(self.master, text="",
                                        font=("Segoe UI", 16, "bold"),
                                        text_color=COLORS["green"],
                                        height=36,
                                        fg_color=COLORS["card"])
        self.status_bar.pack(fill="x", side="bottom")

        self.debug_lbl = ctk.CTkLabel(self.master, text="جاري الكشف...",
                                       font=("Arial", 12), text_color=COLORS["subtext"])
        self.debug_lbl.pack(fill="x", side="bottom")

    # ── END SCREEN ──────────────────────────────
    def _show_end_screen(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self._clear_master()

        card = ctk.CTkFrame(self.master, fg_color=COLORS["card"],
                            corner_radius=24, border_width=1,
                            border_color=COLORS["border"])
        card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.65, relheight=0.72)

        emoji = "🏆"

        ctk.CTkLabel(card, text=emoji, font=("Segoe UI Emoji", 72)).pack(pady=(36, 8))
        ctk.CTkLabel(card, text="!أحسنت",
                     font=("Segoe UI", 32, "bold"),
                     text_color=COLORS["text"]).pack()

        ctk.CTkLabel(card, text="!لقد أنهيت التمارين بنجاح",
                     font=("Segoe UI", 16),
                     text_color=COLORS["subtext"]).pack(pady=(12, 30))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack()

        ctk.CTkButton(row,
                      text="🔄  العب مرة أخرى",
                      font=("Segoe UI", 17, "bold"),
                      height=50, width=200,
                      corner_radius=14,
                      fg_color=COLORS["accent"],
                      hover_color="#6D28D9",
                      command=self._restart).pack(side="left", padx=8)

        ctk.CTkButton(row,
                      text="✖  خروج",
                      font=("Segoe UI", 17),
                      height=50, width=140,
                      corner_radius=14,
                      fg_color=COLORS["card"],
                      hover_color=COLORS["border"],
                      border_color=COLORS["border"],
                      border_width=1,
                      text_color=COLORS["subtext"],
                      command=self.root.destroy).pack(side="left", padx=8)

    # ── GAME LOGIC ──────────────────────────────
    def _start_game(self):
        self.round_count = 0
        self.frame_count = 0
        self._show_game_screen()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.root.after(200, self._new_round)

    def _new_round(self):
        if self.round_count >= TOTAL_ROUNDS:
            self._show_end_screen()
            return

        self.round_count += 1
        self.frame_count  = 0
        self.label        = None
        self.target       = random.randint(1, 10)

        self.number_badge.configure(text=str(self.target), fg_color=COLORS["accent"])
        self.round_lbl.configure(
            text=f"الجولة {self.round_count}/{TOTAL_ROUNDS}")
        self.status_bar.configure(text="")

        s = number_sounds[self.target - 1]
        if s: s.play()

        self._detect_loop()

    def _detect_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._after_id = self.root.after(30, self._detect_loop)
            return

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run mediapipe
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # VIDEO mode requires a STRICTLY monotonically increasing timestamp in ms
        # We'll use a frame-based timestamp to guarantee increasing values
        # (30ms per frame assuming ~30fps)
        if not hasattr(self, '_mp_timestamp'):
            self._mp_timestamp = 0
        self._mp_timestamp += 33 # roughly 30fps inc
        
        try:
            results = hands_detector.detect_for_video(mp_img, self._mp_timestamp)
        except Exception as e:
            self.debug_lbl.configure(text=f"Detection Error: {e}")
            self._after_id = self.root.after(30, self._detect_loop)
            return

        predicted = None
        hand_status = "لا توجد أيدي"

        if results.hand_landmarks:
            hand_status = f"تم اكتشاف الأيدي: {len(results.hand_landmarks)}"
            left  = [0.0] * 63
            right = [0.0] * 63
            detected_sides = []

            for i, lm_list in enumerate(results.hand_landmarks):
                # Mediapipe 'Left' usually means physical right in selfie mode
                # The user reports left hand is detected as right. 
                # We swap them to match physical intent with training data slots.
                side = results.handedness[i][0].category_name
                
                flat = []
                for p in lm_list:
                    flat += [p.x, p.y, p.z]
                
                # SWAPPED mapping for selfie-view correction:
                if side == 'Left':
                    right = flat
                    detected_sides.append("R(phys)")
                else:
                    left = flat
                    detected_sides.append("L(phys)")

            # Inference every N frames
            self.frame_count += 1
            if self.frame_count % INFER_EVERY == 0:
                row  = left + right          # 126 floats
                probs = predict(self.model_layers, row)
                conf  = float(probs.max())
                predicted_idx = int(np.argmax(probs)) + 1
                hand_status = f"الجوانب: {', '.join(detected_sides)} | التوقع: {predicted_idx} ({conf:.2f})"
                
                if conf > CONFIDENCE_THR:
                    predicted = predicted_idx # 1-indexed

        self.debug_lbl.configure(text=hand_status)

        # Render camera feed
        img    = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.BILINEAR)
        imgtk  = ImageTk.PhotoImage(img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

        # Check answer
        if predicted is not None and predicted == self.target:
            self.status_bar.configure(text="🎉 !أحسنت !صحيح  ✅",
                                      text_color=COLORS["green"])
            self.number_badge.configure(fg_color=COLORS["green"])
            if correct_sound: correct_sound.play()
            self._after_id = self.root.after(2000, self._new_round)
        else:
            self._after_id = self.root.after(30, self._detect_loop)

    def _restart(self):
        self._start_game()

    def _exit_game(self):
        """Stop camera and return to start screen."""
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        pygame.mixer.stop()
        self._show_start_screen()

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        pygame.quit()
        self.root.destroy()

    # ── HELPERS ─────────────────────────────────
    def _clear_master(self):
        for w in self.master.winfo_children():
            w.destroy()

    def run(self):
        self.root.mainloop()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = HandCountApp()
    app.run()
