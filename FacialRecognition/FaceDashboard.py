import customtkinter as ctk
import subprocess
import os
import sys

# Set appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Face Recognition System")
        self.geometry("500x450")

        # Title Label
        self.label = ctk.CTkLabel(self, text="Face Recognition Dashboard", font=("Roboto", 24))
        self.label.pack(pady=20)

        # Buttons
        self.btn_capture = ctk.CTkButton(self, text="Step 1: Capture Dataset", 
                                          command=self.run_capture, width=250, height=40)
        self.btn_capture.pack(pady=10)

        self.btn_train = ctk.CTkButton(self, text="Step 2: Train Model", 
                                        command=self.run_training, width=250, height=40)
        self.btn_train.pack(pady=10)

        self.btn_recognize = ctk.CTkButton(self, text="Step 3: Start Recognition", 
                                            command=self.run_recognition, width=250, height=40,
                                            fg_color="green", hover_color="#006400")
        self.btn_recognize.pack(pady=10)

        # Status Label
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack(pady=20)
        
        self.note_label = ctk.CTkLabel(self, text="Press 'ESC' to close camera windows", font=("Roboto", 10), text_color="yellow")
        self.note_label.pack(pady=5)

    def run_script(self, script_name, start_msg, end_msg):
        """Helper to run scripts without freezing the UI"""
        try:
            self.status_label.configure(text=start_msg)
            self.update_idletasks() # Refresh UI
            
            # Use Popen instead of run so the UI stays alive
            process = subprocess.Popen([sys.executable, script_name])
            process.wait() # Still waits, but allows the OS to handle the window better
            
            self.status_label.configure(text=end_msg)
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")

    def run_capture(self):
        self.run_script("01_face_dataset.py", "Capturing... Look at camera", "Capture Complete!")

    def run_training(self):
        self.run_script("02_face_training.py", "Training AI... Please wait", "Training Finished!")

    def run_recognition(self):
        # Recognition usually runs until ESC is pressed
        self.run_script("03_face_recognition.py", "Recognition Active", "Ready")

if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()