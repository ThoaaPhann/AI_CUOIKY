import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from tensorflow import expand_dims
import numpy as np
from keras.models import load_model

# Connect with camera
cap = cv2.VideoCapture(0)

# Load pre-trained model
model = load_model('model_seaanimal.h5')

# Create array to save label
label = np.array(['CLOWNFISH', 'CRAB', 'JELLYFISH', 'OTTER', 'PELICAN',
                  'PENGUIN', 'SEA_TURTLE', 'SHRIMP', 'STARFISH', 'WHALE'])

# Create an array to store component information
detail = np.array([
    'CLOWNFISH are found in warm waters, such as the Red Sea and Pacific Oceans, in sheltered reefs or lagoons, living in anemone. Clownfish eat various small invertebrates and algae, as well as food scraps the anemone leaves behind.',
    'CRABS can walk in all directions, but mostly walk and run sideways. Crabs are decapods, meaning they have 10 legs. Female crabs can release 1000 to 2000 eggs at once. The lifespan of a small crab averages around 3-4 years, but larger species such as the giant Japanese spider crab can live as long as 100 years',
    'JELLYFISH are not actually fish—they are invertebrates, or animals with no backbones. Jellyfish have tiny stinging cells in their tentacles to stun or paralyze their prey before they eat them. Inside their bell-shaped body is an opening that is its mouth.',
    'The charismatic otter, a member of the weasel family, is found on every continent except Australia and Antarctica. Most are small, with short ears and noses, elongated bodies, long tails, and soft, dense fur. There are 13 species in total, ranging from the small-clawed otter to the giant otter.',
    'PELICANS inhabit lakes, rivers, and seacoasts in many parts of the world. With some species reaching a length of 180 cm (70 inches), having a wingspan of 3 metres (10 feet), and weighing up to 13 kg (30 pounds), they are among the largest of living birds.',
    'PENGUINS are flightless seabirds that live almost exclusively below the equator. Some island-dwellers can be found in warmer climates, but most—including emperor, adélie, chinstrap, and gentoo penguins—reside in and around icy Antarctica.',
    'SEA TURTLE do not have teeth, but their jaws have modified “beaks” suited to their particular diet. They do not have visible ears but have eardrums covered by skin. They hear best at low frequencies, and their sense of smell is excellent.',
    'SHRIMP are characterized by a semitransparent body flattened from side to side and a flexible abdomen terminating in a fanlike tail. The appendages are modified for swimming, and the antennae are long and whiplike. Shrimp occur in all oceans—in shallow and deep water—and in freshwater lakes and streams.',
    'STARFISH are marine invertebrates. They typically have a central disc and usually five arms, though some species have a larger number of arms. The aboral or upper surface may be smooth, granular or spiny, and is covered with overlapping plates.',
    'WHALES are warm-blooded creatures that nurse their young. There are two types of whales: toothed and baleen. Toothed whales, as the name suggests, have teeth, which are used to hunt and eat squid, fish, and seals.'
])


class MyWindow:
    def __init__(self, master):
        self.master = master
        master.title("IDENTIFICATION SEA ANIMALS")

        # GUI components
        self.label = tk.Label(master)
        self.text_detail = tk.Text(master, height=5, width=65)
        self.text_detail.insert(tk.END, "SEA ANIMALS INFORMATION HERE:")
        self.text_detail.config(state="disabled")
        self.text_detail.config(bd=0, highlightbackground="white", bg="white")
        self.button_load = tk.Button(master, text="UPLOAD IMAGE", bg="red", fg="black", command=self.upload_image)
        self.button_classify  = tk.Button(master, text="CLASSIFY IMAGE", bg="green", fg="white", command=self.classify_image)
        self.button_start = tk.Button(master, text="START CAMERA", bg="blue", fg="white", command=self.start_camera)
        self.button_stop = tk.Button(master, text="STOP CAMERA", bg="cyan", command=self.stop_camera)
        self.camera_running = False

        # Application info
        self.text_info = tk.Text(master, height=2, width=91)
        self.text_info.insert(tk.END, "IDENTIFICATION SEA ANIMALS ON THE CNN METHOD")
        self.text_info.tag_configure("bold", font=("Arial", 20, "bold"))
        self.text_info.tag_add("bold", "1.0", "end")
        self.text_info.config(state="disabled", fg="black")
        self.text_info.pack()

        # Layout
        self.label.place(x=100, y=100)
        self.text_detail.place(x=100, y=450)
        self.text_info.place(x=30, y=30)
        self.button_load.place(x=550, y=100, width=110, height=50)
        self.button_classify.place(x=550, y=160, width=110, height=50)
        self.button_start.place(x=550, y=220, width=110, height=50)
        self.button_stop.place(x=550, y=280, width=110, height=50)

        # Display initial image
        image_init = Image.open("12345.jpg").resize((300, 300))
        photo = ImageTk.PhotoImage(image_init)
        self.label.config(image=photo)
        self.label.image = photo

        # Exit button and event handling
        root.protocol("WM_DELETE_WINDOW", root.quit)
        self.button_exit = tk.Button(root, text="EXIT", command=root.destroy)
        self.button_exit.place(x=550, y=350, width=110, height=50)

    def upload_image(self):
        # Open a file dialog to choose an image
        file_name = tkinter.filedialog.askopenfilename(filetypes=[('Image Files', ('*.jpg', '*.jpeg', '*.png', '*.bmp'))])

        if file_name:
            # Store the selected image file name
            self.selected_image = file_name

            # Open the image
            image_original = Image.open(self.selected_image)

            # Resize and display the image
            image_resized = image_original.resize((300, 300))
            photo = ImageTk.PhotoImage(image_resized)
            self.label.config(image=photo)
            self.label.image = photo

    def classify_image(self):
        if self.selected_image:
            # Open the selected image
            image_original = Image.open(self.selected_image)

            # Preprocess the image
            image = np.array(image_original)
            image = cv2.resize(image, (100, 100))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Perform prediction
            prediction = model.predict(image)
            max_index = np.argmax(prediction, axis=1)
            max_label = label[max_index]
            max_detail = detail[max_index]

            # Display component details
            self.text_detail.config(state="normal")
            self.text_detail.delete("1.0", tk.END)
            self.text_detail.insert(tk.END, str(max_detail))
            self.text_detail.config(state="disabled")

            # Resize the image for display
            image_resized = image_original.resize((300, 300))

            # Use PIL to draw on the image
            draw = ImageDraw.Draw(image_resized)

            # Set font and size
            font = ImageFont.truetype("arial.ttf", 25)
            bold_font = ImageFont.truetype("arialbd.ttf", 25)

            # Write text with the set font and size
            draw.text((0, 0), str(max_label), fill=(255, 0, 0), font=bold_font)

            # Display the image on the label
            photo = ImageTk.PhotoImage(image_resized)
            self.label.config(image=photo)
            self.label.image = photo

    def start_camera(self):
        if not self.camera_running:
            # Open the camera
            self.cap = cv2.VideoCapture(0)
            self.camera_running = True
            self.update_frame()

    def stop_camera(self):
        if self.camera_running:
            # Stop the camera
            self.cap.release()
            self.camera_running = False

    def update_frame(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            # Process the image to numpy array
            image = cv2.resize(frame, (100, 100))
            image = image / 255.0
            image = expand_dims(image, axis=0)

            # Perform prediction
            prediction = model.predict(image)
            max_index = np.argmax(prediction, axis=1)
            max_label = label[max_index]
            max_detail = detail[max_index]

            # Display the prediction directly on the Camera frame
            cv2.putText(frame, str(max_label), (50, 50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display component details
            self.text_detail.config(state="normal")
            self.text_detail.delete("1.0", tk.END)
            self.text_detail.insert(tk.END, str(max_detail))
            self.text_detail.config(state="disabled")

            if ret:
                # Display the image on the label
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = image.resize((300, 300), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                self.label.config(image=photo)
                self.label.image = photo
                self.master.after(5, self.update_frame)


if __name__ == '__main__':
    root = tk.Tk()

    # Set window size
    root.geometry("800x600")

    # Open and convert the image to Tkinter format
    image = Image.open("background.jpg")
    # Resize the image to match the window size
    image = image.resize((800, 600))
    photo = ImageTk.PhotoImage(image)

    # Create a Canvas widget and draw the image as the background of the canvas
    canvas = tk.Canvas(root, width=800, height=500)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    window = MyWindow(root)

    root.mainloop()
