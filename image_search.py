import os
import json
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# Initialize the Tkinter window
root = Tk()
root.title("Image Captioning and Search System")
root.geometry("900x700")
root.configure(bg="#F0F0F0")

# Load the pre-trained BLIP captioning model and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load Sentence-BERT model for text similarity and use GPU if available
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# Path to store captions
captions_file = 'captions.json'

# Global variables
gallery_path = ''
search_results = []
current_result_index = 0

# Load or initialize the captions dictionary
def load_captions():
    if os.path.exists(captions_file):
        with open(captions_file, 'r') as f:
            return json.load(f)
    return {}

# Save captions to the JSON file
def save_captions(captions_dict):
    with open(captions_file, 'w') as f:
        json.dump(captions_dict, f)

# Generate captions in batches for faster processing
def generate_captions_for_images_batch(images_to_process):
    try:
        images = [Image.open(file_path).convert('RGB') for _, file_path in images_to_process]
        inputs = caption_processor(images=images, return_tensors="pt").to(device)
        caption_ids = caption_model.generate(**inputs)
        captions = [caption_processor.decode(caption_id, skip_special_tokens=True) for caption_id in caption_ids]

        # Return the full paths with their generated captions
        return [(images_to_process[i][1], captions[i]) for i in range(len(images_to_process))]
    except Exception as e:
        print(f"Error processing images: {e}")
        return []

# Generate captions for images in the gallery
def generate_captions_for_gallery_threaded(gallery_path):
    # Load existing captions
    captions_dict = load_captions()

    # Prepare the list of images to process
    images_to_process = []
    for file_name in os.listdir(gallery_path):
        file_path = os.path.join(gallery_path, file_name)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')) and file_path not in captions_dict:
            images_to_process.append((file_name, file_path))

    # Process the images in batches to speed up the process
    batch_size = 8  # Adjust batch size based on your system's memory and GPU capacity
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(images_to_process), batch_size):
            batch = images_to_process[i:i+batch_size]
            futures.append(executor.submit(generate_captions_for_images_batch, batch))

        for future in futures:
            results = future.result()
            for file_path, caption in results:
                if caption:
                    captions_dict[file_path] = caption

    # Save updated captions
    save_captions(captions_dict)
    status_label.config(text="Captions generated and saved!")
    
    # Show a message box indicating completion
    messagebox.showinfo("Info", "Caption generation completed.")

# Wrapper function to run the caption generation in a separate thread
def start_caption_generation():
    if gallery_path:
        # Update the UI to indicate that the process is running
        status_label.config(text="Generating captions for images...")
        root.update_idletasks()

        # Start the thread for caption generation
        thread = threading.Thread(target=generate_captions_for_gallery_threaded, args=(gallery_path,))
        thread.start()
    else:
        status_label.config(text="Please select a gallery folder first.")

# Function to search the most similar captions based on user input
def search_similar_images(user_input, captions_dict, top_k=5):
    # Extract captions and file names (now the full paths)
    file_paths = list(captions_dict.keys())
    captions = list(captions_dict.values())

    # Encode captions and user input into embeddings
    caption_embeddings = similarity_model.encode(captions, convert_to_tensor=True, device=device)
    user_input_embedding = similarity_model.encode(user_input, convert_to_tensor=True, device=device)

    # Compute cosine similarities
    similarities = util.cos_sim(user_input_embedding, caption_embeddings)[0]

    # Get the indices of the top_k most similar captions
    top_indices = similarities.argsort(descending=True)[:top_k]

    # Retrieve the corresponding file paths and captions
    result_images = [(file_paths[idx], captions[idx], similarities[idx].item()) for idx in top_indices]
    return result_images

# Callback function to open folder dialog
def open_gallery():
    global gallery_path
    gallery_path = filedialog.askdirectory()
    if gallery_path:
        start_caption_generation()

# Callback function to search images and display results
def search_images():
    global search_results, current_result_index
    user_input = search_entry.get()
    if user_input:
        # Perform search based on user input
        captions_dict = load_captions()
        try:
            top_k = int(results_entry.get())
        except ValueError:
            top_k = 5  # Default to 5 if invalid input
            results_entry.delete(0, END)
            results_entry.insert(0, "5")

        search_results = search_similar_images(user_input, captions_dict, top_k)
        current_result_index = 0
        display_result()

# Function to display the current result in the GUI
def display_result():
    global current_result_index

    # Clear the previous result
    for widget in results_frame.winfo_children():
        widget.destroy()

    # If no results, display a message
    if not search_results:
        no_results_label = Label(results_frame, text="No matching images found.", font=("Helvetica", 12))
        no_results_label.pack(pady=20)
        return

    # Get the current result
    file_path, caption, similarity = search_results[current_result_index]

    # Open and resize the image
    img = Image.open(file_path).convert('RGB')
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)

    # Display the image
    img_label = Label(results_frame, image=img)
    img_label.image = img  # keep a reference!
    img_label.pack(pady=10)

    # Display the caption
    caption_text = f"Caption: {caption}\nSimilarity: {similarity:.2f}"
    caption_label = Label(results_frame, text=caption_text, wraplength=400, justify=LEFT)
    caption_label.pack(pady=10)

# Functions for navigation
def show_previous():
    global current_result_index
    if current_result_index > 0:
        current_result_index -= 1
        display_result()

def show_next():
    global current_result_index
    if current_result_index < len(search_results) - 1:
        current_result_index += 1
        display_result()

# Function to download the current image
def download_image():
    if search_results:
        file_path, _, _ = search_results[current_result_index]
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=os.path.basename(file_path))
        if save_path:
            try:
                with open(file_path, 'rb') as src_file:
                    data = src_file.read()
                with open(save_path, 'wb') as dest_file:
                    dest_file.write(data)
                status_label.config(text=f"Image saved to {save_path}")
            except Exception as e:
                status_label.config(text=f"Error saving image: {e}")

# GUI Layout
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))
style.configure('TEntry', font=('Helvetica', 10))

# Frame for folder selection and caption generation
folder_frame = ttk.Frame(root)
folder_frame.pack(pady=20)

# Button to open the gallery folder
open_button = ttk.Button(folder_frame, text="Select Gallery Folder", command=open_gallery)
open_button.grid(row=0, column=0, padx=10)

# Label to show status
status_label = ttk.Label(folder_frame, text="", background="#F0F0F0")
status_label.grid(row=0, column=1, padx=10)

# Frame for search input
search_frame = ttk.Frame(root)
search_frame.pack(pady=10)

# Entry box for search query
search_label = ttk.Label(search_frame, text="Search:")
search_label.grid(row=0, column=0, padx=10)
search_entry = ttk.Entry(search_frame, width=40)
search_entry.grid(row=0, column=1, padx=10)

# Entry box for the number of results
results_label = ttk.Label(search_frame, text="Top K results:")
results_label.grid(row=0, column=2, padx=10)
results_entry = ttk.Entry(search_frame, width=5)
results_entry.insert(0, "5")  # Default to 5 results
results_entry.grid(row=0, column=3, padx=10)

# Button to perform the search
search_button = ttk.Button(search_frame, text="Search Images", command=search_images)
search_button.grid(row=0, column=4, padx=10)

# Frame to display the results
results_frame = Frame(root, bg="#FFFFFF")
results_frame.pack(pady=20)

# Navigation buttons
nav_frame = ttk.Frame(root)
nav_frame.pack(pady=10)

prev_button = ttk.Button(nav_frame, text="<< Previous", command=show_previous)
prev_button.grid(row=0, column=0, padx=20)

download_button = ttk.Button(nav_frame, text="Download Image", command=download_image)
download_button.grid(row=0, column=1, padx=20)

next_button = ttk.Button(nav_frame, text="Next >>", command=show_next)
next_button.grid(row=0, column=2, padx=20)

# Start the GUI loop
root.mainloop()
