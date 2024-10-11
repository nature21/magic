import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


def get_click_coordinates_from_image_path(image_path: str) -> tuple[int, int]:
    """
    Opens an image in a UI window and waits for the user to click on the image.
    Returns the coordinates of the click and closes the window.

    :param image_path: Path to the image file.
    :return: A tuple containing the u and v coordinates of the click (axis 1 and axis 0)
    """

    # Define a simple class to hold the application's state.
    class ImageClickApp:
        def __init__(self, master, image_path):
            self.master = master
            self.clicked_coords = None

            # Load and display the image.
            img = Image.open(image_path)
            img_tk = ImageTk.PhotoImage(img)
            self.lbl = tk.Label(master, image=img_tk)
            self.lbl.pack()

            # Bind the click event.
            self.lbl.bind("<Button-1>", self.on_click)

            # Keep a reference to prevent garbage-collection.
            self.lbl.img_tk = img_tk

        def on_click(self, event):
            # Store the coordinates and close the window.
            self.clicked_coords = (event.x, event.y)
            self.master.destroy()

    # Create the Tkinter window.
    root = tk.Tk()
    app = ImageClickApp(root, image_path)

    # Run the event loop and wait for it to finish.
    root.mainloop()

    # Return the coordinates after the window has been closed.
    return app.clicked_coords


def get_click_coordinates_from_array(image_array: np.ndarray, title='') -> tuple[int, int]:
    """
    !!! Warning: use this function carefully, since it uses tk backend,
    which will cause a strange bug used together with open3d!!!

    Opens an image (from a NumPy array) in a UI window and waits for the user to click on the image.
    Returns the coordinates of the click and closes the window.

    :param image_array: A NumPy array representing the image.
    :param title: The title of the window.

    :return: A tuple containing the u and v coordinates of the click (axis 1 and axis 0)
    """

    class ImageClickApp:
        def __init__(self, master, image_array):
            self.master = master
            self.clicked_coords = None

            # Convert the NumPy array to a PIL image and then to a format Tkinter can use.
            img = Image.fromarray(image_array)
            img_tk = ImageTk.PhotoImage(img)
            self.lbl = tk.Label(master, image=img_tk)
            self.lbl.pack()

            self.lbl.bind("<Button-1>", self.on_click)

            self.lbl.img_tk = img_tk

        def on_click(self, event):
            self.clicked_coords = (event.x, event.y)
            self.master.destroy()

    root = tk.Tk()
    root.title(title)
    app = ImageClickApp(root, image_array)

    root.mainloop()

    return app.clicked_coords

# Example usage:
if __name__ == "__main__":
    image_path = "image_scene/hook_with_ball.png"  # Change this to the path of your image.
    coordinates = get_click_coordinates_from_image_path(image_path)
    print(f"Clicked coordinates: {coordinates}")