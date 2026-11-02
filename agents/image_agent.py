from PIL import Image


def read_image(uploaded_file):

    return Image.open(uploaded_file)