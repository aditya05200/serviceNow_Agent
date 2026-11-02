from bs4 import BeautifulSoup


def read_text_file(uploaded_file):

    extension = uploaded_file.name.split(".")[-1]

    content = uploaded_file.read()

    text = content.decode("utf-8")

    if extension == "html":

        soup = BeautifulSoup(
            text,
            "html.parser"
        )

        text = soup.get_text()

    return text