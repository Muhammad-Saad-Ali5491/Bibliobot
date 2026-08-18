from pypdf import PdfReader

def load_pdf(filepath):
    reader = PdfReader(filepath)
    print(f"Number of pages: {len(reader.pages)}")
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text += text + "\n"
    return full_text

if __name__ == "__main__":
    text = load_pdf(r"book\Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.pdf")
    #smaple test
    print(f"Total characters: {len(text)}")
    print(text[:1000])   # checking the data