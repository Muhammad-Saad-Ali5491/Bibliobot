from pypdf import PdfReader

def load_pdf(filepath):
    reader = PdfReader(filepath)
    print(f"Number of pages: {len(reader.pages)}")
    
    full_text = ""
    page_map = []  
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        start = len(full_text)
        full_text += text + "\n"
        end = len(full_text)
        page_map.append((start, end, i + 1))
    
    return full_text, page_map

if __name__ == "__main__":
    text, page_map = load_pdf(r"book\Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.pdf")
    print(f"Total characters: {len(text)}")
    print(text[:1000])