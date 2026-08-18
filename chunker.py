def find_page(offset, page_map):
    for start, end, page_num in page_map:
        if start <= offset < end:
            return page_num
    return page_map[-1][2] if page_map else None

def chunk_text(text, page_map, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end]
        page_num = find_page(start, page_map)
        chunks.append({"text": chunk_str, "page": page_num})
        start += chunk_size - overlap
    return chunks