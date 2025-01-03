from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()

    # Converting ligatured "f" and "i" to non-ligatured
    text = text.replace("ﬁ", "fi")
    
    return text

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )

    text_chunks = text_splitter.split_text(text)
    return text_chunks


if __name__ == "__main__":

    pdf_path = "test-files/rp.pdf"
    text = extract_text_from_pdf(pdf_path)


    print(text)