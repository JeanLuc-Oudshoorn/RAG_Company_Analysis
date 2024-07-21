import os
import glob
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredPowerPointLoader

# Load the environment
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize pages
pages = []

# Define the directory
directory = '../Brainchip'

# Loop over all files in the directory
for filename in glob.glob(os.path.join(directory, '*')):
    # Check the file extension
    if filename.endswith('.pdf'):
        loader = UnstructuredPDFLoader(filename)
        print(f"Processing: {filename}")
    elif filename.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(filename)
        print(f"Processing: {filename}")
    else:
        print(f"Skipping: {filename}")
        continue  # Skip files with other extensions

    # Load and split the file, then add to pages
    pages += loader.load_and_split()

print(f"Length of pages: {len(pages)}")

# Add the website
urls = [
    "https://en.wikipedia.org/wiki/BrainChip",
    "https://en.wikipedia.org/wiki/Spiking_neural_network",
    "https://en.wikipedia.org/wiki/Neuromorphic_engineering",
    "https://brainchip.com/",
    "https://brainchip.com/markets/",
    "https://brainchip.com/technology/",
    "https://investors.brainchip.com/",
    "https://brainchip.com/akida-generations/",
    "https://brainchip.com/metatf-development-environment/",
    "https://brainchip.com/introducing-tenn-revolutionizing-computing-with-an-energy-efficient-transformer-replacement/",
    "https://ieeexplore.ieee.org/abstract/document/10191569",
    "https://www.eetimes.com/brainchip-adds-edge-box-to-chip-and-ip-offerings/",
    "https://stockhead.com.au/tech/brainchips-potential-in-edge-ai-is-undeniable-but-can-sales-catch-up-with-its-cash-burn/",
    "https://www.ig.com/en-ch/news-and-trade-ideas/where-next-for-brainchip-shares--240411",
    "https://finance.yahoo.com/news/brainchip-frontgrade-gaisler-augment-space-060000155.html",
    "https://finance.yahoo.com/news/heres-why-were-not-too-010038669.html",
    "https://capital.com/branchip-brn-stock-forecast#:~:text=The%20share%20price%20could%20drop,at%20the%20end%20of%202023."
]

# collect data using selenium url loader
loader = SeleniumURLLoader(urls=urls)
pages += loader.load_and_split()

print(f"Length of pages after URL loader: {len(pages)}")

# Light preprocessing
for page in pages:
    page.page_content = str(page.page_content).replace("\\n", " ")\
                                              .replace("\\t", " ")\
                                              .replace("\n", " ")\
                                              .replace("\t", " ")

print(f"Length of pages after preprocessing: {len(pages)}")

# Create the vectorstore
vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

# Save the vectorstore
vectorstore.save_local('../vector_stores/brainchip.faiss')
