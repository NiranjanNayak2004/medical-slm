import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path


OAI_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"

SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/esearch.fcgi"
)

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "medical_corpus.txt"

EMAIL = "niranjannayak1209@gmail.com"
TOOL = "MedLensSLM"


def search_pmc(query, retmax=50):

    params = {
        "db": "pmc",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


def fetch_article(pmc_id):

    params = {
        "verb": "GetRecord",
        "identifier": (
            "oai:pubmedcentral.nih.gov:"
            + str(pmc_id)
        ),
        "metadataPrefix": "pmc"
    }

    response = requests.get(
        OAI_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.text


def extract_text(xml_text):

    root = ET.fromstring(xml_text)

    paragraphs = []

    # Ignore XML namespaces by checking tag ending
    for element in root.iter():

        if element.tag.endswith("}p"):

            text = "".join(
                element.itertext()
            ).strip()

            if text:
                paragraphs.append(text)

    return "\n".join(paragraphs)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("       MedLens Medical Corpus Builder")
    print("=" * 60)

    query = (
        "(medicine OR medical OR clinical OR disease)"
        " AND open access[filter]"
    )

    print()
    print("Searching PMC...")

    article_ids = search_pmc(
        query,
        retmax=50
    )

    print(
        "Articles found:",
        len(article_ids)
    )

    all_articles = []

    for index, pmc_id in enumerate(
        article_ids,
        1
    ):

        try:

            print(
                f"[{index}/{len(article_ids)}]"
                f" Downloading PMC{pmc_id}"
            )

            xml_text = fetch_article(
                pmc_id
            )

            article_text = extract_text(
                xml_text
            )

            if article_text:

                all_articles.append(
                    article_text
                )

                print(
                    "  Characters:",
                    len(article_text)
                )

            else:

                print(
                    "  No paragraph text found"
                )

            # Keep requests controlled
            time.sleep(0.5)

        except Exception as error:

            print(
                "  Skipped:",
                pmc_id,
                error
            )

    corpus = "\n\n".join(
        all_articles
    )

    OUTPUT_FILE.write_text(
        corpus,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("Corpus created")
    print("=" * 60)

    print(
        "Articles processed:",
        len(all_articles)
    )

    print(
        "Characters:",
        len(corpus)
    )

    print(
        "Approx. words:",
        len(corpus.split())
    )

    print(
        "Output:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()