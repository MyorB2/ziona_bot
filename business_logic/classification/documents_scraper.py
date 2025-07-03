import ast
import pandas as pd
import re

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pdfplumber
import time

all_data = []


# 1. Wikipedia
def crawl_wikipedia_antisemitism(max_depth=2):
    base_url = "https://en.wikipedia.org/wiki/Antisemitism"
    visited = set()
    to_visit = [base_url]
    data = []

    for _ in range(max_depth):
        next_to_visit = []
        for url in to_visit:
            if url in visited:
                continue
            visited.add(url)
            print(f"[Wikipedia] {url}")
            try:
                res = requests.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                for p in soup.select("p"):
                    text = p.get_text(strip=True)
                    text = clean_wikipedia_text(text)

                    if len(text) > 50:
                        data.append({"source": "Wikipedia", "url": url, "paragraph": text})
                for a in soup.select("a[href^='/wiki/']"):
                    href = a.get("href")
                    if ':' not in href and '#' not in href:
                        full_url = urljoin("https://en.wikipedia.org", href)
                        next_to_visit.append(full_url)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error: {e}")
        to_visit = next_to_visit
    return data


### 2. ADL Hate Symbols
def crawl_adl_hate_symbols():
    base_url = "https://www.adl.org/resources/hate-symbols/search"
    data = []
    try:
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/resources/hate-symbols/']")
        for link in links:
            href = link.get("href")
            full_url = urljoin("https://www.adl.org", href)
            print(f"[ADL Symbol] {full_url}")
            try:
                page = requests.get(full_url)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "ADL_Hate_Symbols", "url": full_url, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"Symbol error: {e}")
    except Exception as e:
        print(f"ADL Search Error: {e}")
    return data


### 3. ADL No Tolerance
def crawl_adl_notolerance():
    base_url = "https://notoleranceforantisemitism.adl.org/resources"
    data = []
    try:
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='https://notoleranceforantisemitism.adl.org']")
        for link in links:
            href = link.get("href")
            if href.endswith(".pdf"):
                continue
            print(f"[ADL Resource] {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "ADL_NoTolerance", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"Resource error: {e}")
    except Exception as e:
        print(f"NoTolerance Error: {e}")
    return data


### 4. IHRA Definition
def extract_ihra_definition():
    url = "https://www.holocaustremembrance.com/resources/working-definitions-charters/working-definition-antisemitism"
    data = []
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        for p in soup.select("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                data.append({"source": "IHRA", "url": url, "paragraph": text})
    except Exception as e:
        print(f"IHRA error: {e}")
    return data


### 5. ISGAP Articles
def crawl_isgap_articles(max_articles=5):
    base_url = "https://isgap.org/articles/"
    data = []
    try:
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='https://isgap.org/article/']")
        for link in links[:max_articles]:
            href = link.get("href")
            print(f"[ISGAP] {href}")
            try:
                article = requests.get(href)
                psoup = BeautifulSoup(article.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "ISGAP", "url": href, "paragraph": text})
            except Exception as e:
                print(f"ISGAP error: {e}")
    except Exception as e:
        print(f"ISGAP base error: {e}")
    return data


# 6. PDF Extraction
def extract_paragraphs_from_pdf(pdf_path, source_name, base_url=None):
    paragraphs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for para in text.split("\n"):
                        para = para.strip()
                        if len(para) > 50:
                            paragraphs.append({
                                "source": source_name,
                                "url": base_url or pdf_path,
                                "paragraph": para
                            })
    except Exception as e:
        print(f"PDF error: {e}")
    return paragraphs


def crawl_yadvashem_articles(max_articles=5):
    data = []
    # 7.1 Yad Vashem Educational Materials
    try:
        print("[Yad Vashem] Crawling educational pages...")
        base_url = "https://www.yadvashem.org/education/educational-materials.html"
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/education/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.yadvashem.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "YadVashem", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    Yad Vashem error: {e}")
    except Exception as e:
        print(f"[Yad Vashem] Error: {e}")

    ### 7.2 Jewish Virtual Library (JVL)
    try:
        print("[JVL] Crawling antisemitism entries...")
        base_url = "https://www.jewishvirtuallibrary.org/antisemitism"
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/antisemitism/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.jewishvirtuallibrary.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "JVL", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    JVL error: {e}")
    except Exception as e:
        print(f"[JVL] Error: {e}")

    ### 7.3 AJC (American Jewish Committee)
    try:
        print("[AJC] Crawling news articles...")
        base_url = "https://www.ajc.org/news"
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/news/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.ajc.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "AJC", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    AJC error: {e}")
    except Exception as e:
        print(f"[AJC] Error: {e}")

    return data


def crawl_yad_vashem(max_articles=5):
    base_url = "https://www.yadvashem.org/education/educational-materials.html"
    data = []
    try:
        print("[Yad Vashem] Crawling educational materials...")
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/education/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.yadvashem.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "YadVashem", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    Yad Vashem error: {e}")
    except Exception as e:
        print(f"[Yad Vashem] Error: {e}")
    return data


def crawl_jvl(max_articles=5):
    base_url = "https://www.jewishvirtuallibrary.org/antisemitism"
    data = []
    try:
        print("[JVL] Crawling antisemitism articles...")
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/antisemitism/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.jewishvirtuallibrary.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "JVL", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    JVL error: {e}")
    except Exception as e:
        print(f"[JVL] Error: {e}")
    return data


def crawl_ajc(max_articles=5):
    base_url = "https://www.ajc.org/news"
    data = []
    try:
        print("[AJC] Crawling news articles...")
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/news/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.ajc.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "AJC", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    AJC error: {e}")
    except Exception as e:
        print(f"[AJC] Error: {e}")
    return data


def crawl_ushmm_holocaust(max_articles=5):
    base_url = "https://encyclopedia.ushmm.org/content/en/article/antisemitism"
    data = []
    try:
        print("[USHMM] Crawling antisemitism articles...")
        visited = set()
        to_visit = [base_url]
        for _ in range(max_articles):
            next_to_visit = []
            for url in to_visit:
                if url in visited:
                    continue
                visited.add(url)
                print(f"  - {url}")
                try:
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for p in soup.select("p"):
                        text = p.get_text(strip=True)
                        if len(text) > 50:
                            data.append({"source": "USHMM", "url": url, "paragraph": text})
                    for a in soup.select("a[href^='/content/en/article/']"):
                        href = urljoin("https://encyclopedia.ushmm.org", a.get("href"))
                        if href not in visited:
                            next_to_visit.append(href)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    USHMM error: {e}")
            to_visit = next_to_visit
    except Exception as e:
        print(f"[USHMM] Error: {e}")
    return data


def crawl_facing_history(max_articles=5):
    base_url = "https://www.facinghistory.org/resource-library"
    data = []
    try:
        print("[FacingHistory] Crawling educational content...")
        res = requests.get(base_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a[href^='/resource-library/']")
        for link in links[:max_articles]:
            href = urljoin("https://www.facinghistory.org", link.get("href"))
            print(f"  - {href}")
            try:
                page = requests.get(href)
                psoup = BeautifulSoup(page.text, 'html.parser')
                for p in psoup.select("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        data.append({"source": "FacingHistory", "url": href, "paragraph": text})
                time.sleep(0.5)
            except Exception as e:
                print(f"    FacingHistory error: {e}")
    except Exception as e:
        print(f"[FacingHistory] Error: {e}")
    return data


def crawl_ushmm_antisemitism():
    url = "https://encyclopedia.ushmm.org/content/en/article/antisemitism"
    data = []
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        for p in soup.select("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                data.append({"source": "USHMM_Antisemitism", "url": url, "paragraph": text})
    except Exception as e:
        print(f"[USHMM] Error: {e}")
    return data


if __name__ == "__main__":
    all_data += crawl_wikipedia_antisemitism(max_depth=1)
    all_data += crawl_adl_hate_symbols()
    all_data += crawl_adl_notolerance()
    all_data += extract_ihra_definition()
    all_data += crawl_isgap_articles()
    all_data += crawl_yad_vashem()
    all_data += crawl_jvl()
    all_data += crawl_ajc()
    all_data += crawl_ushmm_holocaust()
    all_data += crawl_facing_history()
    all_data += crawl_ushmm_antisemitism()

    df = pd.DataFrame(all_data)
    df = df[df['paragraph'].str.len() > 50].dropna()
    df = df.drop_duplicates(subset=['paragraph'])
    df.to_csv("antisemitism_sources_combined.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} paragraphs to antisemitism_sources_combined.csv")

    df = pd.read_csv("antisemitism_sources_combined.csv")
    print(f"Total rows before cleaning: {len(df)}")
    df = df.dropna(subset=['paragraph'])
    df = df[df['paragraph'].str.strip().str.len() > 20]
    df = df.drop_duplicates(subset=['paragraph'])

    print(f"Cleaned rows: {len(df)}")
    print(f"Unique sources: {df['source'].nunique()}")
    print(f"Example URL: {df['url'].iloc[0]}")

    df.to_csv("antisemitism_sources_clean.csv", index=False, encoding="utf-8-sig")
    print("Saved clean file: antisemitism_sources_clean.csv")
    paragraphs = df['paragraph'].tolist()


    def clean_wikipedia_text(text):
        text = re.sub(r"\[\w+]", "", text)
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    # Assuming you have the path to the file you want to upload
    file_to_upload = "/content/mini_df.csv"  # Replace with the actual path if needed
    df = pd.read_csv(file_to_upload)
    df["category_id"] = df["category_id"].apply(ast.literal_eval)
    df_filtered = df[~df["category_id"].apply(lambda x: 0 in x)]
    df_filtered.to_csv("filtered_output.csv", index=False)
