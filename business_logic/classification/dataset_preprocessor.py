import os
import warnings
import nltk
import re
import emoji
import ast
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")


class DatasetPreprocessor:
    def __init__(self, dataset_uk_path, guidebook_path, guidebook_weights_path):
        self.dataset_uk_path = dataset_uk_path
        self.guidebook_path = guidebook_path
        self.guidebook_weights_path = guidebook_weights_path
        self.mapping_dict = None

    @staticmethod
    def normalize_main_category(cat):
        # Extract only the numeric part of the main category ('4b' -> '4')
        match = re.match(r'^(\d+)', cat)
        return match.group(1) if match else cat

    def extract_hierarchy_fields(self, code_text):
        main_cats = []
        sub_cats = []

        if isinstance(code_text, str):
            parts = code_text.split('\\')
            for part in parts:
                match_main = re.match(r'^(\d+[a-zA-Z]?)\)', part)
                if match_main:
                    main_cats.append(match_main.group(1))
                # sub_matches = re.findall(r'\(?([A-Z]+\d+[a-zA-Z]?)', part)
                sub_matches = re.findall(r'\(?([A-Z]+\d+[a-zA-Z0-9]*)', part)

                sub_cats.extend(sub_matches)

            if not main_cats and not sub_cats:
                main_cats.append(code_text.strip())

        normalized_mains = sorted(set([self.normalize_main_category(c) for c in main_cats]))
        unique_subs = sorted(set(sub_cats))

        # dict: main categories <-> sub categories
        sub_map = {main: unique_subs for main in normalized_mains}

        return pd.Series({
            'normalized_main_categories': normalized_mains,
            'extracted_subcategories': unique_subs,
            'category_subcategory_map': sub_map
        })

    @staticmethod
    def clean_text(text):
        if text is None or not isinstance(text, str):
            return None

        # Remove unnecessary templates
        patterns_to_remove = [
            r'"?\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)"?,? by [^:\n]+?:',
            r'\d{1,2} [A-Za-z]+, by [^:\n]+? \(\d+ up votes(, \d+ down votes)?\):',
            r'\d{4}-\d{2}-\d{2}, \d{2}:\d{2}, by [^:\n]+? \(\d+ up votes(, \d+ down votes)?\):',
            r'\d{4}-\d{2}-\d{2}, \d{2}:\d{2}, by [^:\n]+:',
            r'"?\d+ (days?|hours?|minutes?) ago"?,? by [^:\n]+? \(\d+ up votes(, \d+ down votes)?\):',
            r'"?\d+ (days?|hours?|minutes?) ago"?,? by [^:\n]+?:',
            r'by [^:\n]+? \(\d+ up votes(, \d+ down votes)?\):',
            r'by [^:\n]+?:',
            r'- level_\d+:'
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Text cleaning
        text = text.replace('\n', ' ')
        text = re.sub(r'\(\(\(.*?\)\)\)', 'devil Israel', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\(Bild:\s*([^\)]+)\)', r'\1', text)
        text = emoji.demojize(text, language='en')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip() if text.strip() else None

    def map_subcategories_to_mapped(self, subcategories):
        if not isinstance(subcategories, list):
            return []
        mapped = [self.mapping_dict.get(sub.strip().upper(), None) for sub in subcategories]
        return sorted(set([m for m in mapped if m is not None]))

    def replace_6_with_1(lst):
        if isinstance(lst, list) and lst == [6]:
            return [1]
        return lst

    def string_to_list(x):
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return [x]
        elif isinstance(x, list):
            return x
        else:
            return [x]

    @staticmethod
    def is_binary_as(mapped_list):
        # binary mapping
        try:
            if isinstance(mapped_list, str):
                mapped_list = ast.literal_eval(mapped_list)
            if isinstance(mapped_list, list):
                mapped_list = [int(x) for x in mapped_list]
                return 0 if mapped_list == [0] else 1
        except Exception:
            pass
        return 1

    @staticmethod
    def safe_split_int_list(lst):
        result = []
        try:
            if isinstance(lst, str):
                lst = ast.literal_eval(lst)

            if isinstance(lst, list):
                for item in lst:
                    if isinstance(item, str) and ',' in item:
                        result.extend([int(i.strip()) for i in item.split(',') if i.strip().isdigit()])
                    elif isinstance(item, (int, float)) or str(item).strip().isdigit():
                        result.append(int(str(item).strip()))
            return result
        except Exception:
            return []

    @staticmethod
    def clean_and_filter_categories(lst):
        if not isinstance(lst, list):
            return []

        cleaned = []
        for i in lst:
            try:
                if isinstance(i, (int, float)) or str(i).isdigit():
                    cleaned.append(int(i))
            except:
                continue

        cleaned = list(set(cleaned))

        # If there is category 0 along with other categories --> delete
        if 0 in cleaned and len(cleaned) > 1:
            cleaned = [x for x in cleaned if x != 0]

        return sorted(cleaned)

    def preprocess_dataframe(self):
        # # if using google colab
        # drive.mount('/content/drive')
        # dataset_uk_path = '/content/drive/MyDrive/Project/Datasets/UK'
        # guidebook_path = "/content/drive/MyDrive/Project/Datasets/Guidebook.csv"
        # guidebook_weights_path = "/content/drive/MyDrive/Project/Datasets/Guidebook_weights.xlsx"
        # directory_path = '/content/drive/MyDrive/Project/Datasets/UK'

        # Create an empty list to store dataframes
        all_dataframes = []
        df_guidebook = pd.read_csv(self.guidebook_path)
        weighted_categories = pd.read_excel(self.guidebook_weights_path)

        # Create an empty list to store dataframes
        all_dataframes = []

        # Iterate over files in the directory
        for filename in os.listdir(self.dataset_uk_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.dataset_uk_path, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_dataframes.append(df)
                except pd.errors.ParserError as e:
                    print(f"Error reading {filename}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {filename}: {e}")

            # Concatenate all dataframes into a single dataframe
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df = combined_df[["Segment", "Code"]]
            print("All CSV files read and concatenated successfully!")
            # To see the output, run the code.
            # print(combined_df.head()) # You can uncomment this to display the first few rows
            return combined_df
        else:
            print("No CSV files found in the specified directory or errors occurred during reading.")
            return None

    def preprocess(self):
        # Preprocess the data and combine dataframes
        df = self.preprocess_dataframe()

        # Extract hierarchy fields and clean text
        df[['normalized_main_categories', 'extracted_subcategories', 'category_subcategory_map']] = df['Code'].apply(
            self.extract_hierarchy_fields)
        df = df[df['extracted_subcategories'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        # Apply category mapping
        guidebook_df = pd.read_csv("/content/drive/MyDrive/Project/Datasets/new_Guidebook_0905.csv")
        guidebook_df['category_code'] = guidebook_df['category_code'].astype(str).str.strip().str.upper()
        self.mapping_dict = dict(zip(guidebook_df['category_code'], guidebook_df['mapped_code']))
        df['updated_mapped_categories'] = df['extracted_subcategories'].apply(self.map_subcategories_to_mapped,
                                                                              self.mapping_dict)
        df = df[df['updated_mapped_categories'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        # Load and merge scraped tweets
        scraped_tweets = pd.read_csv("/content/drive/MyDrive/Project/Datasets/Scraped tweets/united_3.csv")
        united_3_renamed = scraped_tweets.rename(columns={'categories': 'updated_mapped_categories', 'text': 'Segment'})
        united_3_renamed['updated_mapped_categories'] = united_3_renamed['updated_mapped_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        united_3_reduced = united_3_renamed[['Segment', 'updated_mapped_categories']]
        df = pd.concat([df, united_3_reduced], ignore_index=True)

        # Apply category corrections and ensure list format
        df['updated_mapped_categories'] = df['updated_mapped_categories'].apply(self.replace_6_with_1)
        df["updated_mapped_categories"] = df["updated_mapped_categories"].apply(self.string_to_list)

        # Clean text and remove duplicates
        df['clean_text'] = df['Segment'].apply(self.clean_text)
        df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
        df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

        # Create binary category column
        df['binary_category'] = df['updated_mapped_categories'].apply(self.is_binary_as)

        # Process and clean mapped categories
        df["updated_mapped_categories"] = df["updated_mapped_categories"].apply(self.safe_split_int_list)
        df['updated_mapped_categories'] = df['updated_mapped_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['updated_mapped_categories'] = df['updated_mapped_categories'].apply(self.clean_and_filter_categories)

        # Explode and filter categories
        exploded_df = df.explode('updated_mapped_categories')
        exploded_df = exploded_df[exploded_df['updated_mapped_categories'].isin([0, 1, 2, 3, 4, 5])]

        # Count and print category distribution
        category_counts = (
            exploded_df['updated_mapped_categories']
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={'index': 'mapped_category', 'updated_mapped_categories': 'count'})
        )
