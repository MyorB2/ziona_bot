import os
import re
import warnings
import emoji
import nltk
import pandas as pd
# from google.colab import drive
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")


def get_statistics(combined_df, column):
    # Explode the "extracted_categories" column to count individual categories
    exploded_categories = combined_df[column].explode()
    # Get the value counts for each category
    category_counts = exploded_categories.value_counts()
    # Print the results
    print(f"\n########################-Statistics for {column} column-##################\n{category_counts}\n")


def extract_text(text):
    # Extracts text between 'Published: YYYY-MM-DD' and 'X Comments' patterns.
    pattern_fb = r'Published: \d{4}-\d{2}-\d{2}\n(.*?)\n\d+ Comments'
    match_fb = re.search(pattern_fb, text, re.DOTALL)
    if match_fb:
        text = match_fb.group(1).strip()

    # Extract the text after "- level_\d{1,}:
    pattern_level = r'- level_\d{1,}:\s*(.*)'
    match_level = re.search(pattern_level, text, re.DOTALL)

    if match_level:
        text = match_level.group(1).strip()

    text = re.sub(r"\d{4}-\d{2}-\d{2}, \d{2}:\d{2}, by [A-Za-z0-9_.\s]+ \(.? up votes, .? down votes\):", "", text)
    text = re.sub(
        r'(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}, by [A-Za-z0-9_.\s]+ \(.? up votes, .? down votes\):|'
        r'\d{1,2} [A-Za-z]+, by [A-Za-z0-9_.\s]+ \(.*? up votes\):)',
        '', text)

    # "February", "24 February", "4 days ago", "2021-02-16, 14:56"
    text = re.sub(r'"\d{1,2} \w+"|"\d{1,2} Wo\."|\d{4}-\d{2}-\d{2}, \d{2}:\d{2}|"4 days ago"', '', text)
    # "by Name"
    text = re.sub(r'by [\w\s.-]+', '', text)
    # "- level_2:", "- level_1:"
    text = re.sub(r'- level_\d+', '', text)
    # "(5 up votes)", "(130 up votes, 53 down votes)"
    text = re.sub(r'\(\d+ up votes(, \d+ down votes)?\)', '', text)

    return text


def extract_categories(text, categories):
    found_categories = []
    for category in categories:
        # Check if category is not NaN and is present in the text
        if isinstance(category, str) and category in str(text):
            found_categories.append(category)
    return found_categories


# Function to sort a dictionary by keys
def sort_dict_by_keys(d):
    return dict(sorted(d.items()))


def remove_empty_hierarchies(hierarchy_dict):
    for key, value in list(hierarchy_dict.items()):
        if len(value) == 0:
            hierarchy_dict.pop(key)
    if len(hierarchy_dict) == 0:
        return None
    return hierarchy_dict


# Function to extract hierarchy and create a dictionary
def create_hierarchy_dict(label_text, categories):
    # check if the label is hierarchical
    if label_text[0].isdigit():
        # Split the text by backslash
        parts = label_text.split("\\")
        hierarchy_dict = {}
        codes_for_current_key = []
        current_key = ""
        prev_key = None

        for i in range(len(parts)):
            if parts[i][0].isdigit():
                # Match the hierarchy pattern: a digit followed by ")"
                if match := re.match(r"^(\d{1})\)\s*(.*)", parts[i]):
                    hierarchy = match.group(1)
                    current_key = hierarchy
                    codes_for_current_key = []
                    # Ignore Semiotic category
                    if current_key != '7':
                        hierarchy_dict[current_key] = codes_for_current_key
                # Match the hierarchy pattern: a digit followed by a letter followed by ")"
                elif match := re.match(r"^(\d{1}[a-z]{1})\)\s*(.*)", parts[i]):
                    hierarchy = match.group(1)
                    current_key = hierarchy[0]
                    codes_for_current_key = []
                    hierarchy_dict[current_key] = codes_for_current_key
                # Match the hierarchy pattern: a digit followed by a letter but not followed by ")" and followed by at list one space
                elif match := re.match(r"^(\d{1}[a-z]{0,1})\s?(.*)", parts[i]):
                    hierarchy = match.group(1)
                    current_key = hierarchy[0]
                    codes_for_current_key = []
                    hierarchy_dict[current_key] = codes_for_current_key
            else:
                # Append to the corresponding hierarchy
                codes_for_current_key.extend(extract_categories(parts[i], categories))
                if current_key == '':
                    print("### there is a null key in hierarchy ### ", label_text, parts)
                hierarchy_dict[current_key] = codes_for_current_key
        return hierarchy_dict
    else:
        return {'8': extract_categories(label_text, categories)}


def merge_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in merged_dict:
                merged_dict[k] = []
            if isinstance(v, list):
                merged_dict[k].extend(v)
            else:
                merged_dict[k].append(v)
            merged_dict[k] = list(set(merged_dict[k]))
    return merged_dict


def extract_sup_categories(hierarchical_dict):
    return list(hierarchical_dict.keys())


def extract_sub_categories(hierarchical_dict):
    values = []
    for value in hierarchical_dict.values():
        values.extend(value)
    return list(sorted(set(values)))


def extract_bin_categories(category_list, weighted_categories):
    sum_weights = 0

    for cat in category_list:
        # Filter the DataFrame to find the category
        category_row = weighted_categories[weighted_categories['category_code'] == cat]

        if not category_row.empty:
            weight = category_row['weight'].values[0]
            sum_weights += weight
        else:
            print(f"Category '{cat}' not found in weighted_categories DataFrame.")

    # Calculate the threshold
    threshold = len(category_list) / 2
    result = 1 if sum_weights >= threshold else 0
    return result


# Generate one-hot encoding for a list of categories
def encode_subcategories(categories_list, categories):
    one_hot = [1 if category in categories_list else 0 for category in categories]
    return one_hot


# Generate one-hot encoding for a list of categories
def encode_categories(categories_list, num_unique_categories):
    one_hot = [0] * num_unique_categories
    for category in categories_list:
        one_hot[int(category) - 1] = 1
    return one_hot


def add_categories_columns(combined_df, categories, weighted_categories):
    # Apply the function to the 'Code' column
    combined_df['hierarchy_dict'] = combined_df['Code'].apply(lambda x: create_hierarchy_dict(x, categories))

    # Group by the text column and concatenate the labels
    combined_df = combined_df.groupby('extracted_text', as_index=False).agg({
        'hierarchy_dict': merge_dicts
    })

    combined_df['hierarchy_dict'] = combined_df['hierarchy_dict'].apply(remove_empty_hierarchies)
    combined_df = combined_df.dropna(subset="hierarchy_dict")
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['hierarchy_dict'] = combined_df['hierarchy_dict'].apply(sort_dict_by_keys)

    combined_df['extracted_categories'] = combined_df['hierarchy_dict'].apply(extract_sup_categories)
    combined_df['extracted_subcategories'] = combined_df['hierarchy_dict'].apply(extract_sub_categories)
    combined_df['extracted_bin_categories'] = combined_df['extracted_subcategories'].apply(
        lambda x: extract_bin_categories(x, weighted_categories))
    print("finished creating extracted_categories and Co.")

    print("start creating one_hot_sup_cat")
    combined_df['one_hot_sup_cat'] = combined_df['extracted_categories'].apply(lambda x: encode_categories(x, 8))
    print("start creating one_hot_sub_cat")
    combined_df['one_hot_sub_cat'] = combined_df['extracted_subcategories'].apply(
        lambda x: encode_subcategories(x, categories))
    print("start creating one_hot_bin_cat")
    combined_df['one_hot_bin_cat'] = combined_df['extracted_bin_categories']
    return combined_df


def remove_non_english_characters(text):
    # Keep emojis, digits, whitespaces, and special characters
    return ''.join(c for c in text if emoji.is_emoji(c) or c.isalnum() or c.isspace() or re.match(r"[^\x00-\x7F]", c))


def split_hashtag(hashtag):
    """Split hashtags based on CamelCase and underscores."""
    hashtag = re.sub(r'#', " ", hashtag)  # Remove '#' symbol
    hashtag = hashtag.replace("_", " ")  # Replace underscores with spaces
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag)  # Split CamelCase
    return words


def replace_hashtags(text):
    """Find hashtags and replace them with split words."""
    return re.sub(r"#\w+", lambda match: split_hashtag(match.group()), text)


def split_mentions(hashtag):
    """Split hashtags based on CamelCase and underscores."""
    hashtag = re.sub(r'@', " ", hashtag)  # Remove '#' symbol
    hashtag = hashtag.replace("_", " ")  # Replace underscores with spaces
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag)  # Split CamelCase
    return words


def replace_mentions(text):
    """Find hashtags and replace them with split words."""
    return re.sub(r"@\w+", lambda match: split_mentions(match.group()), text)


def clean_text(text):
    if text is None or not isinstance(text, str):
        return None
    stop_words = set(stopwords.words('english'))

    text = text.split('\n')
    text = " ".join(text)
    # text = re.sub(r'\(\(\(.*\)\)\)','devil Israel',text)  # echo sign
    text = re.sub(r'https?://[^\s]+', ' ', text)  # Remove URLs
    text = re.sub(r'www[^\s]+', ' ', text)  # Remove URLs
    text = replace_hashtags(text)  # Remove hashtags
    text = replace_mentions(text)  # Remove mentions

    # Regex to match the pattern (Bild: <emoji>)
    pattern = r"\(Bild:\s([^\)]+)\)"
    # Custom mapping for a specific emoji
    custom_mapping = {"(Bild: 🍉)": "(Bild: 🇵🇸)"}
    # Replace specific emoji with custom mapping
    for emoji_bild, replacement in custom_mapping.items():
        text = text.replace(emoji_bild, replacement)
    # Replace the (Bild: <emoji>) with just the emoji
    text = re.sub(pattern, r"\1", text)
    # text = emoji.demojize(text)  # text with meaning of emojis "thumbs_up_medium_skin_tone"

    # Regex to remove non-English letters while leaving digits, special characters, whitespace, and emoji
    text = remove_non_english_characters(text)
    # text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s\W]", "", text)  # Remove extra spaces

    text = " ".join(word for word in word_tokenize(text.lower()) if word not in stop_words)
    text.strip()

    if text == "" or text == " ":
        text = None
    return text


def clean_url(text):
    # Regex to extract site name and page path
    match = re.search(r'^(?:https?://)?(?:www\.)?([^/]+?)(?:\.[a-z]+)?(/.*)?$', text)

    if match:
        site_name = match.group(1).split('.')[0]  # Extract only the main domain (ignore subdomains)
        page_path = match.group(2) if match.group(2) else '/'  # Extract page path or default to '/'
        page_path = " ".join(page_path.split('/'))
        text = site_name + page_path
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    if text == "" or text == " ":
        text = None
    return text


def clean_extracted_text(combined_df, categories, as_categories):
    lc1a_indexes = combined_df.index[combined_df['extracted_subcategories'].apply(lambda x: 'LC1a' in x)].tolist()
    lc1a_subset = combined_df.loc[lc1a_indexes].copy()
    lc1a_subset["clean_extracted_text"] = lc1a_subset["extracted_text"].apply(clean_url)
    combined_df['clean_extracted_text'] = combined_df['extracted_text']
    combined_df.loc[lc1a_indexes, 'clean_extracted_text'] = lc1a_subset['clean_extracted_text']

    combined_df["clean_extracted_text"] = combined_df["clean_extracted_text"].apply(clean_text)
    combined_df = combined_df.dropna(subset=["clean_extracted_text"])
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def preprocess_dataframe():
    # # if using google colab
    # drive.mount('/content/drive')
    # dataset_uk_path = '/content/drive/MyDrive/Project/Datasets/UK'
    # guidebook_path = "/content/drive/MyDrive/Project/Datasets/Guidebook.csv"
    # guidebook_weights_path = "/content/drive/MyDrive/Project/Datasets/Guidebook_weights.xlsx"

    dataset_uk_path = '../resources/Guidebook_weights.xlsx'
    guidebook_path = '../resources/Guidebook_weights.xlsx'
    guidebook_weights_path = '../resources/Guidebook_weights.xlsx'

    # Create an empty list to store dataframes
    all_dataframes = []
    df_guidebook = pd.read_csv(guidebook_path)
    weighted_categories = pd.read_excel(guidebook_weights_path)

    # Iterate over files in the directory
    for filename in os.listdir(dataset_uk_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(dataset_uk_path, filename)
            try:
                df = pd.read_csv(filepath)
                all_dataframes.append(df)
                # print(f"Successfully read: {filename}")
            except pd.errors.ParserError as e:
                print(f"Error reading {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while reading {filename}: {e}")

    # Concatenate all dataframes into a single dataframe
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df[['Segment', 'Code']]
        print("All CSV files read and concatenated successfully!")
        # To see the output, run the code.
        # print(combined_df.head()) # You can uncomment this to display the first few rows
    else:
        print("No CSV files found in the specified directory or errors occurred during reading.")
        return None, None

    categories = df_guidebook["Abbreviation/Label"].unique()
    categories = categories[2:]
    unwanted_categories = ['FL', 'MMD', 'MMD1', 'MMD2', 'MMD3', 'Image', 'Video/GIF']
    categories = [c for c in categories if c not in unwanted_categories]

    combined_df['extracted_text'] = combined_df["Segment"].apply(extract_text)
    combined_df = combined_df.dropna(subset="extracted_text")
    combined_df.reset_index(drop=True, inplace=True)

    combined_df = add_categories_columns(combined_df, categories, weighted_categories)

    get_statistics(combined_df, 'extracted_categories')
    get_statistics(combined_df, 'extracted_subcategories')
    get_statistics(combined_df, 'extracted_bin_categories')

    combined_df_cleaned = combined_df.copy()
    combined_df_cleaned = clean_extracted_text(combined_df_cleaned, categories, weighted_categories)
    combined_df_cleaned.to_excel("combined_df_cleaned.xlsx")

    get_statistics(combined_df_cleaned, 'extracted_categories')
    get_statistics(combined_df_cleaned, 'extracted_subcategories')
    get_statistics(combined_df_cleaned, 'extracted_bin_categories')

    return combined_df_cleaned, categories
