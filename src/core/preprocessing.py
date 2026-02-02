# Resampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def decoding_numbers(df_original):
    df = df_original.copy()
    df = df.astype("str")
    df.columns = df.iloc[0, :].values.tolist()
    df = df.drop(0)
    if "Region" in df.columns.values:
        df["Region"] = df["Region"].replace(
            {
                "E12000001": "North East",
                "E12000002": "North West",
                "E12000003": "Yorkshire and the Humber",
                "E12000004": "East Midlands",
                "E12000005": "West Midlands",
                "E12000006": "East of England",
                "E12000007": "London",
                "E12000008": "South East",
                "E12000009": "South West",
                "W92000004": "Wales",
            }
        )
    if "Residence Type" in df.columns.values:
        df["Residence Type"] = df["Residence Type"].replace(
            {"C": "Communal", "H": "Not communal"}
        )
    if "Family Composition" in df.columns.values:
        df["Family Composition"] = df["Family Composition"].replace(
            {
                "1": "Not in a family",
                "2": "Married/same-sex civil" + "partnership couple family",
                "3": "Cohabiting couple family",
                "4": "Lone parent family (male head)",
                "5": "Lone parent family (female head)",
                "6": "Other related family",
                "-9": "No code required",
            }
        )
    if "Population Base" in df.columns.values:
        df["Population Base"] = df["Population Base"].replace(
            {
                "1": "Usual resident",
                "2": "Student living away from home during" + "term-time",
                "3": "Short-term resident",
            }
        )
    if "Sex" in df.columns.values:
        df["Sex"] = df["Sex"].replace({"1": "Male", "2": "Female"})
    if "Marital Status" in df.columns.values:
        df["Marital Status"] = df["Marital Status"].replace(
            {
                "1": "Single (never married or never "
                + "registered a same-sex civil partnership)",
                "2": "Married or in a registered same-sex " + "civil partnership",
                "3": "Separated but still legally married or"
                + "separated but still legally in a "
                + "same-sex civil partnership",
                "4": "Divorced or formerly in a same-sex "
                + "civil partnership which is now "
                + "legally dissolved",
                "5": "Widowed or surviving partner from a"
                + "same-sex civil partnership",
            }
        )
    if "Student" in df.columns.values:
        df["Student"] = df["Student"].replace({"1": "Yes", "2": "No"})
    if "Country of Birth" in df.columns.values:
        df["Country of Birth"] = df["Country of Birth"].replace(
            {"1": "UK", "2": "Non UK", "-9": "No Code required"}
        )
    if "Health" in df.columns.values:
        df["Health"] = df["Health"].replace(
            {
                "1": "Very good health",
                "2": "Good health",
                "3": "Fair health",
                "4": "Bad health",
                "5": "Very bad health",
                "-9": "No code required",
            }
        )
    if "Ethnic Group" in df.columns.values:
        df["Ethnic Group"] = df["Ethnic Group"].replace(
            {
                "1": "White",
                "2": "Mixed",
                "3": "Asian and Asian British",
                "4": "Black or Black British",
                "5": "Chinese or Other ethnic group",
                "- 9": "No code required",
            }
        )
    if "Religion" in df.columns.values:
        df["Religion"] = df["Religion"].replace(
            {
                "1": "No religion",
                "2": "Christian",
                "3": "Buddhist",
                "4": "Hindu",
                "5": "Jewish",
                "6": "Muslim",
                "7": "Sikh",
                "8": "Other religion",
                "9": "Not stated",
                "-9": "No code required",
            }
        )
    if "Economic Activity" in df.columns.values:
        df["Economic Activity"] = df["Economic Activity"].replace(
            {
                "1": "Economically active: Employee",
                "2": "Economically active: Self-employed",
                "3": "Economically active: Unemployed",
                "4": "Economically active: Full-time student",
                "5": "Economically inactive: Retired",
                "6": "Economically inactive: Student",
                "7": "Economically inactive: Looking " + "after home or family",
                "8": "Economically inactive: Long-term" + " sick or disabled",
                "9": "Economically inactive: Other",
                "-9": "No code required",
            }
        )
    if "Occupation" in df.columns.values:
        df["Occupation"] = df["Occupation"].replace(
            {
                "1": "Managers, Directors and Senior Officials",
                "2": "Professional Occupations",
                "3": "Associate Professional and Technical Occupations",
                "4": "Administrative and Secretarial Occupations",
                "5": "Skilled Trades Occupations",
                "6": "Caring, Leisure and Other Service Occupations",
                "7": "Sales and Customer Service Occupations",
                "8": "Process, Plant and Machine Operatives",
                "9": "Elementary Occupations",
                "-9": "No code required",
            }
        )
    if "Industry" in df.columns.values:
        df["Industry"] = df["Industry"].replace(
            {
                "1": "Agriculture, forestry and fishing",
                "2": "Mining and quarrying; Manufacturing; "
                + "Electricity, gas, steam and air "
                + "conditioning system; Water supply",
                "3": "Construction",
                "4": "Wholesale and retail trade; Repair "
                + "of motor vehicles and motorcycles",
                "5": "Accommodation and food service activities",
                "6": "Transport and storage; Information and" + " communication",
                "7": "Financial and insurance activities;" + "Intermediation",
                "8": "Real estate activities; Professional, "
                + "scientific and technical activities;"
                + " Administrative and support service "
                + "activities",
                "9": "Public administration and defence;"
                + "compulsory social security",
                "10": "Education",
                "11": "Human health and social work activities",
                "12": "Other community, social and personal"
                + " service activities; Private households"
                + "employing domestic staff; Extra-"
                + "territorial organisations and bodies",
                "-9": "No code required",
            }
        )
    if "Age" in df.columns.values:
        df["Age"] = df["Age"].replace(
            {
                "1": "0 to 15",
                "2": "16 to 24",
                "3": "25 to 34",
                "4": "35 to 44",
                "5": "45 to 54",
                "6": "55 to 64",
                "7": "65 to 74",
                "8": "75 and over",
            }
        )
    if "Hours worked per week" in df.columns.values:
        df["Hours worked per week"] = df["Hours worked per week"].replace(
            {
                "1": "Part-time: 15 or less hours",
                "2": "Part-time: 16 to 30 hours",
                "3": "Full-time: 31 to 48 hours",
                "4": "Full-time: 49 or more hours",
                "-9": "No code required",
            }
        )
    if "Approximated Social Grade" in df.columns.values:
        df["Approximated Social Grade"] = df["Approximated Social Grade"].replace(
            {"1": "AB", "2": "C1", "3": "C2", "4": "DE", "-9": "No code required"}
        )
    return df


def train_val_test_split(features, target):
    features, features_test, target, target_test = train_test_split(
        features, target, test_size=0.1, random_state=42
    )
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=0.2, random_state=43
    )
    return (
        features_train,
        target_train,
        features_val,
        target_val,
        features_test,
        target_test,
    )


all_columns = [
    "Region",
    "Residence Type",
    "Family Composition",
    "Population Base",
    "Sex",
    "Age",
    "Marital Status",
    "Student",
    "Country of Birth",
    "Health",
    "Ethnic Group",
    "Religion",
    "Economic Activity",
    "Occupation",
    "Industry",
    "Hours worked per week",
]

smotesampler = SMOTE(random_state=42)
undersampler = RandomUnderSampler(random_state=42)
oversampler = RandomOverSampler(random_state=42)

list_resampler = [
    ("res", "passthrough"),
    ("res", smotesampler),
    ("res", undersampler),
    ("res", oversampler),
]


def all_combinations(
    list_input=[
        "Region",
        "Residence Type",
        "Family Composition",
        "Population Base",
        "Sex",
        "Age",
        "Marital Status",
        "Student",
        "Country of Birth",
        "Health",
        "Ethnic Group",
        "Religion",
        "Economic Activity",
        "Occupation",
        "Industry",
        "Hours worked per week",
    ],
):
    """Return list with all combinations of items in input list."""
    feature_combinations = []
    for possible_size_of_combinations in range(0, len(list_input)):
        new_combinations = list(
            itertools.combinations(list_input, possible_size_of_combinations)
        )
        feature_combinations = feature_combinations + new_combinations

    feat_comb = []
    for i in range(len(feature_combinations)):
        features = feature_combinations[i]
        hilfsliste = []
        for j in features:
            hilfsliste.append(j)
        feat_comb.append(hilfsliste)
    return feat_comb


def select_random_features():
    """Select random features for the pca, ohe and numerical list"""
    feat_comb = all_combinations()
    feature_list = feat_comb[random.randint(0, len(feat_comb) - 1)]
    return feature_list
