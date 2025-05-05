import pandas as pd
import spacy
import random

def augment_pref_data(
    input_csv: str,
    output_csv: str,
    n_samples: int = 10,
    seed: int = 42
):
    random.seed(seed)
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(input_csv)

    augmented = []
    for idx, row in df.iterrows():
        prompt = row['prompt']
        chosen_text = str(row['chosen']).strip()
        reject_text = str(row['rejected']).strip()

        if not chosen_text or not reject_text:
            continue

        chosen_sents = [sent.text.strip()
                        for sent in nlp(chosen_text).sents
                        if sent.text.strip()]
        reject_sents = [sent.text.strip()
                        for sent in nlp(reject_text).sents
                        if sent.text.strip()]

        if not chosen_sents:
            chosen_sents = [chosen_text]
        if not reject_sents:
            reject_sents = [reject_text]

        chosen_samples = (random.sample(chosen_sents, n_samples)
                          if len(chosen_sents) >= n_samples
                          else random.choices(chosen_sents, k=n_samples))
        reject_samples = (random.sample(reject_sents, n_samples)
                          if len(reject_sents) >= n_samples
                          else random.choices(reject_sents, k=n_samples))

        for c_sent, r_sent in zip(chosen_samples, reject_samples):
            augmented.append({
                'prompt': prompt,
                'chosen': c_sent,
                'rejected': r_sent
            })

    aug_df = pd.DataFrame(augmented)
    aug_df = aug_df.drop_duplicates(subset=['prompt', 'chosen', 'rejected'])
    aug_df.to_csv(output_csv, index=False)
    print(f"Total：{len(aug_df)}，Saved to {output_csv}")

if __name__ == "__main__":
    augment_pref_data(
        input_csv="hhrlhf_rm_test.csv",
        output_csv="hhrlhf_rm_test_sent.csv",
        n_samples=10,
        seed=123
    )
