from datasets import load_dataset, Dataset 
import json 

def remove_until_period(paragraph):
    paragraph_raw = paragraph
    while True:
        if paragraph == '':
            print('Error!')
            print(paragraph_raw)
            return 0
        if paragraph[-1] == '.' or paragraph[-1] == '!' or paragraph[-1] == '?':
            return paragraph
        else:
            paragraph = paragraph[:-1]

if __name__ == '__main__':
    seed = 2024 
    dataset_name = 'c4'
    dataset_config_name = 'realnewslike'
    dataset = load_dataset(dataset_name, dataset_config_name, split="train", streaming=True)

    dataset = dataset.shuffle(seed=seed)
    ds_iterator = iter(dataset)
    # Sample 1000 examples
    num_samples_total = 1000
    num_samples = 0
    sampled_dataset = []
    while num_samples < num_samples_total:
        example = next(ds_iterator)
        text = example['text']
        if len(text.split()) > 250 and '[word' not in text:
            sampled_dataset.append(text)
            num_samples += 1

    # Calculate average number of words
    total_words = sum(len(sample.split()) for sample in sampled_dataset)
    average_words = total_words / num_samples
    print("Average number of words:", average_words)

    # Remove last 200 tokens and create sampled_dataset_train
    sampled_dataset_train = [sample.rsplit(' ', maxsplit=200)[0] for sample in sampled_dataset]
    #sampled_dataset_train = [remove_until_period(sample) for sample in sampled_dataset_train]
    total_words = sum(len(sample.split()) for sample in sampled_dataset_train)
    average_words = total_words / num_samples
    print("Average number of words after:", average_words)

    # Save sampled_dataset to JSON
    with open('data/C4/sampled_dataset.json', 'w') as f:
        json.dump(sampled_dataset, f)

    # Save sampled_dataset_train to JSON
    with open('data/C4sampled_dataset_train.json', 'w') as f:
        json.dump(sampled_dataset_train, f)
    ...

