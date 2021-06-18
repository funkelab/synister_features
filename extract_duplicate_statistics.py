import json

dataset = '20210609'

def extract_duplicates(statistics):
    duplicate_stats = []
    for i in range(len(statistics)):
        for j in range(i+1, len(statistics)):
            if (statistics[i]['synapse_id'] == statistics[j]['synapse_id']):
                duplicate_stats.append(statistics[i])
                duplicate_stats.append(statistics[j])

    with open(f'duplicate_statistics_{dataset}.json', 'w') as f:
        json.dump(duplicate_stats, f, indent=2)

if __name__ == "__main__":

    with open(f"synapse_statistics_{dataset}.json", 'r') as f:
        statistics = json.load(f)

    extract_duplicates(statistics)
