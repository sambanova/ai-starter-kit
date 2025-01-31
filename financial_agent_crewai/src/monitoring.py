import json


def main() -> None:
    # Open the JSON file
    with open('/Users/francescar/Downloads/traces_cm6i5yceu001z7qzxkvgy39n7.json', 'r') as file:
        # Load the contents of the file into a Python dictionary
        monitoring_data = json.load(file)

    duration_count = 0
    for count, item in enumerate(monitoring_data):
        for elem in item:
            if elem['start_time'] >= '2025-01-31T22:40:00Z':
                if elem['name'] == 'Crew.kickoff':
                    duration_count += int(elem['duration'])
            else:
                continue

    print(round(duration_count / 1000 / 60, 2))


if __name__ == '__main__':
    main()
