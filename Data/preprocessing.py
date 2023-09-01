import csv
from collections import Counter
fake = 'Fake.csv'
true = 'True.csv'
for file in (fake, true):
    with open(file) as fd:
        reader=csv.reader(fd)
        rows = list(reader)
        rows.pop(0)
        print (f"\n{file} has {len(rows)} rows")
        bad_records = [index for index, row in enumerate(rows) if len(row)<1]
        if len(bad_records)>0:
            print(f"For some reason, these {len(bad_records)} rows from {file} have no fields.\
 I think they might getting parsed properly. Maybe they should be removed?")
            print(bad_records)

        titles = [row[0].strip() for index, row in enumerate(rows) if len(row)>0]
        counts = dict(Counter(titles))
        duplicates = {key:value for key, value in counts.items() if value > 1}
        print(f"\n\n{file} has {len(duplicates)} duplicate titles")

        unique_titles = list(set(titles))
        new_title = f"titles_only_{file}"

        print(type(unique_titles))
        print(f"Creating {new_title} with {len(unique_titles)} titles.")
        with open(new_title, 'w', newline='\n') as f:
            writer = csv.writer(f)
            for title in unique_titles:
                writer.writerow([title])
