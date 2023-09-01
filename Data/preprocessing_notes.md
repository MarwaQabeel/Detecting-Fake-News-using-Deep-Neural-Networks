# Identify duplicates across two files 

```
comm -12 <(sort True.csv) <(sort Fake.csv)
```
 
# Identify duplicate rows within a file

```
cat True.csv | sort | uniq -d | wc -l
cat Fake.csv | sort | uniq -d | wc -l
```


# Removing exactly duplicate rows from within True or within Fake

```
awk '!a[$0]++' True.csv > True_cleaned.csv
awk '!a[$0]++' Fake.csv > Fake_cleaned.csv
```
