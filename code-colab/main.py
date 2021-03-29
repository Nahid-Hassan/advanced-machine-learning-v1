# Naive Bayes
from collections import Counter

# read normal text file
file = open('datasets/normal.txt', 'r')
text = file.read()
file.close()

# split text
text = text.lower()
text = text.split()  # return list
len_text = len(text)

# create counter dict
hist_dict_normal = Counter(text)
print(hist_dict_normal)

probability_dict_normal = {}

for key, value in hist_dict_normal.items():
    # p(Dear|Normal)
    probability_dict_normal[key] = round(value / len_text, 5)

print(probability_dict_normal)

# read spam text file
file = open('datasets/spam.txt', 'r')
text = file.read()
file.close()

# split text
text = text.lower()
text = text.split()
len_text = len(text)

# hist_dict_spam
hist_dict_spam = Counter(text)
print(hist_dict_spam)

probability_dict_spam = {}
for key, value in hist_dict_spam.items():
    # p(Dear|Spam)
    probability_dict_spam[key] = round(value / len_text, 5)

# probability_dict_spam['lunch'] = 0
print(probability_dict_spam)

# Check text "Dear Friend" is normal or spam!!!!!!!!

# p(normal) = 8 / 8 + 4
# 8 is normal 4 is spam
p_normal = 8 / (8 + 4)
p_spam = 4 / (8 + 4)

# p(normal | "dear friend")
text = 'Dear Friend Money Money'
text = text.lower()

normal = p_normal
spam = p_spam

for t in text.lower().split():
    normal *= probability_dict_normal[t]
    spam *= probability_dict_spam[t]

    print("Normal: " + str(normal) + ', Spam: ' + str(spam))

print(normal)
print(spam)

print("Is text 'Dear Friend' is normal or spam?")
print('Answer: ', end=' ')
if normal > spam:
    print("Normal Text")
elif spam > normal:
    print("Spam Text")
else:
    print("50-50")
