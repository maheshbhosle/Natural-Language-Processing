#Cleaning punctutions

from string import punctuation

# def remove_punctuation(text):
#     letters = []
#     for character in text:
#         if character not in punctuation:
#             letters.append(character)
#
#     return ''.join(letters)

remove_punctuation = lambda text : ''.join([character for character in text if character not in punctuation])


print(punctuation)
question = "T.E.'s don't drink tea! Do you?"
clean_question = remove_punctuation(question)
print(question)
print(clean_question)
