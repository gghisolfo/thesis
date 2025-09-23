import pickle
import sys

# # Prende il percorso del file dalla riga di comando
# file_path = sys.argv[1]

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print(data)


import pickle
import sys

# Prende il percorso del file dalla riga di comando
file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Stampa solo le prime 15 righe, o meno se il dataset è più piccolo
for i, row in enumerate(data):
    if i >= 2:
        break
    print(row)
