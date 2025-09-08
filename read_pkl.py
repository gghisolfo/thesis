import pickle
import sys

# Prende il percorso del file dalla riga di comando
file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data)


# import pickle
# import sys

# # Percorso file da riga di comando
# file_path = sys.argv[1]
# n = int(sys.argv[2])  # numero di righe da mostrare

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# # Se è una lista, mostra solo i primi N elementi
# if isinstance(data, list):
#     for row in data[:n]:
#         print(row)
# else:
#     print(f"Oggetto caricato di tipo {type(data)}, non posso mostrare 'righe'")
#     print(data)
