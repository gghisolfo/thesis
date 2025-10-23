# import pickle
# import sys

# # Prende il percorso del file dalla riga di comando
# file_path = sys.argv[1]

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print(data)


# import pickle
# import sys

# # Prende il percorso del file dalla riga di comando
# file_path = sys.argv[1]

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# # Stampa solo le prime 15 righe, o meno se il dataset è più piccolo
# for i, row in enumerate(data):
#     if i >= 2:
#         break
#     print(row)

import pickle
import sys
from pprint import pprint  # per una stampa più leggibile

file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("\n=== INFO SUL FILE ===")
print("Tipo dell'oggetto caricato:", type(data))

if hasattr(data, '__len__'):
    print("Lunghezza:", len(data))
else:
    print("Lunghezza: N/A")

print("\n=== CONTENUTO PARZIALE ===")
if isinstance(data, dict):
    print("Chiavi del dizionario:", list(data.keys()))
    for k, v in list(data.items())[:2]:  # mostra le prime 2 chiavi
        print(f"\nChiave: {k}\nTipo valore: {type(v)}")
        if isinstance(v, (list, dict)):
            print("Esempio contenuto:")
            pprint(v[:2] if isinstance(v, list) else list(v.items())[:2])
        else:
            print("Valore:", v)

elif isinstance(data, (list, tuple)):
    print("Primi 2 elementi:")
    for i, v in enumerate(data[:2]):
        print(f"\nElemento {i}: tipo={type(v)}")
        pprint(v)

else:
    print("Oggetto non iterabile:")
    pprint(data)

