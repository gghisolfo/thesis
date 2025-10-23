import pickle
import json
import sys
import re
from pprint import pprint


# python extract_rules_from_pkl.py best_population.pkl rules.json

def safe_dir(obj):
    """Restituisce solo gli attributi 'utili' di un oggetto."""
    return [a for a in dir(obj) if not a.startswith("_")]

def parse_rule_string(rule_str):
    """
    Tenta di interpretare una regola come stringa.
    Esempio di input: 'left_arrow_pressed<class 'core.property.Pos_x'>1-2'
    """
    match = re.match(r"(\w+)<class 'core\.property\.(\w+)'>\s*([-\d.]+)?([-\d.]+)?", rule_str)
    if not match:
        return {"raw": rule_str}
    event, prop, a, b = match.groups()
    a = a or "?"
    b = b or "?"
    formula = f"{prop}(i+1) = {a}*{prop}(i) + {b}"
    return {"event": event, "property": prop, "formula": formula}

def extract_rules_from_object(obj):
    """
    Estrae le regole da un singolo oggetto, se presenti.
    Ritorna una lista di dizionari.
    """
    rules = []
    if hasattr(obj, "rules") and isinstance(obj.rules, dict):
        for rule_id, rule in obj.rules.items():
            if isinstance(rule, str):
                parsed = parse_rule_string(rule)
            elif hasattr(rule, "__dict__"):
                # tenta di accedere ai campi interni se è un oggetto complesso
                parsed = {k: str(v) for k, v in rule.__dict__.items()}
            else:
                parsed = {"raw": str(rule)}
            rules.append(parsed)
    return rules

def extract_rules_from_individual(individual):
    """
    Esamina un oggetto 'Individual' e ne estrae tutte le regole.
    """
    rules_dict = {}
    if hasattr(individual, "objects") and isinstance(individual.objects, dict):
        for name, obj in individual.objects.items():
            obj_rules = extract_rules_from_object(obj)
            if obj_rules:
                rules_dict[name] = obj_rules
    return rules_dict

def main():
    if len(sys.argv) < 3:
        print("Uso: python extract_rules_from_pkl.py input.pkl output.json")
        sys.exit(1)

    input_pkl = sys.argv[1]
    output_json = sys.argv[2]

    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    # Se è un dizionario con un singolo individuo
    if isinstance(data, dict) and len(data) == 1:
        first_val = list(data.values())[0]
        if hasattr(first_val, "objects"):
            data = first_val

    # Se l'oggetto caricato non è un Individual, prova a cercarlo
    if not hasattr(data, "objects"):
        print("❌ L'oggetto non sembra contenere un attributo 'objects'.")
        print("Tipo:", type(data))
        print("Attributi:", safe_dir(data))
        sys.exit(1)

    print(f"✅ Caricato oggetto di tipo {type(data)}")

    rules = extract_rules_from_individual(data)

    if not rules:
        print("⚠️ Nessuna regola trovata.")
    else:
        print(f"✅ Estratte regole da {len(rules)} oggetti.")
        pprint({k: len(v) for k, v in rules.items()})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Regole salvate in {output_json}")

if __name__ == "__main__":
    main()
