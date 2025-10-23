import json
import re
import pickle
from typing import Dict, List, Any


def extract_rules_from_individual(individual) -> Dict[str, Any]:
    """
    Estrae le regole da un oggetto Individual e le converte in formato JSON.
    
    Args:
        individual: Oggetto Individual contenente gli oggetti e le regole
        
    Returns:
        Dizionario con le regole estratte in formato strutturato
    """
    rules_data = {
        "metadata": {
            "total_objects": len(individual.object_dict),
            "total_rules": sum(1 for rules in individual.rules.values() if rules)
        },
        "objects": []
    }
    
    for obj_id, obj in individual.object_dict.items():
        # Prendi le regole da individual.rules invece che da obj.rules
        obj_rules = individual.rules.get(obj_id, [])
        if not obj_rules:
            continue
            
        obj_data = {
            "object_id": obj_id,
            "object_names": obj.names if hasattr(obj, 'names') else [],
            "last_properties": extract_properties(obj),
            "rules": []
        }
        
        # Le regole sono in una lista, non un dizionario
        for rule in obj_rules:
            rule_data = extract_rule_info(rule)
            if rule_data:
                obj_data["rules"].append(rule_data)
        
        if obj_data["rules"]:
            rules_data["objects"].append(obj_data)
    
    return rules_data


def extract_properties(obj) -> Dict[str, Any]:
    """Estrae le proprietà dell'oggetto."""
    props = {}
    if hasattr(obj, 'last_properties'):
        for prop in obj.last_properties:
            prop_name = prop.__class__.__name__
            prop_value = prop.value if hasattr(prop, 'value') else str(prop)
            props[prop_name] = prop_value
    return props


def extract_rule_info(rule) -> Dict[str, Any]:
    """
    Estrae informazioni da una regola.
    
    Args:
        rule: Oggetto regola da analizzare
        
    Returns:
        Dizionario con le informazioni della regola
    """
    rule_info = {
        "causes": [],
        "effects": [],
        "delay_frames": 0
    }
    
    # Estrai le cause (eventi trigger)
    if hasattr(rule, 'causes'):
        rule_info["causes"] = [str(cause) for cause in rule.causes]
    
    # Estrai gli effetti
    if hasattr(rule, 'effects'):
        for effect in rule.effects:
            effect_data = parse_effect(effect)
            if effect_data:
                rule_info["effects"].append(effect_data)
    
    # Estrai il delay
    if hasattr(rule, 'delay'):
        rule_info["delay_frames"] = rule.delay
    
    return rule_info


def parse_effect(effect) -> Dict[str, Any]:
    """
    Analizza un effetto e ne estrae i parametri.
    
    Args:
        effect: Stringa o oggetto effetto
        
    Returns:
        Dizionario con i parametri dell'effetto
    """
    effect_str = str(effect)
    
    # Pattern per estrarre: property(i+1) = multiplier * property(i) + constant
    pattern = r'(\w+)\(i\+1\)\s*=\s*([-\d.]+)\s*\*\s*\w+\(i\)\s*\+\s*([-\d.]+)'
    match = re.search(pattern, effect_str)
    
    if match:
        return {
            "property": match.group(1),
            "multiplier": float(match.group(2)),
            "constant": float(match.group(3)),
            "formula": effect_str
        }
    
    return {
        "raw": effect_str
    }


def load_and_extract_rules(filepath: str) -> Dict[str, Any]:
    """
    Carica il file pickle ed estrae le regole.
    
    Args:
        filepath: Percorso del file pickle
        
    Returns:
        Dizionario con le regole estratte
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Il file contiene un dizionario con chiave 0
    if isinstance(data, dict) and 0 in data:
        individual = data[0]
        return extract_rules_from_individual(individual)
    
    return {"error": "Formato file non riconosciuto"}


def save_rules_to_json(rules_data: Dict[str, Any], output_path: str):
    """
    Salva le regole estratte in un file JSON.
    
    Args:
        rules_data: Dizionario con le regole
        output_path: Percorso del file JSON di output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    
    print(f"Regole salvate in: {output_path}")


def print_rules_summary(rules_data: Dict[str, Any]):
    """Stampa un riepilogo delle regole estratte."""
    print("\n" + "="*60)
    print("RIEPILOGO REGOLE ESTRATTE")
    print("="*60)
    print(f"Oggetti totali: {rules_data['metadata']['total_objects']}")
    print(f"Regole totali: {rules_data['metadata']['total_rules']}")
    print("\n")
    
    for obj in rules_data['objects']:
        print(f"Oggetto ID {obj['object_id']}: {obj['object_names'][0] if obj['object_names'] else 'N/A'}")
        print(f"  Proprietà: {obj['last_properties']}")
        print(f"  Numero regole: {len(obj['rules'])}")
        
        for i, rule in enumerate(obj['rules'], 1):
            print(f"\n  Regola {i}:")
            print(f"    Cause: {', '.join(rule['causes'])}")
            print(f"    Delay: {rule['delay_frames']} frame(s)")
            print(f"    Effetti:")
            for effect in rule['effects']:
                if 'formula' in effect:
                    print(f"      - {effect['property']}: {effect['formula']}")
                else:
                    print(f"      - {effect.get('raw', 'N/A')}")
        print("\n" + "-"*60)


# Esempio di utilizzo
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python script.py <file_pickle_input> [file_json_output]")
        print("\nEsempio:")
        print("  python script.py game_data.pkl")
        print("  python script.py game_data.pkl rules_output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "rules_extracted.json"
    
    try:
        # Carica ed estrai le regole
        print(f"Caricamento file: {input_file}")
        rules = load_and_extract_rules(input_file)
        
        # Stampa riepilogo
        print_rules_summary(rules)
        
        # Salva in JSON
        save_rules_to_json(rules, output_file)
        
        print("\n✓ Estrazione completata con successo!")
        
    except FileNotFoundError:
        print(f"Errore: File '{input_file}' non trovato")
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante l'estrazione: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)