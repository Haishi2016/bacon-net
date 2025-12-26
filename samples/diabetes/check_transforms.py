import json

data = json.load(open('diabetes_tree_structure.json'))

# Find features with peak transformation
peaks = [i for i, f in enumerate(data['features']) if f['transformation'] == 'peak']
print(f'Peak transformations at positions: {peaks}')
print()

if peaks:
    for i in peaks[:5]:
        feat = data['features'][i]
        print(f"Pos {i}: {feat['original_name']} -> {feat['display_name']}")
        print(f"  Transformation: {feat['transformation']}")
        print(f"  Parameters: {feat.get('parameters', {})}")
        print()
