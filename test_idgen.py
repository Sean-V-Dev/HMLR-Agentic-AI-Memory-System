from hmlr.memory.id_generator import IDGenerator

gen = IDGenerator()
print(f'✅ IDGenerator imported successfully')
print(f'Generated dossier ID: {gen.generate_id("dos")}')
print(f'Generated fact ID: {gen.generate_id("fact")}')
print(f'Generated provenance ID: {gen.generate_id("prov")}')
