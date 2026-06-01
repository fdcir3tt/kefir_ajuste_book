import sys, os

# Ruta absoluta basada en la ubicación de este conf.py
here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(here, '../../src/kefir_ajuste/src'))
sys.path.insert(0, os.path.join(here, '../../src/kefir_ajuste/scripts'))

print("=== conf.py cargado ===")
print(sys.path[:3])