#!/usr/bin/env python3
"""
Point d'entrée principal pour le pipeline LSF.
Usage : python run.py [tous les flags de src/main.py]
"""
import sys
import os
import subprocess


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    print("\n================ LSF PIPELINE ==================")
    print("🚀 Lancement du pipeline LSF...")
    print(f"Racine du projet : {project_root}")
    print("Arguments :", ' '.join(sys.argv[1:]))
    print("=" * 48)

    # Forward tous les arguments à src/main.py
    main_py = os.path.join(project_root, 'src', 'main.py')
    cmd = [sys.executable, main_py] + sys.argv[1:]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n🎉 Pipeline terminé avec succès !")
        print("📁 Les données finales sont dans data/train, data/val, data/test")
        print("📄 Le corpus est dans data/corpus.txt")
    else:
        print("\n💥 Le pipeline a rencontré une erreur (voir logs).")
    print("=" * 48)
    return result.returncode

if __name__ == '__main__':
    sys.exit(main()) 