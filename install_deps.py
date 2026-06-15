import subprocess
import sys
packages = [
    'flask',
    'pymongo',
    'langchain-core==1.2.5',
    'langchain-groq',
    'langchain-community',
    'sentence-transformers',
    'faiss-cpu',
    'pydantic-core'
]
print('Using interpreter:', sys.executable)
cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'] + packages
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print('returncode=', result.returncode)
print(result.stdout)
print(result.stderr)
