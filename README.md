# stocksAgents

Experimento de agentes para pesquisa de informações de ações. Não é recomendação de
investimento e não executa ordens financeiras.

## Configuração

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure credenciais somente por variáveis de ambiente locais. Antes de mudanças,
valide a sintaxe com `python -m compileall -q .` e registre limites, fontes e data dos
dados usados.
