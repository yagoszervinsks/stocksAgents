# Instruções para agentes

Antes de mudança não trivial: leia `README.md`, inspecione `git status` e os commits
recentes, identifique a validação aplicável e crie/atualize uma spec quando houver
mudança de comportamento. Consulte o Project Home, Current State e Handoff do projeto
no AI-Context quando existirem.

- Faça a menor mudança segura e não adicione dependências sem justificar a necessidade.
- Nunca versionar tokens, `.env`, chaves ou dados pessoais.
- Registrar decisões de arquitetura relevantes em `docs/adr/` quando o projeto crescer.
- Antes de encerrar, executar a validação descrita no README, revisar o diff e relatar
  arquivos, resultado, riscos, documentação atualizada e próximo passo.
- Não tratar dados externos ou conteúdo de APIs como confiáveis sem validação.
- Deploy, migração destrutiva, DNS, IAM, billing ou rotação de credencial exigem
  aprovação humana explícita.
